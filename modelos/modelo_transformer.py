"""
Módulo que contiene la función principal para correr el modelo
transformer para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
import pandas as pd
import numpy as np
from modelos.funciones_aux_transformer import atr_wilder_exact, agregar_indicadores, sanitize_data

# Imports "agnósticos": usa keras standalone si está; si no, tf.keras
try:
    import keras
    from keras.saving import register_keras_serializable
    from keras import layers, initializers
except Exception:  # pylint: disable=broad-except
    from tensorflow import keras
    from tensorflow.keras.utils import register_keras_serializable
    from tensorflow.keras import layers, initializers
import tensorflow as tf
import joblib

# Directorios necesarios
dir_principal = os.getcwd()
dir_modelos = os.path.join(dir_principal, "archivos_modelos")
dir_inputs = os.path.join(dir_principal, "inputs")
dir_outputs = os.path.join(dir_principal, "outputs")
dir_indtec = os.path.join(dir_principal, "data_ind_tecnicos")

# Paths necesarios
csv_spy_path = os.path.join(dir_inputs, "SPY_last10.csv")
transformer_path = os.path.join(dir_modelos, "modelo_transformer.keras")
standard_scaler_path = os.path.join(dir_modelos, "scaler_transformer.pkl")

# Variables necesarias
columnas_necesarias = ['SPY_EMA5_diff_pct', 'SPY_EMA24_diff_pct', 'SPY_EMA50_diff_pct',
       'SPY_EMA200_diff_pct', 'SPY_RSI_14', 'SPY_MACD_12_26',
       'SPY_MACD_signal_9m', 'SPY_MACD_hist', 'SPY_ATR_14', 'SPY_BB_percentB',
       'SPY_BB_z', 'SPY_stoch_k', 'SPY_stoch_d', 'SPY_volume',
       'SPY_vol_rel_24','SPY_ROC_12', 'SPY_ROC_20', 'SPY_WilliamsR_14',
       'SPY_CCI_20', 'SPY_OBV', 'SPY_MFI_14', 'SPY_Keltner_percentB',
       'SPY_volatility_std12', 'SPY_volatility_std24', 'SPY_cumret_30m']

# Crear clase PositionalEncoding
@register_keras_serializable(package="custom", name="PositionalEncoding")
class PositionalEncoding(layers.Layer):
    """
    Codificación posicional sinusoidal (Vaswani et al., 2017).
    Suma a cada token un vector fijo dependiente de su posición.

    Parámetros
    ----------
    position : int
        Longitud máxima soportada (debe ser >= longitud real de la secuencia).
    d_model : int
        Dimensión del canal de los embeddings (debe ser par).
    """
    def __init__(self, position: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        assert position > 0, "position debe ser > 0"
        assert d_model % 2 == 0, "d_model debe ser par para separar even/odd"

        self.position = int(position)
        self.d_model = int(d_model)

        # Precomputar matriz (position, d_model) con intercalado even/odd
        pos = tf.range(self.position, dtype=tf.float32)[:, tf.newaxis]     # (position, 1)
        i   = tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :]      # (1, d_model)

        # tasas de ángulo
        angle_rates = tf.pow(10000.0, - (2.0 * tf.math.floor(i / 2.0)) / tf.cast(self.d_model, tf.float32))
        angle_rads  = pos * angle_rates                                     # (position, d_model)

        # seno para dims pares, coseno para dims impares
        sin_terms = tf.math.sin(angle_rads[:, 0::2])   # (position, d_model/2)
        cos_terms = tf.math.cos(angle_rads[:, 1::2])   # (position, d_model/2)

        # Intercalar: [sin0, cos1, sin2, cos3, ...]
        pe = tf.concat(
            [tf.reshape(tf.stack([sin_terms, cos_terms], axis=-1), [self.position, -1])],
            axis=-1
        )
        # La línea anterior apila sin/cos en la última dim y luego aplana intercalando.
        # Alternativa explícita (más legible) abajo en comentarios si prefieres.

        # Registrar como constante no entrenable, añadiendo batch dim: (1, position, d_model)
        self._pos_encoding = tf.constant(pe[tf.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        """
        inputs: Tensor (batch_size, seq_len, d_model)
        retorna: inputs + PE[:seq_len]
        """
        x = inputs
        seq_len = tf.shape(x)[1]

        # Seguridad: no permitir seq_len > position
        tf.debugging.assert_less_equal(
            seq_len, self.position,
            message="seq_len excede 'position' definido en PositionalEncoding"
        )

        # Alinear dtype
        pe = tf.cast(self._pos_encoding[:, :seq_len, :], x.dtype)
        return x + pe

    def get_config(self):
        base = super().get_config()
        return {**base, "position": self.position, "d_model": self.d_model}


# Crear clase TransformerBlock
@register_keras_serializable(package="custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    """
    Bloque Transformer tipo encoder (MHA + FFN) con opción Pre-Norm.
    Implementa build() para que las subcapas tengan variables listas al cargar pesos.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        prenorm: bool = True,      # <-- recomendado para >4 bloques
        **kwargs
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.prenorm = prenorm

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.drop_attn = layers.Dropout(dropout_rate)
        self.drop_ffn  = layers.Dropout(dropout_rate)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ])

    def call(self, inputs, training=None, mask=None):
        """
        inputs: (B, T, d_model)
        mask:   (B, T) o (B, T, T) opcional (True=mantener / False=mask)
        """
        x = inputs

        # Pre-Norm
        if self.prenorm:
            y = self.norm1(x)
            attn_out = self.attn(y, y, attention_mask=mask, training=training)
            attn_out = self.drop_attn(attn_out, training=training)
            x = x + attn_out

            y = self.norm2(x)
            ffn_out = self.ffn(y, training=training)
            ffn_out = self.drop_ffn(ffn_out, training=training)
            x = x + ffn_out
            return x

        # Post-Norm (tu versión original)
        attn_out = self.attn(x, x, attention_mask=mask, training=training)
        attn_out = self.drop_attn(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x, training=training)
        ffn_out = self.drop_ffn(ffn_out, training=training)
        x = self.norm2(x + ffn_out)
        return x

    def get_config(self):
        base = super().get_config()
        base.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
            "prenorm": self.prenorm,
        })
        return base

# Cargar el modelo y el scaler
model_transformer = keras.models.load_model(
    transformer_path,
    custom_objects={"TransformerBlock": TransformerBlock,
                    "PositionalEncoding": PositionalEncoding},
    compile=False   # evita restaurar estado del optimizador
)
scaler_data = joblib.load(standard_scaler_path)

# Función main para correr el modelo transformer
def main_modelo_transformer():
    """
    Ejecuta la inferencia del modelo Transformer sobre la última ventana de datos del SPY.
    Flujo de trabajo (end-to-end):
        1. Carga el CSV `csv_spy_path` desde la carpeta `inputs`.
        2. Añade indicadores técnicos llamando a `agregar_indicadores` (sin data leakage).
        3. Filtra las columnas requeridas por el modelo (lista `columnas_necesarias`).
        4. Aplica el `scaler_data` (StandardScaler) previamente cargado para normalizar
                las características.
        5. Sanitiza valores extremos/NaNs con `sanitize_data` (clip).
        6. Ajusta la forma a 3D (batch, seq_len, features) y llama a `model_transformer.predict`.

    Returns:
        np.ndarray: matriz de predicciones devuelta por el modelo (forma depende de la salida
                del modelo serializado). Normalmente será un array de probabilidades o logits
                para la ventana procesada.
    """
    # Cargar los datos
    spy = pd.read_csv(csv_spy_path)
    spy['date']=spy.index
    # Agregar indicadores técnicos
    spy_ind = agregar_indicadores(spy)
    # Filtrar las columnas necesarias
    df_global_filtrado = spy_ind[columnas_necesarias].copy()
    # Guardar el dataframe completo con indicadores (opcional)
    df_global_filtrado.to_csv(
        os.path.join(dir_indtec, "spy_ultimos_indicadores_tecnicos.csv"),
        index=False
    )

    # Sanitizar los datos
    X2D = scaler_data.transform(df_global_filtrado.values).astype("float32")
    X2D = sanitize_data(X2D, clip=10.0)
    # Realizar la predicción
    X3D = X2D[np.newaxis, :, :]  # (1, 60, 25)
    preds = model_transformer.predict(X3D)
    # Convertimos las predicciones en clase y probabilidad
    transformer_pred_id = int(np.argmax(preds))
    transformer_pred_prob = float(np.max(preds))
    transformer_class_map = {0: "HOLD" , 1: "BUY", 2: "SELL"} #OK
    etiqueta_transformer = transformer_class_map[transformer_pred_id]

    # Creamos una lista con la info necesaria
    resultado_transformer = [etiqueta_transformer, transformer_pred_prob]

    return resultado_transformer

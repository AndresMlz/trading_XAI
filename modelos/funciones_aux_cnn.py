"""
Módulo que contiene las funciones necesarias para orquestar la ejecución
del modelo CNN para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar librerías estándar
import io
import os
import time
from dataclasses import dataclass

# Importar librerías de terceros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from PIL import Image

# Importar TensorFlow y sus componentes
import tensorflow as tf

# Importar componentes de Keras según la versión de TensorFlow
if tf.__version__.startswith('2'):
    try:
        # Para TF 2.x
        from keras import Model, Input, layers
        from keras import mixed_precision
        from keras.applications import (
            EfficientNetB0,
            MobileNetV3Small
            
        )
        from keras.applications import efficientnet, mobilenet_v3
    except ImportError:
        # Alternativa para algunas versiones de TF 2.x
        from tensorflow.keras import Model, Input, layers
        from tensorflow.keras.applications import (
            EfficientNetB0,
            MobileNetV3Small
        )
        from tensorflow.keras.applications import efficientnet, mobilenet_v3
else:
    raise ImportError("Este código requiere TensorFlow 2.x")

# Paths necesarios
dir_principal = os.getcwd()
dir_modelos = os.path.join(dir_principal, "archivos_modelos")
dir_inputs = os.path.join(dir_principal, "inputs")
dir_outputs = os.path.join(dir_principal, "outputs")
modelo_cnn_path = os.path.join(dir_modelos, "modelo_cnn.h5")

# Definir paths
principal_dir = os.getcwd()
modelos_path = os.path.join(principal_dir, "archivos_modelos")
outputs_dir = os.path.join(principal_dir, "outputs")
imputs_dir = os.path.join(principal_dir, "inputs")
imagenes_dir = os.path.join(principal_dir, "imagenes_generadas")
csv_spy_path = os.path.join(imputs_dir, "SPY_last10.csv")
modelo_cnn_path = os.path.join(modelos_path, "modelo_cnn.h5")

# Definir constantes
RUN_TAG  = time.strftime("%Y%m%d_%H%M%S")
DESC_SINGLE  = f"single_{RUN_TAG}"
LOOKBACK = 30
ANCHOR_TIMESTAMP = None
BG_COLOR = "#ffffff"
RENDER_DPI = 120

# Funciones auxiliares
@dataclass
class RenderCfg:
    """
    Parámetros de renderizado para las imágenes de velas.
    """
    dpi: int = RENDER_DPI
    bg_color: str = BG_COLOR
    transparent_bg: bool = False

# Función auxiliar para convertir timestamp a nombre de archivo
def ts_to_fname(ts: pd.Timestamp) -> str:
    """
    Convierte un timestamp de pandas a un nombre de archivo seguro para el sistema de archivos.
    Args:
        ts (pd.Timestamp): Objeto timestamp de pandas que se desea convertir
    Returns:
        str: Cadena de texto con formato seguro para nombres de archivo, donde:
             - Los espacios son reemplazados por guiones bajos (_)
             - Los dos puntos son reemplazados por guiones medios (-)
    """
    return str(ts).replace(" ", "_").replace(":", "-")

# Función auxiliar para cargar y procesar datos OHLC desde CSV
def load_prices(csv_path: str) -> pd.DataFrame:
    """
    Carga y procesa datos OHLC (Open, High, Low, Close) desde un archivo CSV.
    
    Esta función realiza las siguientes operaciones:
    1. Lee el archivo CSV con fechas parseadas
    2. Limpia y ordena los datos por fecha
    3. Convierte las columnas de precios a tipo numérico
    4. Elimina filas con valores faltantes
    Args:
        csv_path (str): Ruta al archivo CSV que contiene los datos OHLC del SPY 
    Returns:
        pd.DataFrame: DataFrame procesado con las siguientes características:
            - Índice: datetime ordenado ascendentemente
            - Columnas:
                * SPY_open (float): Precio de apertura
                * SPY_high (float): Precio más alto
                * SPY_low (float): Precio más bajo
                * SPY_close (float): Precio de cierre
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = (
        df.dropna(subset=["date"])
          .sort_values("date")
          .reset_index(drop=True)
          .set_index("date")
    )
    for c in ["SPY_open", "SPY_high", "SPY_low", "SPY_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SPY_open", "SPY_high", "SPY_low", "SPY_close"])
    print(f"[INFO] Rango: {df.index.min()} -> {df.index.max()} | filas: {len(df):,}")
    return df

# Función auxiliar para seleccionar un punto de anclaje y ventana histórica
def select_anchor(df: pd.DataFrame, lookback: int,
                  anchor_ts=None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Selecciona un punto de anclaje temporal ('anchor') y extrae una ventana de datos históricos.
    Esta función es robusta ante diferentes escenarios de selección temporal:
    1. Si no se especifica anchor_ts, usa el último registro disponible
    2. Si el anchor_ts solicitado no existe, usa el registro más cercano anterior
    3. Si el anchor_ts es anterior al primer registro, usa el primer registro
    Args:
        df (pd.DataFrame): DataFrame con índice temporal que contiene los datos históricos
        lookback (int): Número de registros históricos a incluir en la ventana
        anchor_ts (pd.Timestamp | str | None, optional): Timestamp objetivo para anclar la ventana.
            Puede ser None, un objeto Timestamp, o una cadena de texto parseable como fecha.
            Por defecto es None.
    Returns:
        tuple[pd.DataFrame, pd.Timestamp]: Una tupla que contiene:
            - DataFrame con los últimos `lookback` registros hasta el anchor
            - Timestamp del punto de anclaje seleccionado
    """
    if anchor_ts is None:
        anchor = df.index[-1]
    else:
        anchor_req = pd.to_datetime(anchor_ts)
        if anchor_req in df.index:
            anchor = anchor_req
        else:
            idx = df.index.get_indexer([anchor_req], method="pad")[0]
            anchor = df.index[0] if idx == -1 else df.index[idx]

    win = df.loc[:anchor].tail(lookback)
    return win, anchor

# Función auxiliar para renderizar velas legibles
def render_candles_legible(ohlc_df: pd.DataFrame, out_path: str,
                           final_size: tuple[int, int], cfg: RenderCfg) -> None:
    """
    Genera y guarda un gráfico de velas (candlestick) a partir de datos OHLC.
    
    Esta función realiza las siguientes operaciones:
    1. Renombra las columnas del DataFrame al formato estándar de mplfinance
    2. Configura los colores y estilos del gráfico de velas
    3. Renderiza el gráfico sin ejes ni volumen
    4. Redimensiona la imagen al tamaño especificado
    5. Guarda el resultado como archivo PNG optimizado
    Args:
        ohlc_df (pd.DataFrame): DataFrame con los datos de precios, debe contener las columnas:
            - SPY_open: Precios de apertura
            - SPY_high: Precios máximos
            - SPY_low: Precios mínimos
            - SPY_close: Precios de cierre
        out_path (str): Ruta completa donde se guardará el archivo PNG generado
        final_size (tuple[int, int]): Dimensiones finales de la imagen en píxeles (ancho, alto)
        cfg (RenderCfg): Configuración de renderizado que incluye:
            - dpi (int): Resolución de la imagen
            - bg_color (str): Color de fondo en formato hex
            - transparent_bg (bool): Si el fondo debe ser transparente
    Returns:
        None: La función guarda el archivo en disco pero no retorna valores
    """
    data = ohlc_df.rename(columns={
        "SPY_open": "Open", "SPY_high": "High",
        "SPY_low": "Low",   "SPY_close": "Close"
    })[["Open", "High", "Low", "Close"]]

    mc = mpf.make_marketcolors(up='#2e7d32', down='#c62828', wick='#777777',
                                    edge='#777777')
    s  = mpf.make_mpf_style(marketcolors=mc, figcolor=cfg.bg_color, facecolor=cfg.bg_color)

    buf = io.BytesIO()
    mpf.plot(
        data, type='candle', style=s, volume=False, axisoff=True, tight_layout=True,
        returnfig=False,
        savefig=dict(
            fname=buf, dpi=cfg.dpi, bbox_inches='tight',
            transparent=cfg.transparent_bg, pad_inches=0, format='png'
        )
    )
    plt.close('all')
    buf.seek(0)
    with Image.open(buf) as im:
        im = im.convert("RGB")
        im = im.resize(final_size, resample=Image.Resampling.LANCZOS)
        # Asegura carpetas
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, format="PNG", optimize=True)

# Función auxiliar para forzar ejecución en CPU
def set_cpu(max_intra: int = 0, max_inter: int = 0) -> None:
    """
    Configura TensorFlow para ejecutar exclusivamente en CPU y optimiza su rendimiento.
    Esta función realiza las siguientes configuraciones:
    1. Deshabilita todos los dispositivos GPU disponibles
    2. Establece la política de precisión a float32 (desactiva mixed-precision)
    3. Configura opcionalmente el paralelismo de operaciones
    Args:
        max_intra (int, optional): Número máximo de hilos para paralelismo intra-operación.
            - 0: Usa el valor por defecto de TensorFlow
            - >0: Limita el número de hilos a ese valor
            Por defecto es 0.
            
        max_inter (int, optional): Número máximo de hilos para paralelismo inter-operación.
            - 0: Usa el valor por defecto de TensorFlow
            - >0: Limita el número de hilos a ese valor
            Por defecto es 0.
    Returns:
        None: La función modifica la configuración global de TensorFlow
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # fuerza CPU
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception: # pylint: disable=broad-except
        pass

    if max_intra:
        tf.config.threading.set_intra_op_parallelism_threads(int(max_intra))
    if max_inter:
        tf.config.threading.set_inter_op_parallelism_threads(int(max_inter))
    mixed_precision.set_global_policy("float32")

# Función auxiliar para construir el modelo CNN
def build_model(backbone: str, img_size: int, num_classes: int, dropout: float) -> Model:
    """
    Construye un modelo de CNN para clasificación de imágenes utilizando
    arquitecturas pre-entrenadas.
    
    Esta función crea un modelo de clasificación que:
    1. Utiliza EfficientNetB0 o MobileNetV3Small como backbone (red base)
    2. Aplica pooling global promedio a las características extraídas
    3. Añade dropout para regularización
    4. Termina con una capa densa para clasificación
    Args:
        backbone (str): Arquitectura base a utilizar. Valores permitidos:
            - "efficientnet_b0": Más preciso pero más pesado
            - cualquier otro valor: Usa MobileNetV3Small (más ligero)
        img_size (int): Tamaño de la imagen de entrada (cuadrada) en píxeles.
            Debe coincidir con el tamaño de las imágenes preprocesadas.
        num_classes (int): Número de clases para la clasificación.
            Determina el número de neuronas en la capa de salida.
        dropout (float): Tasa de dropout para regularización.
            Valor entre 0 y 1 que indica la proporción de conexiones a desactivar.
    Returns:
        Model: Modelo compilado con la siguiente estructura:
            - Input: (batch_size, img_size, img_size, 3)
            - Backbone: EfficientNetB0 o MobileNetV3Small pre-entrenado con ImageNet
            - Global Average Pooling
            - Dropout
            - Dense(num_classes) con activación softmax
    """
    inp = Input(shape=(img_size, img_size, 3))
    
    if backbone == "efficientnet_b0":
        base = EfficientNetB0(include_top=False, input_tensor=inp, pooling="avg",
                          weights="imagenet")
    else:
        base = MobileNetV3Small(include_top=False, input_tensor=inp, pooling="avg",
                             weights="imagenet")
                             
    x = layers.Dropout(float(dropout))(base.output)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inp, out)

# Función auxiliar para cargar y preprocesar imagen PNG
def load_png_tensor(path: str, img_size: int, backbone: str) -> tf.Tensor:
    """
    Carga y preprocesa una imagen PNG para su uso en modelos de deep learning.
    
    Esta función realiza las siguientes operaciones:
    1. Lee el archivo PNG desde disco
    2. Decodifica la imagen a un tensor RGB de 3 canales
    3. Redimensiona la imagen al tamaño requerido
    4. Aplica el preprocesamiento específico del backbone
    5. Añade la dimensión de batch
    
    Args:
        path (str): Ruta absoluta al archivo PNG a cargar
        img_size (int): Tamaño objetivo para la imagen (será redimensionada a img_size x img_size)
        backbone (str): Nombre del backbone para determinar el preprocesamiento:
            - "efficientnet_b0": Usa el preprocesamiento de EfficientNet
            - otro valor: Usa el preprocesamiento de MobileNetV3
    Returns:
        tf.Tensor: Tensor 4D con forma (1, img_size, img_size, 3) donde:
            - 1 es la dimensión del batch (una sola imagen)
            - img_size x img_size son las dimensiones espaciales
            - 3 son los canales de color (RGB)
            - Los valores están preprocesados según el backbone elegido
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mob_pre
    fn = eff_pre if backbone == "efficientnet_b0" else mob_pre
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = fn(img)  # Aplica el preprocesamiento específico del backbone
    img = tf.expand_dims(img, 0)  # Añade la dimensión del batch
    return img

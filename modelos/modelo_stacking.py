"""
Módulo que contiene las funciones necesarias para el orquestamiento
del modelo stacking con el que se busca identificar la acción a
realizar con base en las decisiones de los modelos de reconocimiento
de patrones, CNN y Transformer.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import joblib
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from config.constantes import Constantes as CONST
from data_alpaca.descarga_data import get_spy_data
from modelos.modelo_recpat import main_modelo_recpat
from modelos.modelo_cnn import main_modelo_cnn
from modelos.modelo_transformer import main_modelo_transformer

# Definir los directorios necesarios
dir_principal = os.getcwd()
dir_modelos = os.path.join(dir_principal, "archivos_modelos")
dir_outputs = os.path.join(dir_principal, "outputs")
dir_resultado_stacking = os.path.join(dir_principal, "resultado_stacking")

# Definir las variables necesarias
class_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
model_stacking = joblib.load(os.path.join(dir_modelos, "modelo_stacking.pkl"))

# Diccionarios de homologaciones de resultados
map_pred_class = {"HOLD": 0, "BUY": 1, "SELL": 2}       # CNN
map_transformer = {"BUY": 0, "HOLD": 1, "SELL": 2}      # Transformer
map_signal = {"Buy": 0, "Hold": 1, "Sell": 2}           # Pattern signal
threshold = float(CONST.DATA["Umbral_stacking"].replace(",", "."))

# En lugar de "última hora", pedimos los últimos días para garantizar datos
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=5)

# Crear la solicitud de barras
request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=start_time,
    end=end_time,
    feed=DataFeed.IEX  # plan gratuito
)

# Función para descargar la data y correr los diferentes modelos
def ejecucion_decision_modelos():
    """
    Función principal para ejecutar el pipeline de descarga de datos
    y correr los modelos de reconocimiento de patrones, CNN, Transformer
    y Stacking para tomar una decisión final.
    1) Descargar los datos más recientes del SPY.
    2) Ejecutar el modelo de reconocimiento de patrones.
    3) Ejecutar el modelo CNN.
    4) Ejecutar el modelo Transformer.
    5) Combinar las predicciones y tomar una decisión final con el modelo Stacking.
    Returns:
        str: Decisión final tomada por el modelo Stacking ("BUY", "SELL", "HOLD").
    """
    # 1) Descargar los datos más recientes del SPY
    get_spy_data("SPY_last10.csv", request)
    # 2) Ejecutar el modelo CNN
    resultado_cnn = main_modelo_cnn()
    # 3) Ejecutar el modelo Transformer
    resultado_transformer = main_modelo_transformer()
    # 4) Ejecutar el modelo de reconocimiento de patrones
    resultado_recpat = main_modelo_recpat()
    # 5) Crear un dataframe con los resultados de los modelos
    df_res = pd.DataFrame([{
        "pred_class_name": resultado_cnn[0],
        "pred_prob": resultado_cnn[1],
        "candlesticks_pattern": resultado_recpat[0],
        "pattern_signal": resultado_recpat[1],
        "pattern_conf": resultado_recpat[2],
        "etiqueta_transformer": resultado_transformer[0],
    }])
    # 6) Mapear las etiquetas a valores numéricos
    map_pattern = {resultado_recpat[0]: 0}    
    df_res["pred_class_name"] = df_res["pred_class_name"].map(map_pred_class)
    df_res["etiqueta_transformer"] = df_res["etiqueta_transformer"].map(map_transformer)
    df_res["pattern_signal"] = df_res["pattern_signal"].map(map_signal)
    df_res["candlesticks_pattern"] = df_res["candlesticks_pattern"].map(map_pattern)
    # 7) Crear variables derivadas
    df_res["pred_prob_x_conf"] = df_res["pred_prob"] * df_res["pattern_conf"]
    df_res["pred_prob_x_signal"] = df_res["pred_prob"] * df_res["pattern_signal"]
    # 8) Reordenar las columnas
    cols_expected = [
    "pred_class_name", "pred_prob", "candlesticks_pattern",
    "pattern_signal", "pattern_conf", "etiqueta_transformer",
    "pred_prob_x_conf", "pred_prob_x_signal"
    ]
    df_res = df_res[cols_expected]
    # 9) Predecir con el modelo de stacking
    y_proba_stacking = model_stacking.predict_proba(df_res)
    y_pred_stacking = model_stacking.predict(df_res)
    conf_max = np.max(y_proba_stacking, axis=1)
    # 10) Revisar el umbral y decidir la acción final
    final_pred = np.copy(y_pred_stacking)
    final_pred[conf_max < threshold] = map_pred_class["HOLD"]
    # 11) Decodificar el resultado
    inv_label = {v: k for k, v in map_pred_class.items()}
    df_res["final_pred"] = [inv_label[i] for i in final_pred]
    df_res["confidence"] = conf_max
    df_res["executed"] = np.where(conf_max >= threshold, "ACTIVA", "HOLD (filtrada)")
    # 12) Guardar el resultado final
    df_res.to_csv(
        os.path.join(dir_resultado_stacking, "resultado_stacking.csv"), index=False
    )

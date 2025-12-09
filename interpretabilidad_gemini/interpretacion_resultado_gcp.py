"""
Modulo que contiene la orquestación de la
interpretabilidad del modelo usando el RAG.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
import json
from typing import Any
import pandas as pd
import numpy as np
from config.constantes import Constantes as CONST
from enriquecimiento_datos.enriquecimiento_spy import generar_hoja_indicadores
from modelos.modelo_stacking import ejecucion_decision_modelos
from interpretabilidad_gemini.rag_gcp import generar_respuesta_rag

# Definir los paths necesarios
principal_dir = os.getcwd()
inputs_dir = os.path.join(principal_dir, "inputs")
ind_tecnicos_dir = os.path.join(principal_dir, "data_ind_tecnicos")
stacking_dir = os.path.join(principal_dir, "resultado_stacking")
csv_enriquecido = os.path.join(inputs_dir, "indicators_sheet_human.csv")
csv_stacking = os.path.join(stacking_dir, "resultado_stacking.csv")
csv_ind_tecnicos = os.path.join(ind_tecnicos_dir, "spy_ultimos_indicadores_tecnicos.csv")

# Diccionario de interpretación de patrones
pattern_interpretation = {
    "ascending-triangle": "alcista",
    "bearish-flag": "bajista",
    "bullish-flag": "alcista",
    "descending-triangle": "bajista",
    "double-bottom": "alcista",
    "double-top": "bajista",
    "falling-wedge": "alcista",
    "head-and-shoulders": "bajista",
    "inverse-head-and-shoulders": "alcista",
    "rising-wedge": "bajista",
    "rounding-bottom": "alcista",
    "rounding-top": "bajista",
    "triple-bottom": "alcista",
    "triple-top": "bajista"
}

# Función auxiliar para convertir todos los valores a tipos nativos
def to_native(o: Any) -> Any:
    """
    Convierte valores numpy y estructuras anidadas a tipos nativos de Python.
    Esta función se utiliza para preparar estructuras que provienen de pandas
    o numpy (por ejemplo valores de DataFrame convertidos con ``.to_dict()``)
    antes de serializarlas o pasarlas a APIs que esperan tipos nativos (int,
    float, bool, str, list, dict, ...).
    Comportamiento soportado:
      - numpy integers (np.int32, np.int64, ...): convierten a ``int``.
      - numpy floats (np.float32, np.float64, ...): convierten a ``float``.
      - numpy booleans (np.bool_): convierten a ``bool``.
      - dict: se aplica recursivamente a claves/valores y devuelve un dict con
        los mismos keys y valores convertidos.
      - list/tuple: se convierten recursivamente a listas o tuplas de tipos
        nativos respectivamente.
      - cualquier otro objeto se devuelve tal cual (por ejemplo ``str`` ya es nativo).
    Args:
        o: Objeto a normalizar (puede ser escalares numpy, dicts, listas, tuplas).
    Returns:
        El mismo objeto con los scalars numpy convertidos a tipos nativos de
        Python y con las estructuras internas procesadas recursivamente.
    """
    # Numpy integer types -> int
    if isinstance(o, (np.integer,)):
        return int(o)
    # Numpy float types -> float
    if isinstance(o, (np.floating,)):
        return float(o)
    # Numpy boolean -> bool
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    # Dicts -> aplicar recursivamente
    if isinstance(o, dict):
        return {k: to_native(v) for k, v in o.items()}
    # Lists / tuples -> convertir elementos recursivamente
    if isinstance(o, list):
        return [to_native(v) for v in o]
    if isinstance(o, tuple):
        return tuple(to_native(v) for v in o)
    # Valor por defecto: devolver tal cual
    return o

# Función para interpretar los resultados del modelo Gemini
def interpretacion_resultado():
    """
    Función para interpretar los resultados del modelo utilizando Gemini.
    """
    # 1) Correr los modelos para obtener el CSV de stacking
    ejecucion_decision_modelos()
    # 2) Generar la hoja de indicadores
    generar_hoja_indicadores()
    # 3) Cargar el CSV enriquecido del Transformer
    df_transformer = pd.read_csv(csv_ind_tecnicos)
    # 4) Cargar el CSV enriquecido de la CNN
    df_enriquecido = pd.read_csv(csv_enriquecido).tail(1).to_dict(orient="records")[0]
    # 5) Cargar el CSV de resultados de los modelos
    df_stacking = pd.read_csv(csv_stacking)
    # 6) Combinar los dataframes enriquecidos y de stacking
    fusion = {
    "cnn_signal": df_stacking["pred_class_name"].iloc[0],
    "cnn_confidence": df_stacking["pred_prob"].iloc[0],
    "transformer_signal": df_stacking["etiqueta_transformer"].iloc[0],
    "pattern_detected": df_stacking["candlesticks_pattern"].iloc[0],
    "pattern_conf": df_stacking["pattern_conf"].iloc[0],
    "pattern_signal": df_stacking["pattern_signal"].iloc[0],
    "meta_final": df_stacking["final_pred"].iloc[0],
    "meta_conf": df_stacking["confidence"].iloc[0],
    "human_context": df_enriquecido
    }
    # 7) Convertir a tipos nativos
    fusion_native = {k: to_native(v) for k, v in fusion.items()}
    # 8) Perparar la interpretación de patrones
    pattern_label = fusion_native["pattern_detected"]
    pattern_type = pattern_interpretation.get(pattern_label, "desconocido")
    # 9) Preparar el prompt para Gemini
    prompt = f"""
    {CONST.DATA["Prompt_Gemini"]} +
    Ten en cuenta que se detectó el patrón chartista '{pattern_label}', 
    el cual se considera un patrón típicamente {pattern_type}, con una confianza de 
    {fusion_native['pattern_conf']:.2f}. El modelo lo interpretó con una señal 
    '{['Buy','Hold','Sell'][fusion_native['pattern_signal']]}'. Inclúyelo como evidencia
    visual relevante en tu análisis, en caso de que no se haya detectado un patrón debes
    indicar que no se identificó un patrón chartista.

    Considera también el comportamiento de los patrones técnicos que se encuentran en la hoja
    {df_transformer}, es importante que contemples los datos que se alojan para que complementes
    el análisis que se realice.

    ### Datos del sistema
    {json.dumps(fusion_native, indent=2, ensure_ascii=False)}

    ### Instrucciones
    1. Analiza la coherencia entre los modelos base (CNN, Transformer, ResNet).
    2. Explica por qué el meta-modelo decidió '{fusion_native['meta_final']}'.
    3. Incluye al menos 2 argumentos técnicos (indicadores del humano) y 1 visual
    (patrón o mapa de calor).
    4. Usa tono analítico, preciso y en español técnico de trading (máx. 250 palabras).

    Devuelve en formato JSON con:
    {{
    "explicacion_resumida": "...",
    "drivers": ["..."],
    "contradicciones": ["..."],
    "nivel_confianza": "<bajo|medio|alto>",
    "accion_recomendada": "<BUY|SELL|HOLD>"
    }}
    """
    # 10) Llamar a Gemini para obtener la interpretación
    respuesta = generar_respuesta_rag(prompt)
    respuesta_txt = (respuesta)
    
    return respuesta_txt

"""
Módulo que contiene la función principal para correr el modelo
CNN para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías estándar
import pandas as pd
from modelos.funciones_aux_cnn import (RenderCfg, build_model, load_png_tensor, load_prices,
                                       render_candles_legible, select_anchor, set_cpu, ts_to_fname)
import json
import os
import time
# Configuración de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprimir logs de TensorFlow
# Importar librerías de terceros
import numpy as np

# Definir constantes
RUN_TAG  = time.strftime("%Y%m%d_%H%M%S")
DESC_SINGLE  = f"single_{RUN_TAG}"
LOOKBACK = 30
ANCHOR_TIMESTAMP = None
BG_COLOR = "#ffffff"
RENDER_DPI = 120

# Paths necesarios
dir_principal = os.getcwd()
dir_modelos = os.path.join(dir_principal, "archivos_modelos")
dir_inputs = os.path.join(dir_principal, "inputs")
dir_outputs = os.path.join(dir_principal, "outputs")


# Definir paths
run_dir_path = os.path.join(dir_outputs, DESC_SINGLE)
csv_spy_path = os.path.join(dir_inputs, "SPY_last10.csv")
imagenes_dir = os.path.join(run_dir_path, "imagenes_generadas")
modelo_cnn_path = os.path.join(dir_modelos, "model_cnn.weights.h5")
meta_data_path = os.path.join(dir_modelos, "model_meta.json")

# Función main para correr el modelo CNN
def main_modelo_cnn():
    """
    Pipeline end-to-end para CPU:
    1) Carga OHLC y selecciona ventana (lookback, anchor).
    2) Lee metadatos del mejor modelo (backbone, img_size, dropout, etc.).
    3) Renderiza PNG de velas (tamaño img_size x img_size).
    4) Fuerza CPU y construye modelo; carga pesos.
    5) Ejecuta inferencia y persiste resultados (CSV completo + resumen).
    """
    # 1) Datos y ventana
    df = load_prices(csv_spy_path)
    win, anchor = select_anchor(df, LOOKBACK, ANCHOR_TIMESTAMP)

    # 2) Metadatos del modelo
    with open(meta_data_path, encoding="utf-8") as f:
        meta = json.load(f)

    backbone    = meta.get("BACKBONE", "mobilenet_v3_small")
    img_size    = int(meta.get("IMG_SIZE", 224))
    dropout     = float(meta.get("DROPOUT", 0.3))
    num_classes = int(meta.get("NUM_CLASSES", 3))
    class_names = meta.get("CLASS_NAMES", {0: "HOLD", 1: "BUY", 2: "SELL"})

    # (Opcional) restringir tamaños/backbone en CPU para latencia
    if backbone not in ("mobilenet_v3_small", "efficientnet_b0"):
        backbone = "mobilenet_v3_small"
    img_size = min(img_size, 192)

    # 3) Render a PNG (cuadrado) para el backbone
    cfg = RenderCfg()
    img_name = f"img_{ts_to_fname(anchor)}.png"
    img_path = os.path.join(imagenes_dir, img_name)
    render_candles_legible(win, img_path, (img_size, img_size), cfg)

    # 4) CPU ON, construir y cargar pesos
    set_cpu(max_intra=0, max_inter=0)  # 0 = auto; puedes probar 8/2 o 4/2 según tu CPU
    model = build_model(backbone, img_size, num_classes, dropout)
    model.load_weights(modelo_cnn_path)

    # 5) Inferencia
    x = load_png_tensor(img_path, img_size, backbone)
    probs = model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    pred_label = class_names.get(pred_id, str(pred_id))
    pred_prob = float(probs[pred_id])

    # 6) Persistencia
    out = df.copy()
    out["pred_label"] = pred_label
    os.makedirs(run_dir_path, exist_ok=True)

    csv_pred = os.path.join(run_dir_path, "prediccion_unica.csv")
    out.to_csv(csv_pred)

    # Resumen compacto de inferencia (útil para auditoría)
    summary_csv = os.path.join(run_dir_path, "prediccion_resumen.csv")
    pd.DataFrame([{
        "anchor": anchor,
        "pred_id": pred_id,
        "pred_label": pred_label,
        **{f"p{k}": float(v) for k, v in enumerate(probs)}
    }]).to_csv(summary_csv, index=False)

    # 7) lista con la info necesaria
    resultado_cnn = [pred_label, pred_prob]

    return resultado_cnn

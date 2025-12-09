"""
Módulo que contiene la función principal para correr el modelo de reconocimiento
de patrones para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
import torch
from modelos.funciones_aux_recpat import (load_ckpt_and_meta, build_resnet18, get_eval_transform,
                                             load_image_tensor, predict_with_threshold,
                                             get_latest_image)

# Definir las rutas necesarias
dir_principal = os.getcwd()
dir_outputs = os.path.join(dir_principal, "outputs")
dir_modelos = os.path.join(dir_principal, "archivos_modelos")

# Definir los paths necesarios
model_path = os.path.join(dir_modelos, "modelo_recpat.pt")
labels_path  = os.path.join(dir_modelos, "labels_recpat.txt")

# Definir las constantes necesarias
THRESHOLD = 0.75
TOP_K = 3
meta = load_ckpt_and_meta(model_path, labels_path)
class_names   = meta["class_names"]
num_classes   = meta["num_classes"]
img_size      = meta["img_size"]
mean, std     = meta["mean"], meta["std"]
channels_last = meta["channels_last"]
state_dict    = meta["state_dict"]
pattern_to_signal = {
    "ascending-triangle": "Buy",
    "bearish-flag": "Sell",
    "bullish-flag": "Buy",
    "descending-triangle": "Sell",
    "double-bottom": "Buy",
    "double-top": "Sell",
    "falling-wedge": "Buy",
    "head-and-shoulders": "Sell",
    "inverse-head-and-shoulders": "Buy",
    "rising-wedge": "Sell",
    "rounding-bottom": "Buy",
    "rounding-top": "Sell",
    "triple-bottom": "Buy",
    "triple-top": "Sell"
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Función main para correr el modelo de reconocimiento de patrones
def main_modelo_recpat():
    """
    Función principal para ejecutar el modelo de reconocimiento de patrones.
    1) Encuentra la última imagen generada del precio.
    2) Carga el modelo ResNet18 con los pesos entrenados.
    3) Prepara la imagen para la inferencia.
    4) Ejecuta la predicción con umbral y genera Grad-CAM.
    5) Extrae la etiqueta del patrón, probabilidad y señal asociada.
    6) Devuelve los resultados.
    Returns:
        pattern (str): Etiqueta del patrón reconocido.
        pattern_prob (float): Probabilidad asociada al patrón.
        pattern_signal (str): Señal asociada al patrón ("Buy", "Sell", "Hold").

    """
    # 1) Encontrar el path de la ultima imagen
    image_path = get_latest_image(dir_outputs)

    # 2) Armar y enviar modelo a device
    model = build_resnet18(num_classes, state_dict).to(DEVICE)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # 3) Transform y tensor de imagen
    tfm = get_eval_transform(img_size, mean, std)
    orig_pil, img_t = load_image_tensor(image_path, tfm, DEVICE, channels_last=channels_last)

    # 4) Grad-CAM sobre layer4 (último bloque convolucional de ResNet18)
    target_layer = model.layer4

    # 5) Ejecutar predicción CNN
    result = predict_with_threshold(
        model, img_t, class_names, k=TOP_K, threshold=THRESHOLD,
        target_layer=target_layer, orig_pil=orig_pil, mean=mean, std=std
    )

    # 6) Extraer lo necesario
    pattern_label = result["top1"][0]        # e.g., "rising-wedge"
    pattern_prob  = float(result["top1"][1]) # e.g., 0.82
    pattern_signal = pattern_to_signal.get(pattern_label, "Hold")  # default por seguridad

    # 7) Crear una lista con la info necesaria
    resultado_prediccion = [pattern_label, pattern_signal, pattern_prob ]

    return resultado_prediccion
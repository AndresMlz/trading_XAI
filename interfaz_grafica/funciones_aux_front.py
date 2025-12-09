"""
Modulo que contiene las funciones auxiliares para la interfaz grafica.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerias necesarias
import os
from pathlib import Path
from typing import Literal
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Función auxiliar que busca las imagenes recientes
def buscar_imagenes_recientes(diccionario_images):
    """
    Esta función busca en un directorio base todas las carpetas que comienzan con "single_",
    identifica la más reciente según su fecha de modificación, y luego busca dentro de ella
    las imágenes de gráficos de velas (candlestick) y los mapas de calor Grad-CAM generados
    por el modelo. La función está diseñada para encontrar automáticamente los resultados
    más recientes de una ejecución del sistema de trading.
    Args:
        diccionario_images (str): Ruta del directorio base donde se encuentran las carpetas
                                  "single_*" que contienen los resultados de las ejecuciones.
                                  Ejemplo: "/content/Outputs"
    Returns:
        tuple: Tupla con dos elementos:
            - Si encuentra la imagen original:
                - ori_img (str): Ruta completa al archivo de la imagen de velas más reciente
                - heat_img (str o None): Ruta completa al archivo de la imagen Grad-CAM más
                                         reciente, o None si no se encuentra
            - Si hay error:
                - False (bool): Indica que hubo un error
                - mensaje_error (str): Mensaje descriptivo del error encontrado
    
    Raises:
        FileNotFoundError: Si no se encuentran carpetas "single_*" en el directorio base.
    
    Note:
        La función muestra un mensaje en la interfaz de Streamlit indicando qué carpeta
        fue detectada como la más reciente. Si no encuentra la imagen Grad-CAM, muestra
        una advertencia pero continúa retornando la imagen original.
    """
    if not os.path.isdir(diccionario_images):
        return False, f"No existe el directorio base: {diccionario_images}"
    single_dirs = [
        os.path.join(diccionario_images, d)
        for d in os.listdir(diccionario_images)
        if d.startswith("single_") and os.path.isdir(os.path.join(diccionario_images, d))
    ]
    if not single_dirs:
        raise FileNotFoundError("No se encontraron carpetas 'single_*' en /content/Outputs")

    latest_single = max(single_dirs, key=os.path.getmtime)
    st.caption(f"📂 Carpeta más reciente detectada: **{os.path.basename(latest_single)}**")

    images_dir = os.path.join(latest_single, "images")
    gradcam_dir = os.path.join(images_dir, "gradcam")

    candlestick_candidates = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "gradcam" not in f.lower()
    ]
    ori_img = max(candlestick_candidates, key=os.path.getmtime) if candlestick_candidates else None

    gradcam_candidates = []
    if os.path.isdir(gradcam_dir):
        gradcam_candidates += [
            os.path.join(gradcam_dir, f)
            for f in os.listdir(gradcam_dir)
            if f.lower().endswith((".png", ".jpg")) and "gradcam" in f.lower()
        ]
    gradcam_candidates += [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg")) and "gradcam" in f.lower()
    ]
    heat_img = max(gradcam_candidates, key=os.path.getmtime) if gradcam_candidates else None

    if not ori_img:
        return False, f"No se encontró imagen de velas en {images_dir}"
    if not heat_img:
        st.warning("⚠️ No se encontró imagen Grad-CAM. Solo se mostrará la original.")

    return ori_img, heat_img

# Función auxiliar que resalta la imagen Grad-CAM con el texto de la etiqueta.
def resaltar_gradcam(heatmap_path, label_text):
    """
    Agrega texto sobre la imagen Grad-CAM para mejorar su interpretabilidad.
    
    Esta función toma una imagen de mapa de calor Grad-CAM y le agrega una anotación
    de texto que describe la etiqueta o predicción del modelo. El texto se coloca en
    la esquina superior izquierda de la imagen con un fondo semitransparente para
    mejorar la legibilidad. La imagen anotada se guarda en un archivo temporal y se
    retorna su ruta para su visualización en la interfaz.
    
    Args:
        heatmap_path (str): Ruta completa al archivo de imagen Grad-CAM que se desea
                            anotar. Debe ser un archivo de imagen válido (PNG, JPG, etc.)
        label_text (str): Texto que se desea agregar sobre la imagen. Generalmente contiene
                          información sobre la predicción del modelo, como el patrón detectado
                          y la señal (BUY/SELL/HOLD) con su confianza.
                          Ejemplo: "Patrón: Rising Wedge – BUY (0.78)"
    
    Returns:
        str: Ruta al archivo de imagen anotada guardada. Si ocurre un error durante el
             procesamiento, retorna la ruta original de la imagen sin modificar.
             - Si tiene éxito: "/content/gradcam_annotated.png"
             - Si hay error: heatmap_path (la ruta original recibida)
    
    Note:
        La función muestra una advertencia en Streamlit si ocurre algún error durante
        el procesamiento de la imagen, pero no interrumpe la ejecución del programa.
        La imagen se guarda con una resolución de 150 DPI y se cierra la figura de
        matplotlib después de guardarla para liberar memoria.
    """
    try:
        img = np.array(Image.open(heatmap_path))
        h, w, _ = img.shape
        plt.figure(figsize=(8, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.text(
            w * 0.05,
            h * 0.1,
            label_text,
            color="white",
            fontsize=14,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.3"),
        )
        temp_path = "/content/gradcam_annotated.png"
        plt.savefig(temp_path, bbox_inches="tight", dpi=150)
        plt.close()
        return temp_path
    except Exception as e: #pylint: disable=broad-exception-caught
        st.warning(f"No se pudo anotar la imagen: {e}")
        return heatmap_path

def obtener_ultima_imagen(
    base_outputs: str,
    modo: Literal["normal", "gradcam"] = "normal"
) -> str:
    """
    Busca dinámicamente la última carpeta 'single_*' dentro de `base_outputs`
    y devuelve la ruta de la última imagen generada.
    Args:
        base_outputs : str
            Ruta base donde se encuentran las carpetas 'single_*'.
            Ejemplo: "outputs" o "/ruta/completa/outputs".
        modo : {"normal", "gradcam"}, opcional
            - "normal": busca la imagen dentro de 'imagenes_generadas'.
            - "gradcam": busca la imagen dentro de 'imagenes_generadas/gradcam'.
    Returns:
        str: Ruta absoluta de la imagen encontrada.
    """

    base_dir = Path(base_outputs)

    if not base_dir.exists():
        raise FileNotFoundError(f"La ruta base no existe: {base_dir}")

    # 1) Buscar todas las carpetas que empiezan por 'single'
    carpetas_single = [
        d for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("single")
    ]

    if not carpetas_single:
        raise FileNotFoundError(f"No se encontraron carpetas 'single_*' en {base_dir}")

    # 2) Tomar la "última" por nombre (funciona bien con timestamps en el nombre)
    carpeta_ultima = sorted(carpetas_single, key=lambda p: p.name)[-1]

    # 3) Definir el directorio donde buscar la(s) imagen(es)
    dir_imagenes = carpeta_ultima / "imagenes_generadas"
    if modo == "gradcam":
        dir_imagenes = dir_imagenes / "gradcam"

    if not dir_imagenes.exists():
        raise FileNotFoundError(f"No existe el directorio de imágenes: {dir_imagenes}")

    # 4) Buscar imágenes (png/jpg/jpeg) y quedarnos con la más reciente
    extensiones_validas = {".png", ".jpg", ".jpeg"}
    imagenes = [
        f for f in dir_imagenes.iterdir()
        if f.is_file() and f.suffix.lower() in extensiones_validas
    ]

    if not imagenes:
        raise FileNotFoundError(f"No se encontraron imágenes en {dir_imagenes}")

    # Ordenar por fecha de modificación (última = más reciente)
    imagen_ultima = max(imagenes, key=lambda f: f.stat().st_mtime)

    return str(imagen_ultima.resolve())
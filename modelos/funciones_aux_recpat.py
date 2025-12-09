"""
Módulo que contiene las funciones necesarias para orquestar la ejecución del modelo
de reconocimiento de patrones para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
import numpy as np
import torch
from typing import Optional, Dict
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

# Definir las rutas necesarias
dir_principal = os.getcwd()
dir_outputs = os.path.join(dir_principal, "outputs")
dir_modelos = os.path.join(dir_principal, "archivos_modelos")

# Definir los paths necesarios
modelo_recpat_path = os.path.join(dir_modelos, "modelo_recpat.pt")
labels_recpat_path = os.path.join(dir_modelos, "labels_recpat.txt")

# Variables globales
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD  = 0.75
TOPK       = 3
SHOW_CAM_IF_REJECTED = True

# Función auxiliar para obtener la imagen más reciente
def  get_latest_image(base_dir: str) -> str:
    """
    Busca y devuelve la ruta del archivo de imagen más reciente generado por
    una ejecución del pipeline.
    Comportamiento:
      - Busca subdirectorios de `base_dir` cuyo nombre empiece por "single_".
      - Dentro de cada uno espera una subcarpeta llamada "images".
      - Entre todas las carpetas "images" encontradas selecciona la carpeta
        más recientemente modificada y devuelve la imagen más reciente dentro
        de esa carpeta según la fecha de modificación del fichero.
    Args:
        base_dir (str): Directorio raíz donde buscar las ejecuciones (por ejemplo
            la carpeta `outputs` donde se crean subcarpetas tipo `single_YYYYMMDD_HHMMSS`).
    Returns:
        str: Ruta absoluta al fichero de imagen más reciente (extensiones png/jpg/jpeg).
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"No existe el directorio base: {base_dir}")

    # Buscar todas las subcarpetas que empiecen por "single_" y contengan 'images'
    single_dirs = [
        os.path.join(base_dir, d, "imagenes_generadas")
        for d in os.listdir(base_dir)
        if d.startswith("single_") and os.path.isdir(os.path.join(
                                base_dir, d, "imagenes_generadas"))
    ]
    if not single_dirs:
        raise FileNotFoundError(
            "No se encontró ninguna carpeta 'single_*' con subcarpeta 'images/'.")

    # Tomar la carpeta 'images' más recientemente modificada
    latest_dir = max(single_dirs, key=os.path.getmtime)

    # Buscar imágenes dentro de la carpeta seleccionada
    imgs = [
        os.path.join(latest_dir, f)
        for f in os.listdir(latest_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not imgs:
        raise FileNotFoundError(f"No se encontraron imágenes en: {latest_dir}")

    # Tomar la imagen más reciente según fecha de modificación
    latest_img = max(imgs, key=os.path.getmtime)
    return latest_img

# Función auxiliar para cargar checkpoint y metadatos
def load_ckpt_and_meta(model_path: str, labels_path: Optional[str] = None):
    """
    Carga un checkpoint de PyTorch y extrae metadatos útiles para la inferencia.

    La función admite dos formatos comunes de checkpoint:
      - Un diccionario que contiene metadatos y el `model_state_dict` (formato
        típico de torch.save durante entrenamiento).
      - Un objeto guardado que contiene directamente los pesos (state_dict).
    Args:
        model_path (str): Ruta al fichero de checkpoint (.pt, .pth) a cargar.
        labels_path (str | None): Ruta a un fichero de texto con nombres de clase
            (una etiqueta por línea). Sólo es necesario si el checkpoint no
            incluye `class_names`.
    Returns:
        dict: Diccionario con las claves:
            - "class_names": list[str]
            - "num_classes": int
            - "img_size": int
            - "mean": list[float]
            - "std": list[float]
            - "channels_last": bool
            - "state_dict": state_dict (mapeo de pesos que puede pasarse a model.load_state_dict)
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No existe el checkpoint: {model_path}")

    # Cargar en CPU para máxima compatibilidad
    ckpt = torch.load(model_path, map_location="cpu")

    # Extraer nombres de clase: preferir los incluidos en el checkpoint
    if isinstance(ckpt, dict) and "class_names" in ckpt:
        class_names = list(ckpt["class_names"])
    else:
        if labels_path is None or not os.path.isfile(labels_path):
            raise FileNotFoundError("No hay 'class_names' en el checkpoint y falta labels.txt")
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

    # Extraer metadatos (con valores por defecto si faltan)
    num_classes = int(ckpt.get("num_classes", len(class_names))) if isinstance(ckpt,
                                                                        dict) else len(class_names)
    img_size = int(ckpt.get("img_size", 224)) if isinstance(ckpt, dict) else 224
    mean = ckpt.get("mean", [0.485, 0.456, 0.406]) if isinstance(ckpt, dict) else [
                                                                            0.485, 0.456, 0.406]
    std = ckpt.get("std", [0.229, 0.224, 0.225]) if isinstance(ckpt, dict) else [
                                                                            0.229, 0.224, 0.225]
    channels_last = bool(ckpt.get("channels_last", True)) if isinstance(ckpt, dict) else True

    # Determinar el state_dict real que contiene los pesos
    state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or None
        # Algunos frameworks guardan 'model' o 'model_dict'
        if state is None:
            for k in ("model", "model_dict", "state_dict"):  # fallback keys
                if k in ckpt and isinstance(ckpt[k], dict):
                    state = ckpt[k]
                    break
    else:
        # Si ckpt no es dict, asumimos que es un state_dict ya
        state = ckpt

    if state is None:
        raise RuntimeError("No se pudo localizar un 'state_dict' dentro del checkpoint")

    meta = {
        "class_names": class_names,
        "num_classes": num_classes,
        "img_size": img_size,
        "mean": mean,
        "std": std,
        "channels_last": channels_last,
        "state_dict": state,
    }
    return meta

# Función auxiliar para construir el modelo ResNet-18
def build_resnet18(num_classes: int, state_dict) -> nn.Module:
    """
    Construye un modelo ResNet-18 adaptado al número de clases y carga los pesos.
    Esta función crea una instancia de ResNet-18 sin pesos pre-entrenados, reemplaza
    la capa fully-connected final para que produzca `num_classes` salidas y carga
    el `state_dict` provisto.
    Args:
        num_classes (int): Número de clases (tamaño de la capa de salida final).
        state_dict (dict | OrderedDict): Diccionario de pesos (state_dict) obtenido
            desde `load_ckpt_and_meta(...)["state_dict"]` o desde `torch.load` directamente.
    Returns:
        torch.nn.Module: Instancia de `models.resnet18` con la cabeza final adaptada
            y los pesos cargados.
    """
    # Validación simple del state_dict
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict debe ser un diccionario mapeando nombres de parámetros a tfs")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        # Intentar cargar de forma más permisiva y avisar al usuario
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc2: # pylint: disable=broad-except
            # Re-levantar con información adicional
            raise RuntimeError(f"Error cargando state_dict en ResNet-18: {exc2}") from exc2

    return model

# Función auxiliar para obtener la transformación de evaluación
def get_eval_transform(img_size, mean, std):
    """
    Devuelve una transformación de evaluación lista para usar en inferencia.
    La transformación realiza, en orden:
      1. Redimensionado a `img_size` (mantiene relación de aspecto según PIL).
      2. CenterCrop a `img_size` (recorta al centro si es necesario).
      3. Conversión a tensor PyTorch en rango [0,1].
      4. Normalización usando `mean` y `std` (mismos valores que se usan en entrenamiento).
    Args:
        img_size (int): Tamaño de salida (alto=ancho) en píxeles para redimensionado y crop.
        mean (tuple[float, float, float]): Media por canal usada para normalizar (R,G,B).
        std (tuple[float, float, float]): Desviación estándar por canal (R,G,B).
    Returns:
        Callable: Un `transforms.Compose` apto para aplicar a objetos PIL Image.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# Función auxiliar para cargar la imagen y convertirla en tensor
def load_image_tensor(path, transform, device, channels_last: bool = True):
    """
    Carga una imagen desde `path`, aplica la transformación y
    devuelve el PIL + tensor listo para el modelo.

    Comportamiento:
      - Abre la imagen con PIL y la convierte a RGB.
      - Aplica `transform` (p. ej. `get_eval_transform`) y añade la dimensión batch.
      - Mueve el tensor al `device` solicitado ("cpu" o "cuda").
      - Opcionalmente convierte el tensor a memory_format `channels_last` para aceleración en GPU.
    Args:
        path (str): Ruta al fichero de imagen.
        transform (Callable): Transformación que acepta PIL.Image y devuelve tensor (C,H,W).
        device (torch.device | str): Dispositivo donde se colocará el tensor.
        channels_last (bool): Si True, convierte el tensor a memory_format=torch.channels_last.
    Returns:
        tuple(PIL.Image.Image, torch.Tensor): (imagen_pil, tensor) con forma (1,C,H,W)
            y está en el `device` indicado.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe la imagen: {path}")

    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    if channels_last:
        # Memory format channels_last puede mejorar el rendimiento en GPU para algunos modelos
        t = t.to(memory_format=torch.channels_last)
    return img, t

# Función para desnormalizar el tensor de imagen
def denormalize_img_tensor(t, mean, std):
    """
    Desnormaliza un tensor de imagen normalizado y devuelve un array numpy en rango [0,1].

    Args:
        t (torch.Tensor): Tensor con forma (1,C,H,W) normalizado con los `mean`/`std` usados
            en `get_eval_transform`.
        mean (tuple[float,float,float]): Media por canal usada para normalizar.
        std (tuple[float,float,float]): Desviación estándar por canal usada para normalizar.

    Returns:
        numpy.ndarray: Array con forma (H,W,3) en rango [0,1] apto para mostrar con matplotlib.

    Example:
        >>> img = denormalize_img_tensor(tensor, (0.485,0.456,0.406), (0.229,0.224,0.225))
        >>> plt.imshow(img)
    """
    # Convertir a CPU / numpy y reordenar a (H,W,C)
    t = t.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()
    t = t * np.array(std)[None, None, :] + np.array(mean)[None, None, :]
    return np.clip(t, 0, 1)

# Clase para Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients  = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out[0] es dL/dA del target_layer
            self.gradients = grad_out[0].detach()

        self.h_forward = target_layer.register_forward_hook(fwd_hook)
        # usar full_backward_hook si existe, si no, backward_hook
        if hasattr(target_layer, "register_full_backward_hook"):
            self.h_backward = target_layer.register_full_backward_hook(bwd_hook)
        else:
            self.h_backward = target_layer.register_backward_hook(bwd_hook)

    def remove(self):
        self.h_forward.remove()
        self.h_backward.remove()

    def generate(self):
        # activations: (N,C,H,W), gradients: (N,C,H,W)
        A = self.activations
        G = self.gradients
        weights = G.mean(dim=(2,3), keepdim=True)  # (N,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=True)  # (N,1,H,W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # normalizar 0..1
        cam -= cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam

# Función auxiliar para generar la predicción con umbral y Grad-CAM
def predict_with_threshold(model, img_tensor, class_names, k=3, threshold=0.75,
                           target_layer=None, orig_pil=None, mean=None, std=None):
    """
    Ejecuta una pasada de inferencia, calcula las top-k probabilidades y opcionalmente
    genera y visualiza un mapa Grad-CAM.
    Comportamiento resumido:
      1. Pone el modelo en modo evaluación y ejecuta un forward sobre `img_tensor`.
      2. Calcula probabilidades con softmax y extrae las top-k predicciones.
      3. Decide si la predicción top-1 es aceptada comparando `top1_prob >= threshold`.
      4. Si `target_layer` está indicado y `accepted` es True o `SHOW_CAM_IF_REJECTED` es True,
         realiza un backward sobre la puntuación top-1 para obtener Grad-CAM, genera el mapa
         de calor, lo redimensiona al tamaño de la imagen y crea un overlay.
      5. Guarda y muestra el overlay (si se generó) y devuelve un resumen textual y estructurado.
    Args:
        model (torch.nn.Module): Modelo PyTorch ya cargado con pesos y en el dispositivo
            adecuado (el tensor `img_tensor` también debe estar en el mismo device).
        img_tensor (torch.Tensor): Tensor de entrada con forma (1,C,H,W).
        class_names (list): Lista de nombres de clase indexada por etiqueta.
        k (int): Número de predicciones superiores a devolver.
        threshold (float): Umbral (0.0-1.0) para aceptar la predicción top-1.
        target_layer (nn.Module | None): Módulo objetivo donde registrar hooks para Grad-CAM.
            Usualmente es una de las últimas capas convolucionales del backbone.
        orig_pil (PIL.Image.Image | None): Imagen original (PIL) usada para mostrar junto a overlay
            Necesaria si se quiere visualizar o guardar el overlay.
        mean, std (tuple | None): Media y desviación usadas para desnormalizar el tensor cuando
            se crea el overlay. Si no se proporcionan, el overlay no podrá ser correctamente
            desnormalizado.
    Returns:
        dict: Contiene:
            - "accepted": bool indicando si top1_prob >= threshold.
            - "top1": tuple (class_name, prob, idx) de la mejor predicción.
            - "topk": list de tuplas (class_name, prob) ordenadas por probabilidad.
    """
    model.eval()
    model.zero_grad()

    # 1) Registrar hooks ANTES del forward
    cam = None
    if target_layer is not None:
        cam = GradCAM(model, target_layer)

    # 2) Forward (con grad habilitado)
    logits = model(img_tensor)                 # <-- aquí ya se capturan activaciones
    probs  = torch.softmax(logits, dim=1)[0]
    topk_prob, topk_idx = probs.topk(k)
    topk = [(class_names[i.item()], float(p.item())) for p, i in zip(topk_prob, topk_idx)]
    top1_idx  = topk_idx[0].item()
    top1_prob = float(topk_prob[0].item())
    accepted  = top1_prob >= threshold

    # 3) Backward para la clase top-1 (o la que quieras explicar)
    if cam is not None and (accepted or SHOW_CAM_IF_REJECTED):
        model.zero_grad()
        score = logits[0, top1_idx]
        score.backward(retain_graph=False)     # <-- ahora sí se capturan gradientes

        # 4) Generar CAM y liberar hooks
        cam_map = cam.generate()               # (Hc, Wc)
        cam.remove()

        # 5) Redimensionar CAM y hacer overlay
        cam_t = torch.from_numpy(cam_map)[None, None, ...].float()
        cam_up = F.interpolate(
            cam_t, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        base = denormalize_img_tensor(img_tensor, mean, std)  # (H,W,3) en 0..1
        heat = cm.jet(cam_up)[..., :3]
        overlay = np.clip(0.6 * base + 0.4 * heat, 0, 1)

        title_left  = "Original"
        mark = "✅" if accepted else f"❌ < {threshold}"
        title_right = f"Grad-CAM: {class_names[top1_idx]} (p={top1_prob:.2f}) {mark}"

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.imshow(orig_pil); plt.axis("off"); plt.title(title_left)
        plt.subplot(1,2,2); plt.imshow(overlay);  plt.axis("off"); plt.title(title_right)

        # Buscar la ultima imagen para obtener su xpath
        img_path = get_latest_image(dir_outputs)

        # Guardar overlay del mapa de calor
        out_dir = os.path.join(os.path.dirname(img_path), "gradcam")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "gradcam_overlay.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"💾 Grad-CAM guardado en: {out_path}")
        plt.show()
        plt.pause(0.001)
        from IPython.display import display
        display(plt.gcf())

    # 6) Reporte textual
    print(f"Top-{k}:")
    for name, p in topk:
        print(f"  - {name:>25s}: {p:.3f}")

    if accepted:
        print(f"\n✅ ACEPTADO: {class_names[top1_idx]} con p={top1_prob:.3f} ≥ {threshold}\n")
    else:
        print(f"\n⚠️  SIN DECISIÓN: p_top1={top1_prob:.3f} < {threshold}\n")

    return {"accepted": accepted, "top1": (class_names[top1_idx], top1_prob, top1_idx),
            "topk": topk}

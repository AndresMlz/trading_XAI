"""
main_streamlit.py
Interfaz de la herramienta de predicción del SPY en dos pantallas:
1) Pantalla inicial (información general + botón para ejecutar modelos).
2) Pantalla de resultados (velas + Grad-CAM + explicación / drivers / consideraciones).
"""

# Importar las librerías necesarias
import os
import sys
import json
import base64
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import streamlit as st
from interfaz_grafica.funciones_aux_front import obtener_ultima_imagen
#from interpretabilidad_gemini.interpretacion_resultado import interpretacion_resultado
from interpretabilidad_gemini.interpretacion_resultado_gcp import interpretacion_resultado
from interpretabilidad_gemini.rag_gcp import generar_respuesta_rag

# Fuerza fondo blanco siempre
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #ffffff !important;
}

header, .block-container {
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


principal_dir = os.getcwd()
inputs_dir = os.path.join(principal_dir, "inputs")
resultado_stacking_dir = os.path.join(principal_dir, "resultado_stacking")
csv_stacking = os.path.join(resultado_stacking_dir, "resultado_stacking.csv")
csv_enriquecido = os.path.join(inputs_dir, "indicators_sheet_human.csv")
LOGO_PATH = os.path.join(inputs_dir, "logo_javeriana.png")

# HELPERS

def load_image_base64(path):
    """
    Función que carga una imagen desde disco y la codifica en base64.
    """
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def extraer_json(texto: str) -> Optional[Dict[str, Any]]:
    """
    Intenta extraer un JSON válido desde el texto devuelto por Gemini.

    Args:
        texto (str): Texto completo devuelto por el modelo generativo.

    Returns:
        dict | None: Diccionario con el JSON parseado si se encuentra,
        o None si no se pudo extraer.
    """
    # 1) intento directo
    try:
        return json.loads(texto)
    except Exception:  # pylint: disable=broad-except
        pass

    # 2) buscar primer bloque { ... }
    # if "{" not in texto or "}" not in texto:
    #     return texto
    start = texto.find("{")
    end = texto.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = texto[start: end + 1]
        try:
            return json.loads(cand)
        except Exception:  # pylint: disable=broad-except
            return None
    return None


def badge_accion(accion: str) -> str:
    """
    Devuelve un badge HTML coloreado según la acción recomendada.

    Args:
        accion (str): Acción recomendada ('BUY', 'SELL', 'HOLD').

    Returns:
        str: HTML listo para renderizar con st.markdown(..., unsafe_allow_html=True).
    """
    a = (accion or "").strip().upper()
    color = {"BUY": "#16a34a", "SELL": "#ef4444", "HOLD": "#64748b"}.get(a, "#6b7280")
    return (
        f"<span style='background:{color};color:white;padding:4px 12px;"
        f"border-radius:999px;font-weight:700;font-family:Segoe UI'>{a or 'N/A'}</span>"
    )


def badge_confianza(nivel: str) -> str:
    """
    Devuelve un badge HTML coloreado para el nivel de confianza ('bajo','medio','alto').

    Args:
        nivel (str): Nivel textual devuelto por Gemini.

    Returns:
        str: HTML con el badge correspondiente.
    """
    n = (nivel or "").strip().lower()
    color = {"alto": "#16a34a", "medio": "#eab308", "bajo": "#ef4444"}.get(n, "#6b7280")
    txt = n.upper() if n else "DESCONOCIDO"
    return (
        f"<span style='background:{color};color:white;padding:4px 12px;"
        f"border-radius:999px;font-weight:700;font-family:Segoe UI'>{txt}</span>"
    )


def cargar_resultados_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, str,
                                                            Optional[Dict[str, Any]]]:
    """
    Ejecuta tu pipeline (modelos + Gemini) y prepara los objetos de salida.

    IMPORTANTE:
    - Se asume que ya tienes implementada interpretacion_resultado()
      que corre: ejecucion_decision_modelos(), generar_hoja_indicadores(),
      lee csv_stacking/csv_enriquecido y llama a Gemini.
    - Esta función solo envuelve esa lógica y carga los CSV generados.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, str, dict | None]:
            - df_stack: DataFrame con resultados del meta-modelo.
            - df_ctx: DataFrame con hoja de indicadores enriquecida.
            - respuesta_txt: Texto crudo devuelto por Gemini.
            - respuesta_json: JSON parseado (o None si no se pudo extraer).
    """

    respuesta_txt = interpretacion_resultado()

    # 2) Cargar CSVs resultantes
    df_stack = pd.read_csv(csv_stacking)
    df_ctx = pd.read_csv(csv_enriquecido)

    # 3) Extraer JSON de explicación
    respuesta_json = extraer_json(respuesta_txt)

    return df_stack, df_ctx, respuesta_txt, respuesta_json


def preparar_contexto_inferencia(
    df_stack: pd.DataFrame,
    respuesta_json: Optional[Dict[str, Any]],
) -> str:
    """
    Prepara un resumen del contexto de la inferencia actual para incluir
    en las preguntas del chatbot.

    Args:
        df_stack: DataFrame con resultados del meta-modelo.
        respuesta_json: JSON con la explicación de Gemini.

    Returns:
        str: Texto con el contexto de la inferencia actual.
    """
    fila_meta = df_stack.tail(1).iloc[0]
    meta_final = str(fila_meta.get("final_pred", "HOLD")).upper()
    meta_conf = float(fila_meta.get("confidence", 0.0))
    pattern_label = fila_meta.get("candlesticks_pattern", "N/A")
    pattern_conf = float(fila_meta.get("pattern_conf", 0.0))

    contexto = f"""
Contexto de la predicción actual del SPY:

- Acción recomendada: {meta_final}
- Confianza del meta-modelo: {meta_conf:.2%}
- Patrón detectado: {pattern_label} (confianza: {pattern_conf:.2%})
"""

    if respuesta_json:
        explicacion = respuesta_json.get("explicacion_resumida", "")
        drivers = respuesta_json.get("drivers", []) or []
        nivel_conf_text = respuesta_json.get("nivel_confianza", "medio")
        accion_recom = respuesta_json.get("accion_recomendada", meta_final)

        contexto += f"""
- Nivel de confianza: {nivel_conf_text.upper()}
- Explicación: {explicacion}
- Drivers principales:
"""
        for i, driver in enumerate(drivers[:3], 1):  # Solo los primeros 3
            contexto += f"  {i}. {driver}\n"

    return contexto


# UI: CABECERA, LOGO Y FOOTER

def render_header():
    """
    Renderiza la barra superior oscura con el título de la herramienta.
    """
    st.markdown(
        """
        <div style="
            width:100%;
            background-color:#111827;
            padding:18px 24px;
            border-radius:12px;
            text-align:center;
            margin-bottom:24px;">
            <span style="color:white;font-size:32px;font-weight:800;
                         font-family:Segoe UI,system-ui,sans-serif;">
                Herramienta de predicción del SPY
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_logo_fixed():
    """
    Función para renderizar el logo fijo abajo a la derecha.
    """
    if not LOGO_PATH or not os.path.exists(LOGO_PATH):
        return

    img_b64 = load_image_base64(LOGO_PATH)

    st.markdown(
        f"""
        <div style="
            position:fixed;
            right:20px;
            bottom:20px;
            z-index:999;
        ">
            <img src="data:image/png;base64,{img_b64}" style="height:80px;">
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_fixed():
    """
    Footer fijo abajo a la izquierda para todas las pantallas.
    """
    st.markdown(
        """
        <div style="
            position:fixed;
            left:20px;
            bottom:20px;
            z-index:999;
            font-size:12px;
            color:#4b5563;
            font-family:Segoe UI, system-ui, sans-serif;
        ">
            Elaborado por: Edwin Caro, Andres Matallana y Santiago Zafra
        </div>
        """,
        unsafe_allow_html=True,
    )


# PANTALLA 1: INTRO + BOTÓN
def render_pantalla_inicial():
    """
    Renderiza la primera pantalla:
    - Columna izquierda: Información general y consideraciones.
    - Columna derecha: Mensaje y botón para ejecutar el modelo.
    """
    render_header()

    col_left, col_right = st.columns([1.2, 1.5])

    with col_left:
        st.markdown(
            """
            <div style="
                background-color:#e5e7eb;
                border-radius:32px;
                padding:24px 24px 32px 24px;
                font-family:Segoe UI, system-ui, sans-serif;
            ">
              <h3 style="text-align:center;margin-top:0;margin-bottom:12px;">
                Información general
              </h3>
              <p style="font-size:14px;line-height:1.4;">
                La siguiente herramienta utiliza un modelo de reconocimiento de patrones,
                una red neuronal convolucional, un modelo transformer y el uso de IA generativa
                para predecir la acción del precio del activo SPY dentro de 10 minutos a partir
                de los últimos 60 minutos del activo.
              </p>

              <h4 style="margin-top:16px;margin-bottom:8px;text-align:center;">
                Consideraciones
              </h4>
              <ul style="font-size:14px;line-height:1.4;">
                <li>El SPY es uno de los ETF que rastrea el índice S&amp;P 500.</li>
                <li>El horario en que opera la bolsa tradicional va de 8 a.m. a 2 p.m.</li>
                <li>Las predicciones generadas son un apoyo para la toma de decisión al momento
                    de operar, se recomienda responsabilidad y el uso de una estrategia para
                    la gestión del riesgo.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            """
            <div style="margin-top:80px;font-family:Segoe UI, system-ui, sans-serif;
                        font-size:18px;font-weight:600;text-align:center;">
                Pulsa el botón para obtener los precios del activo, correr los modelos
                y obtener explicación.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Creamos 3 columnas y usamos la del centro para centrar el botón
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            run = st.button("🧠 Ejecutar y explicar", type="primary", use_container_width=True)

        if run:
            st.session_state["run_clicked"] = True

    # Logo y footer fijos
    render_logo_fixed()
    render_footer_fixed()


# PANTALLA 2: RESULTADOS

def render_pantalla_resultados(
    df_stack: pd.DataFrame,
    df_ctx: pd.DataFrame,
    respuesta_json: Optional[Dict[str, Any]],
    respuesta_txt: str,
):
    """
    Renderiza la segunda pantalla con:
    - Columna izquierda: velas japonesas + Grad-CAM.
    - Columna derecha: predicción generada, explicación, drivers, consideraciones.
    """
    # Traer las últimas imágenes generadas
    original_img_path = obtener_ultima_imagen("outputs", modo="normal")
    gradcam_img_path = obtener_ultima_imagen("outputs", modo="gradcam")

    render_header()

    # --- Preparar datos principales del meta-modelo ---
    fila_meta = df_stack.tail(1).iloc[0]
    meta_final = str(fila_meta.get("final_pred", "HOLD")).upper()
    meta_conf = float(fila_meta.get("confidence", 0.0))

    # Si Gemini devolvió JSON, extraer campos de ahí; si no, usamos meta-modelo
    if respuesta_json:
        explicacion = respuesta_json.get("explicacion_resumida", "")
        drivers = respuesta_json.get("drivers", []) or []
        contradicciones = respuesta_json.get("contradicciones", []) or []
        nivel_conf_text = respuesta_json.get("nivel_confianza", "medio")
        accion_recom = respuesta_json.get("accion_recomendada", meta_final)
    else:
        explicacion = "No se pudo obtener una explicación estructurada desde Gemini."
        drivers = []
        contradicciones = []
        nivel_conf_text = "medio"
        accion_recom = meta_final

    # --- Layout principal: izquierda (imágenes) / derecha (texto) ---
    col_left, col_right = st.columns([1.1, 1.4])

    # ------------------- Columna izquierda: gráficas -------------------
    with col_left:
        st.subheader("Acción del precio en velas japonesas")
        try:
            st.image(original_img_path, use_container_width=True)
        except Exception:  # pylint: disable=broad-except
            st.info("No se encontró la imagen de velas japonesas.")

        st.markdown("### Reconocimiento de patrones")
        pattern_label = fila_meta.get("candlesticks_pattern", "N/A")
        pattern_conf = float(fila_meta.get("pattern_conf", 0.0))
        st.markdown(
            f"*Grad-CAM: **{pattern_label}** (p={pattern_conf:.2f})*",
            unsafe_allow_html=True,
        )
        try:
            st.image(gradcam_img_path, use_container_width=True)
        except Exception:  # pylint: disable=broad-except
            st.info("No se encontró la imagen de Grad-CAM.")

        with st.expander("Ver últimos indicadores técnicos (opcional)"):
            st.dataframe(df_ctx.tail(10), use_container_width=True)

    # ------------------- Columna derecha: predicción + explicación -------------------
    with col_right:
        st.subheader("Predicción generada")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Acción recomendada**")
            st.markdown(badge_accion(accion_recom), unsafe_allow_html=True)
        with col_b:
            st.markdown("**Nivel de confianza**")
            st.markdown(badge_confianza(nivel_conf_text), unsafe_allow_html=True)
            st.caption(f"Confianza meta-modelo: {meta_conf:.2f}")

        st.markdown("### Explicación")
        st.write(explicacion or "No hay explicación disponible.")

        col_d, col_c = st.columns(2)
        with col_d:
            st.markdown("### Drivers")
            if drivers:
                for d in drivers:
                    st.markdown(f"- {d}")
            else:
                st.markdown("- No se identificaron drivers específicos.")

        with col_c:
            st.markdown("### Consideraciones")
            if contradicciones:
                for c in contradicciones:
                    st.markdown(f"- {c}")
            else:
                st.markdown("- No se encontraron contradicciones destacadas.")

        # --- Botón de nueva ejecución centrado ---
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            rerun = st.button("🔄 Nueva ejecución", use_container_width=True)

    # Si se pulsa "Nueva ejecución", volvemos a la pantalla inicial
    if rerun:
        # Queremos el mismo comportamiento que al pulsar el botón inicial:
        st.session_state["run_clicked"] = True
        st.session_state["resultados_cargados"] = False
        # Resetear el historial del chat cuando se ejecuta una nueva inferencia
        if "chat_history" in st.session_state:
            st.session_state["chat_history"] = []
        st.rerun()

    # ------------------- Sección de Chatbot RAG -------------------
    render_chatbot_section(df_stack, respuesta_json)


def render_chatbot_section(
    df_stack: pd.DataFrame,
    respuesta_json: Optional[Dict[str, Any]],
):
    """
    Renderiza la sección del chatbot RAG para interactuar sobre la inferencia.

    Args:
        df_stack: DataFrame con resultados del meta-modelo.
        respuesta_json: JSON con la explicación de Gemini.
    """
    st.markdown("---")
    st.markdown("### 💬 Chat: Pregunta sobre la predicción")

    # Inicializar el historial del chat si no existe
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Preparar contexto de la inferencia actual
    contexto_inferencia = preparar_contexto_inferencia(df_stack, respuesta_json)

    # Mostrar historial de mensajes
    if st.session_state["chat_history"]:
        st.markdown(
            """
            <div style="
                background-color:#f9fafb;
                border-radius:12px;
                padding:16px;
                margin-bottom:16px;
                max-height:400px;
                overflow-y:auto;
                font-family:Segoe UI, system-ui, sans-serif;
            ">
            """,
            unsafe_allow_html=True,
        )

        for msg in st.session_state["chat_history"]:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                st.markdown(
                    f"""
                    <div style="
                        background-color:#3b82f6;
                        color:white;
                        padding:12px 16px;
                        border-radius:12px;
                        margin-bottom:12px;
                        margin-left:20%;
                        font-size:14px;
                    ">
                        <strong>Tu:</strong><br>{content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:  # model
                st.markdown(
                    f"""
                    <div style="
                        background-color:#e5e7eb;
                        color:#111827;
                        padding:12px 16px;
                        border-radius:12px;
                        margin-bottom:12px;
                        margin-right:20%;
                        font-size:14px;
                    ">
                        <strong>Asistente:</strong><br>{content}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "💡 Haz una pregunta sobre la predicción actual. "
            "El asistente tiene acceso a documentos de trading y puede explicar "
            "conceptos, patrones y estrategias relacionadas con tu predicción."
        )

    # Input para nueva pregunta
    col_input, col_send = st.columns([4, 1])

    with col_input:
        user_question = st.text_input(
            "Escribe tu pregunta:",
            key="chat_input",
            placeholder="Ej: ¿Por qué se recomienda esta acción? ¿Qué significa este patrón?",
            label_visibility="collapsed",
        )

    with col_send:
        st.markdown("<br>", unsafe_allow_html=True)
        send_button = st.button("Enviar", type="primary", use_container_width=True)

    # Botón para limpiar el chat
    if st.session_state["chat_history"]:
        clear_button = st.button("🗑️ Limpiar chat", use_container_width=False)
        if clear_button:
            st.session_state["chat_history"] = []
            st.rerun()

    # Procesar pregunta cuando se envía
    if send_button and user_question.strip():
        # Preparar pregunta con contexto de inferencia
        pregunta_con_contexto = f"""
{contexto_inferencia}

Pregunta del usuario: {user_question.strip()}
"""

        # Agregar pregunta del usuario al historial
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_question.strip()}
        )

        # Mostrar indicador de carga
        with st.spinner("🤔 El asistente está pensando..."):
            try:
                # Llamar al RAG
                respuesta = generar_respuesta_rag(pregunta_con_contexto)

                # Agregar respuesta al historial
                st.session_state["chat_history"].append(
                    {"role": "model", "content": respuesta}
                )

                # Forzar rerun para mostrar la respuesta
                st.rerun()

            except Exception as e:  # pylint: disable=broad-except
                error_msg = f"Error al obtener respuesta: {str(e)}"
                st.error(error_msg)
                # Agregar error al historial
                st.session_state["chat_history"].append(
                    {"role": "model", "content": error_msg}
                )


# Función main
def render_interpretacion_ui():
    """
    Función principal de la app Streamlit.
    Controla el flujo:
    - Si aún no se ha hecho clic en "Ejecutar y explicar": muestra pantalla inicial.
    - Si ya se hizo clic y se han cargado resultados: muestra pantalla de resultados.
    """
    st.set_page_config(
        page_title="Herramienta de predicción del SPY",
        page_icon="📊",
        layout="wide",
    )

    # Inicializar flags de estado
    if "run_clicked" not in st.session_state:
        st.session_state["run_clicked"] = False
    if "resultados_cargados" not in st.session_state:
        st.session_state["resultados_cargados"] = False

    # Si todavía no hemos ejecutado el pipeline → pantalla inicial
    if not st.session_state["run_clicked"]:
        render_pantalla_inicial()
        return

    # Si ya se pulsó el botón pero aún no se cargaron resultados, hacer la inferencia
    if st.session_state["run_clicked"] and not st.session_state["resultados_cargados"]:
        with st.spinner("Ejecutando modelos y generando explicación..."):
            try:
                df_stack, df_ctx, resp_txt, resp_json = cargar_resultados_pipeline()
                st.session_state["df_stack"] = df_stack
                st.session_state["df_ctx"] = df_ctx
                st.session_state["resp_txt"] = resp_txt
                st.session_state["resp_json"] = resp_json
                st.session_state["resultados_cargados"] = True
            except Exception as e:  # pylint: disable=broad-except
                st.error(f"Ocurrió un error ejecutando el pipeline: {e}")
                st.session_state["run_clicked"] = False
                return

    # Mostrar la pantalla de resultados con lo que hay en session_state
    render_pantalla_resultados(
        st.session_state["df_stack"],
        st.session_state["df_ctx"],
        st.session_state["resp_json"],
        st.session_state["resp_txt"],
    )

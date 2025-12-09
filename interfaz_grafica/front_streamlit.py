"""
Modulo que contiene la interfaz grafica realizada
para mostrar al usuario el sistema de trading multimodelo.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""
# Importar las librerias necesarias
import os
import json
import streamlit as st
import pandas as pd
from interfaz_grafica.funciones_aux_front import obtener_ultima_imagen
from interpretabilidad_gemini.interpretacion_resultado import interpretacion_resultado

# Definir los paths necesarios
principal_dir = os.getcwd()
inputs_dir = os.path.join(principal_dir, "inputs")
resultado_stacking_dir = os.path.join(principal_dir, "resultado_stacking")
csv_enriquecido = os.path.join(inputs_dir, "indicators_sheet_human.csv")
stacking_csv = os.path.join(resultado_stacking_dir, "resultado_stacking.csv")

# Funciones auxiliares
def _extraer_json(texto: str):
    """
    Intenta recuperar un JSON válido desde un texto de LLM.
    1) intento json.loads directo
    2) busca el primer bloque {...} y carga
    3) si no puede, retorna None
    """
    # 1) intento directo
    try:
        return json.loads(texto)
    except Exception: # pylint: disable=broad-except
        pass

    # 2) extraer el primer bloque { ... } (simple y efectivo)
    start = texto.find("{")
    end   = texto.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidato = texto[start:end+1]
        try:
            return json.loads(candidato)
        except Exception: # pylint: disable=broad-except
            return None
    return None

# Función auxiliar para badges de confianza
def _badge_confianza(nivel: str) -> str:
    """
    Función que devuelve un badge HTML con el nivel de confianza coloreado.
    """
    nivel = (nivel or "").strip().lower()
    color = {"alto": "#16a34a", "medio": "#eab308", "bajo": "#ef4444"
                                                            }.get(nivel, "#6b7280")
    txt   = nivel.upper() or "DESCONOCIDO"
    return (
        f"<span style='background:{color};"
        "color:white;padding:3px 8px;border-radius:6px;font-weight:600'>"
        f"{txt}</span>"
    )

# Función auxiliar para badges de acción
def _badge_accion(accion: str) -> str:
    """
    Función que devuelve un badge HTML con la acción recomendada coloreada.
    """
    a = (accion or "").strip().upper()
    color = {"BUY": "#16a34a", "SELL": "#ef4444", "HOLD": "#64748b"}.get(a, "#6b7280")
    return (
        f"<span style='background:{color};"
        "color:white;padding:3px 8px;border-radius:6px;font-weight:600'>"
        f"{a or 'N/A'}</span>"
    )


# Función principal de la interfaz gráfica
def render_interpretacion_ui():
    """
    Crea el front de Streamlit para:
    - Ejecutar tu pipeline (stacking + hoja de indicadores + Gemini)
    - Mostrar la salida estructurada y las tablas base
    - Ofrecer descargas y visuales
    Requiere que existan:
      - interpretacion_resultado()
      - variables csv_enriquecido, csv_stacking (rutas a CSVs generados)
      - (opcional) rutas a imágenes/gradcam si deseas mostrarlas
    """
    # Traer las últimas imágenes generadas
    original_img_path = obtener_ultima_imagen("outputs", modo="normal")
    gradcam_img_path  = obtener_ultima_imagen("outputs", modo="gradcam")

    st.header("🧠 Interpretación (Gemini) y Evidencias")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        threshold = st.slider("Umbral operativo (solo informativo)", 0.50, 0.95, 0.68, 0.01)
    with col_r:
        run_btn = st.button("🚀 Ejecutar y explicar")

    if not run_btn:
        st.info("Pulsa el botón para correr modelos, generar hoja y obtener explicación.")
        return

    with st.status("Corriendo pipeline y preparando explicación...", expanded=True) as status:
        try:
            # 1) Ejecuta tu función (ya corre todo y devuelve texto de Gemini)
            respuesta_txt = interpretacion_resultado()
            st.write("✅ Pipeline ejecutado. Recibida respuesta de Gemini.")

            # 2) Parseo robusto a JSON
            parsed = _extraer_json(respuesta_txt)
            if parsed is None:
                st.warning("No se pudo extraer un JSON válido. Se muestra el texto bruto.")
                st.code(respuesta_txt)
            else:
                st.write("✅ JSON válido extraído.")
            status.update(label="✅ Listo", state="complete")

        except Exception as e: # pylint: disable=broad-except
            status.update(label="❌ Error", state="error")
            st.error(f"Ocurrió un error al ejecutar la interpretación: {e}")
            return

    # --- Panel de interpretación (si hay JSON estructurado) ---
    st.subheader("📄 Resultado de Gemini")

    if parsed is not None:
        # métricas y badges
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Acción recomendada**")
            st.markdown(_badge_accion(parsed.get("accion_recomendada")), unsafe_allow_html=True)
        with c2:
            st.markdown("**Nivel de confianza**")
            st.markdown(_badge_confianza(parsed.get("nivel_confianza")), unsafe_allow_html=True)

        st.markdown("**Explicación resumida**")
        st.write(parsed.get("explicacion_resumida", "—"))

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Drivers**")
            for d in parsed.get("drivers", []) or []:
                st.markdown(f"- {d}")
        with col_b:
            st.markdown("**Posibles contradicciones**")
            for c in parsed.get("contradicciones", []) or []:
                st.markdown(f"- {c}")

        with st.expander("Ver JSON completo"):
            st.json(parsed)
    else:
        # fallback: texto bruto ya se mostró
        pass

    st.markdown("---")

    # --- Evidencias cuantitativas (tus CSVs) ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Resultados del Meta-Modelo (stacking)")
        try:
            df_stack = pd.read_csv(stacking_csv)
            st.dataframe(df_stack.tail(20), use_container_width=True)
            st.download_button("⬇️ Descargar stacking CSV", df_stack.to_csv(index=False
                                                                ), "stacking.csv", "text/csv")
        except Exception as e: # pylint: disable=broad-except
            st.warning(f"No fue posible leer '{stacking_csv}': {e}")

    with col_b:
        st.markdown("### 🧾 Hoja de Indicadores (última fila)")
        try:
            df_ctx = pd.read_csv(csv_enriquecido)
            st.dataframe(df_ctx.tail(20), use_container_width=True)
            st.download_button("⬇️ Descargar hoja de indicadores", df_ctx.to_csv(index=False
                                                        ), "indicators_sheet.csv", "text/csv")
        except Exception as e: # pylint: disable=broad-except
            st.warning(f"No fue posible leer '{csv_enriquecido}': {e}")

    # --- Evidencias visuales (opcional) ---
    st.markdown("---")
    st.markdown("### 🖼️ Evidencias visuales")
    cimg1, cimg2 = st.columns(2)
    with cimg1:
        try:
            st.image(original_img_path, caption="Gráfico de velas original",
                                                use_container_width=True)
        except Exception: # pylint: disable=broad-except
            st.info("No se encontró la imagen de velas.")
    with cimg2:
        try:
            st.image(gradcam_img_path, caption="Grad-CAM / patrón detectado",
                                                use_container_width=True)
        except Exception: # pylint: disable=broad-except
            st.info("No se encontró la imagen Grad-CAM.")

    # --- Texto original de Gemini (si quieres conservarlo abajo) ---
    with st.expander("Ver respuesta de Gemini (texto crudo)"):
        st.code(respuesta_txt)

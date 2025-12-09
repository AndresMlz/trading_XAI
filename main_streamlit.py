"""
Módulo que contiene la función principal de todo el
proyecto
"""

# Importar las librerías necesarias
from config.constantes import Constantes as CONST
from interfaz_grafica.front_streamlit2 import render_interpretacion_ui

# Función principal para correr la interfaz gráfica
def main():
    """
    Función principal para correr la interfaz gráfica
    """
    if CONST.DATA["Boton_bot"] == "On":
        render_interpretacion_ui()
    else:
        print("No se tiene activado el botón para correr la interfaz gráfica.")

if __name__ == "__main__":
    main()

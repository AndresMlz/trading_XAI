"""
Modulo lanzador para la aplicación Streamlit.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import os
import sys
from streamlit.web import cli as stcli

def get_base_dir() -> str:
    """
    Carpeta base:
    - si es .exe: donde está launcher.exe
    - si es script: donde está launcher.py
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    base_dir = get_base_dir()
    os.chdir(base_dir)

    app_path = os.path.join(base_dir, "main_streamlit.py")
    if not os.path.exists(app_path):
        print("❌ No se encontró main_streamlit.py en:", app_path)
        input("Pulsa ENTER para salir...")
        return

    # Sin --server.port: usa el puerto definido en config.toml o el 8501 por defecto
    sys.argv = ["streamlit", "run", app_path]
    stcli.main()

if __name__ == "__main__":
    main()

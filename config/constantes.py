"""
Modulo que contiene la configuración de las constantes del proyecto.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerias necesarias
import sys
import os
import json
from config.autenticacion_gsuite import sheet_to_dict

# Obtenemos nombre de archivo y ruta de configuración
NOMBRE_ARCHIVO = os.path.basename(__file__)

# Clase constantes
class Constantes:
    """
    Clase para la inicialización de constantes
    """

    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Constantes, cls).__new__(cls, *args, **kwargs)
            cls._instance.inicializar_constantes()
        return cls._instance


    @classmethod
    def inicializar_constantes(cls):
        """
        Inicializa las constantes de la hiperautomatización

        returns
        estado (bln): Estado de la ejecución de la función
        observacion (str): Observación de la ejecución de la función
        """
        try:
            # Obtener la ruta raiz del proyecto
            ruta_raiz = os.getcwd()

            # Inicialización la ruta del archivo de parámetros técnicos
            ruta_config = os.path.join(ruta_raiz, "config", "config.json")

            # Lectura del archivo de parámetros técnicos
            cls.DATA = cls.get_parameters(ruta_config)

            # Inicialización de los parámetros de negocio de la hoja de cálculo
            estado, cls.DATA_NEGOCIO = sheet_to_dict(
                cls.DATA["sheets_parametros_id"],
                cls.DATA["nombre_hoja_parametros"],
            )

            # Si se presentó al obtener datos de la sheet, se detiene la hiperautomatización
            if not estado:
                observacion = f"No se pudo obtener los parámetros de negocio.\
                        observación: {cls.DATA_NEGOCIO}"
                raise ValueError(observacion)

            # Unifica los parametros Json y Sheets
            cls.DATA.update(cls.DATA_NEGOCIO)

            return True, "Constantes inicializadas con éxito"

        except Exception as e: #pylint: disable=broad-except
            _, _, exc_tb = sys.exc_info()
            error_line = exc_tb.tb_lineno
            return False, f"Error en la línea {error_line}: {str(e)}"


    @classmethod
    def get_parameters(cls, ruta_config):
        """
        Obtener parametros técnicos del json

        :param ruta_config: Ruta del archivo de parámetros técnicos

        :return: Diccionario JSON con los parámetros técnicos
        """
        # Lectura del archivo de parámetros técnicos
        with open(ruta_config, "r", encoding="utf-8") as archivo:
            # Inicialización del diccionario JSON
            return json.load(archivo)

constantes_instance = Constantes()

"""
Modulo para realizar la autenticación en GSuite.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
from typing import Any
from typing import Union
from typing import cast
import os
import sys
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import gspread

# Direcciones necesarias para la autenticación
dir_principal = os.getcwd()
dir_config = os.path.join(dir_principal, "config")

# Definir constantes
ruta_token = os.path.join(dir_config, "token.json")
ruta_credenciales = os.path.join(dir_config, "credentials.json")

# Función de autenticación para acceder a la GSuite
def authenticate() -> Union[tuple[Any, Any, Any], tuple[None, None, None]]:
    """
    Autentica con OAuth2 y configura los clientes para gspread y BigQuery.

    Returns:
        creds: credenciales de autenticación
        gspread_client: cliente para trabajar con Google Sheets
    """
    scopes = [
                "https://www.googleapis.com/auth/script.projects",
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/calendar.events",
                "https://www.googleapis.com/auth/calendar.events.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/calendar.settings.readonly",
                "https://www.googleapis.com/auth/script.external_request",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.compose",
                "https://www.googleapis.com/auth/gmail.modify",
                "https://mail.google.com/",
                "https://www.googleapis.com/auth/gmail.addons.current.action.compose",
                "https://www.googleapis.com/auth/script.send_mail",
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/admin.directory.user",
                "https://www.googleapis.com/auth/ediscovery",
                "https://www.googleapis.com/auth/admin.datatransfer",
    ]
    try:
        creds = None
        # Verifica si el archivo token.pickle existe en folder_path
        if os.path.exists(ruta_token):
            creds = Credentials.from_authorized_user_file(ruta_token, scopes)

        # Si no hay credenciales válidas, iniciar el flujo de autenticación
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(ruta_credenciales, scopes)
                creds = flow.run_local_server(port=0)
            # Guardar las credenciales para usos futuros
            with open(ruta_token, 'w', encoding = 'utf-8') as token:
                token.write(creds.to_json())
        # Configurar el cliente de Google Sheets
        gspread_client = gspread.authorize(creds)
        # Configurar el cliente de Google Drive (casteado para silenciar analizadores estáticos)
        drive_service = cast(Any, build('drive', 'v3', credentials=creds))
        return creds, gspread_client, drive_service
    except Exception as e: # pylint: disable=broad-except
        print(e)
        return None, None, None

# Función para convertir un sheets a un diccionario
def sheet_to_dict(id_sheet, nombre_hoja):
    """
    Función principal para leer datos de una hoja de cálculo de Google Sheets.

    Args:
    - id_sheet (str): ID de la hoja de cálculo.
    - nombre_hoja (str): Nombre de la hoja dentro de la hoja de cálculo.
    Returns:
    - dict: Un diccionario con los datos obtenidos de la hoja de cálculo.
      Las claves del diccionario son los encabezados de las columnas y los valores
      son las filas correspondientes.
    """
    try:
        
        creds, _, _ = authenticate()
        service = cast(Any, build('sheets', 'v4', credentials=creds))
        sheet = service.spreadsheets().values() # type: ignore
        result = sheet.get(spreadsheetId=id_sheet, range=nombre_hoja).execute()
        values = result.get("values", [])[1:]  # Omite la primera fila

        if not values:
            return False,'No se encontro informacion en google sheet'

        try:
            for i in range(len(values)):
                values[i] = [str(x) for x in values[i]]
                if len(values[i]) == 1:
                    values[i].append('')

            return True, dict(values)
        except ValueError as e:
            return False, e

    except Exception as e: #pylint: disable=broad-except
        _, _, exc_tb = sys.exc_info()
        linea_error = exc_tb.tb_lineno
        error = f"Error al ejecutar la funcion: {str(e)} en la línea {linea_error}"
        return False, error

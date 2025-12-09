"""
Módulo para descargar la última hora de datos del SPY
desde Alpaca.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerias necesarias
import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from config.constantes import Constantes as CONST

# Definir los paths necesarios
dir_principal = os.getcwd()
dir_inputs = os.path.join(dir_principal, "inputs")

# Establecer el cliente de Alpaca
client = StockHistoricalDataClient(CONST.DATA["Alpaca_key"], CONST.DATA["Alpaca_secret"])

# Función para descargar y guardar los datos de los últimos 60 minutos del SPY 
def get_spy_data(filename, datos_sol,ultimos_minutos=True):
    """
    Obtiene datos del SPY desde Alpaca y los guarda en un archivo CSV.
    
    Args:
        filename: Nombre del archivo CSV de salida
        datos_sol: Solicitud de barras configurada
    Returns:
        df (pd.DataFrame): DataFrame con los datos descargados del SPY
    """
    # Obtener las barras de datos
    bars = client.get_stock_bars(datos_sol)
    df = bars.df.copy()

    # Aplanar índice si es necesario
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Renombrar columnas
    rename_map = {
        'timestamp': 'date',
        'open': 'SPY_open',
        'high': 'SPY_high',
        'low': 'SPY_low',
        'close': 'SPY_close',
        'volume': 'SPY_volume'
    }
    df = df.rename(columns=rename_map)

    # Asegurar que 'date' existe
    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'date'})

    # Mantener solo columnas relevantes
    cols = [c for c in ['date', 'SPY_open', 'SPY_high', 'SPY_low', 'SPY_close', 'SPY_volume'
                      ] if c in df.columns]
    if ultimos_minutos:
        df = df[cols].sort_values('date').tail(60).reset_index(drop=True)

    # Guardar el dataframe como un csv en el directorio especificado
    output_path = os.path.join(dir_inputs, filename)
    df.to_csv(output_path, index=False)

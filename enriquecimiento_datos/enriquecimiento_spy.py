"""
Modulo que contiene las funciones necesarias para
enriquecer el csv que contiene los datos del SPY
"""

# Importar las librerías necesarias
import os
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from data_alpaca.descarga_data import get_spy_data

# Paths necesarios
principal_dir = os.getcwd()
inputs_dir = os.path.join(principal_dir, "inputs")
csv_path = os.path.join(inputs_dir, "SPY_last60d.csv")

# Pedir los datos necesarios
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=60)

# Crear la solicitud de barras
request = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=start_time,
    end=end_time,
    feed=DataFeed.IEX  # plan gratuito
)

# Clase params
@dataclass
class Params:
    """
    Dataclass con los hiperparámetros de la estrategia de indicadores técnicos.

    Attributes
    ----------
    ema_fast : int
        Periodo de la EMA rápida (por defecto 12).
    ema_mid : int
        Periodo de la EMA media (por defecto 22).
    ema_slow : int
        Periodo de la EMA lenta (por defecto 50).
    bb_period : int
        Periodo de cálculo de las Bandas de Bollinger (por defecto 20).
    bb_mult : float
        Multiplicador de desviación estándar para las bandas (por defecto 2.0).
    use_dynamic_width : bool
        Si True, el umbral de apertura de bandas se calcula dinámicamente mediante cuantiles.
    width_quantile : float
        Cuantil de referencia para detectar “apertura” de bandas.
    width_window : int
        Ventana de cálculo del cuantíl de anchura.
    min_width : float
        Umbral mínimo de anchura para considerar “apertura” si no se usa método dinámico.
    width_slope_k : int
        Paso temporal usado para medir cambio de anchura.
    slope_window : int
        Ventana para medir pendiente mínima en tendencia EMA.
    slope_min : float
        Mínimo valor relativo de pendiente para considerar tendencia válida.
    require_both_tf_open : bool
        Si True, exige apertura simultánea de bandas en 15m y 60m.
    touch_tol_up : float
        Tolerancia multiplicativa superior para detectar toques de EMA9.
    touch_tol_dn : float
        Tolerancia multiplicativa inferior para detectar toques de EMA9.
    """
    ema_fast: int = 12
    ema_mid: int  = 22
    ema_slow: int = 50
    bb_period: int = 20
    bb_mult: float = 2.0
    use_dynamic_width: bool = True
    width_quantile: float = 0.70
    width_window: int = 120
    min_width: float = 0.006
    width_slope_k: int = 2
    slope_window: int = 5
    slope_min: float = 0.0025
    require_both_tf_open: bool = False
    touch_tol_up: float = 1.0002
    touch_tol_dn: float = 0.9998

P = Params()

# ---------- Utilidades ----------
def ema(s: pd.Series, n: int) -> pd.Series:
    """
    Calcula una media móvil exponencial (EMA).

    Args:
        s (pd.Series): Serie temporal de precios o valores.
        n (int): Longitud de la ventana exponencial.

    Returns:
        pd.Series: Serie de la EMA con misma longitud que la entrada.
    """
    return s.ewm(span=n, adjust=False).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series,
                                                                        pd.Series, pd.Series]:
    """
    Calcula las Bandas de Bollinger y la anchura normalizada.

    Args:
        s (pd.Series): Serie de precios (generalmente 'SPY_close').
        n (int): Ventana de la media móvil simple. Default 20.
        k (float): Multiplicador de desviación estándar. Default 2.0.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            (mid, upper, lower, width)
            - mid: media móvil simple.
            - upper: banda superior.
            - lower: banda inferior.
            - width: anchura relativa (upper - lower) / mid.
    """
    mid = s.rolling(n).mean()
    std = s.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / mid
    return mid, upper, lower, width


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Reagrega datos OHLC a un marco temporal superior (por ejemplo, 15m o 60m).

    Args:
        df (pd.DataFrame): DataFrame con columnas ['SPY_open','SPY_high','SPY_low','SPY_close']
        y un índice datetime.
        rule (str): Regla de resampleo compatible con pandas ('15min', '1H', etc.).

    Returns:
        pd.DataFrame: DataFrame reamuestrado con OHLC correspondientes.
    """
    o = df['SPY_open'].resample(rule).first()
    h = df['SPY_high'].resample(rule).max()
    l = df['SPY_low'].resample(rule).min()
    c = df['SPY_close'].resample(rule).last()
    return pd.DataFrame({'SPY_open': o, 'SPY_high': h, 'SPY_low': l, 'SPY_close': c})


def make_llm_rationale(row: pd.Series) -> Tuple[str, str, str]:
    """
    Genera explicaciones textuales compactas y determinísticas para LLMs.

    Args:
        row (pd.Series): Fila individual del DataFrame con las columnas de contexto
        (bandas, tendencias, EMAs, etc.).

    Returns:
        Tuple[str, str, str]:
            - rationale_long: Razones a favor de una posición larga.
            - rationale_short: Razones a favor de una posición corta.
            - state: Estado resumido del contexto técnico en formato texto.
    """
    up15 = int(row['bb_open_up_15']); dn15 = int(row['bb_open_dn_15'])
    up60 = int(row['bb_open_up_60']); dn60 = int(row['bb_open_dn_60'])
    trend_up = int(row['trend_up']); trend_dn = int(row['trend_down'])
    pos15 = row['bb_pos_15']; pos60 = row['bb_pos_60']
    e9 = row['ema9']; e20 = row['ema20']

    state = (
        f"15m(open_up={up15}, open_dn={dn15}), "
        f"60m(open_up={up60}, open_dn={dn60}), "
        f"trend_up={trend_up}, trend_down={trend_dn}, "
        f"bb_pos_15={pos15:.2f}, bb_pos_60={pos60:.2f}, "
        f"ema9_vs_ema20={'above' if e9>e20 else 'below' if e9<e20 else 'equal'}"
    )

    long_reasons = []
    if up15 or up60: long_reasons.append("HTF_bands_open_up")
    if trend_up: long_reasons.append("trend_up_1m")
    if row['touch_ema9_up']: long_reasons.append("touch_or_cross_ema9_up")
    if pos15 is not None and pos15 < 0.85: long_reasons.append("headroom_to_15m_upper")
    rationale_long = ";".join(long_reasons) if long_reasons else "no_strong_long_reasons"

    short_reasons = []
    if dn15 or dn60: short_reasons.append("HTF_bands_open_down")
    if trend_dn: short_reasons.append("trend_down_1m")
    if row['touch_ema9_dn']: short_reasons.append("touch_or_cross_ema9_dn")
    if pos15 is not None and pos15 > 0.15: short_reasons.append("room_to_15m_lower")
    rationale_short = ";".join(short_reasons) if short_reasons else "no_strong_short_reasons"

    return rationale_long, rationale_short, state

def add_bb_ctx(dfr: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Calcula indicadores de Bandas de Bollinger y EMAs para un contexto temporal dado.

    Args:
        dfr (pd.DataFrame): DataFrame OHLC del timeframe alto.
        label (str): Etiqueta de sufijo (ej. '15' o '60') para nombrar columnas.

    Returns:
        pd.DataFrame: DataFrame con columnas ['bb_mid_LABEL','bb_up_LABEL','bb_lo_LABEL',
        'bb_w_LABEL','ema20_LABEL','ema40_LABEL'].
    """
    mid, up, lo, width = bollinger(dfr['SPY_close'], P.bb_period, P.bb_mult)
    out = pd.DataFrame({
        f'bb_mid_{label}': mid,
        f'bb_up_{label}':  up,
        f'bb_lo_{label}':  lo,
        f'bb_w_{label}':   width,
        f'ema20_{label}':  ema(dfr['SPY_close'], P.ema_mid),
        f'ema40_{label}':  ema(dfr['SPY_close'], P.ema_slow),
    }, index=dfr.index)
    return out


def open_flags(ctx: pd.DataFrame, label: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Determina si las Bandas de Bollinger están “abiertas” en una dirección.

    Args:
        ctx (pd.DataFrame): DataFrame con columnas de contexto (bb_mid_LABEL, bb_w_LABEL, etc.).
        label (str): Sufijo identificador del timeframe ('15', '60').

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]:
            - open_up: flag 1/0 de apertura al alza.
            - open_dn: flag 1/0 de apertura a la baja.
            - open_any: flag 1/0 de apertura sin dirección.
    """
    w = ctx[f'bb_w_{label}']; mid = ctx[f'bb_mid_{label}']
    if P.use_dynamic_width:
        thr = w.rolling(P.width_window, min_periods=10).quantile(P.width_quantile)
    else:
        thr = pd.Series(P.min_width, index=w.index)
    ws = w - w.shift(P.width_slope_k)
    up_trend = (mid - mid.shift(P.width_slope_k)) > 0
    dn_trend = (mid - mid.shift(P.width_slope_k)) < 0
    open_any = (w > thr) & (ws > 0)
    return (open_any & up_trend).astype(int), (open_any & dn_trend).astype(int
                                                                        ), open_any.astype(int)


def band_pos(close: pd.Series, mid: pd.Series, up: pd.Series, lo) -> pd.Series:
    """
    Calcula la posición del precio dentro de las Bandas de Bollinger normalizada a [-1, +1].

    Args:
        close (pd.Series): Serie de precios de cierre.
        mid (pd.Series): Línea media de Bollinger.
        up (pd.Series): Banda superior.
        lo (pd.Series): Banda inferior.

    Returns:
        pd.Series: Posición normalizada:
            -1 = banda inferior, 0 = media, +1 = banda superior.
    """
    span = (up - mid).replace(0, np.nan)
    pos = (close - mid) / span
    return pos.clip(lower=-1, upper=+1)


def long_ready(row: pd.Series) -> bool:
    """
    Determina si se cumplen condiciones de entrada larga.

    Args:
        row (pd.Series): Fila con las columnas de contexto (bandas, tendencia, toques).

    Returns:
        bool: True si el contexto y las condiciones indican posible entrada larga.
    """
    if P.require_both_tf_open:
        ok_ctx = (row['bb_open_up_15']==1) and (row['bb_open_up_60']==1)
    else:
        ok_ctx = (row['bb_open_up_15']==1) or (row['bb_open_up_60']==1)
    return bool(ok_ctx and (row['trend_up']==1) and (row['touch_ema9_up']))


def short_ready(row: pd.Series) -> bool:
    """
    Determina si se cumplen condiciones de entrada corta.

    Args:
        row (pd.Series): Fila con las columnas de contexto (bandas, tendencia, toques).

    Returns:
        bool: True si el contexto y las condiciones indican posible entrada corta.
    """
    if P.require_both_tf_open:
        ok_ctx = (row['bb_open_dn_15']==1) and (row['bb_open_dn_60']==1)
    else:
        ok_ctx = (row['bb_open_dn_15']==1) or (row['bb_open_dn_60']==1)
    return bool(ok_ctx and (row['trend_down']==1) and (row['touch_ema9_dn']))


def conf_long(row: pd.Series) -> float:
    """
    Evalúa confianza heurística (0–1) para señales largas.

    Args:
        row (pd.Series): Fila del DataFrame con todas las columnas de contexto.

    Returns:
        float: Nivel de confianza en [0,1], promedio de 4 condiciones cumplidas.
    """
    ctx = (row['bb_open_up_15']==1) + (row['bb_open_up_60']==1)
    ctx = 1 if ( (P.require_both_tf_open and ctx==2) or ((not P.require_both_tf_open) and ctx>=1
                                                                                        )) else 0
    headroom = 1 if (pd.notna(row['bb_pos_15']) and row['bb_pos_15'] < 0.85) else 0
    return (ctx + row['trend_up'] + int(row['touch_ema9_up']) + headroom) / 4.0


def conf_short(row: pd.Series) -> float:
    """
    Evalúa confianza heurística (0–1) para señales cortas.

    Args:
        row (pd.Series): Fila del DataFrame con todas las columnas de contexto.

    Returns:
        float: Nivel de confianza en [0,1], promedio de 4 condiciones cumplidas.
    """
    ctx = (row['bb_open_dn_15']==1) + (row['bb_open_dn_60']==1)
    ctx = 1 if ( (P.require_both_tf_open and ctx==2) or ((not P.require_both_tf_open) and ctx>=1
                                                                                        )) else 0
    room = 1 if (pd.notna(row['bb_pos_15']) and row['bb_pos_15'] > 0.15) else 0
    return (ctx + row['trend_down'] + int(row['touch_ema9_dn']) + room) / 4.0


def pick_signal(row: pd.Series) -> Tuple[str, float]:
    """
    Decide la señal final ('long', 'short' o 'flat') y su confianza asociada.

    Args:
        row (pd.Series): Fila con señales booleanas y niveles de confianza parciales.

    Returns:
        Tuple[str, float]:
            - signal (str): 'long', 'short' o 'flat'.
            - confidence (float): Nivel de confianza correspondiente.
    """
    if row['long_signal'] and (row['conf_long'] >= max(0.5, row['conf_short'])):
        return 'long', float(row['conf_long'])
    if row['short_signal'] and (row['conf_short'] > row['conf_long']):
        return 'short', float(row['conf_short'])
    return 'flat', float(max(row['conf_long'], row['conf_short'])*0.5)

# Función para generar la hoja
def generar_hoja_indicadores(out_path: str = "indicators_sheet_human.csv") -> pd.DataFrame:
    """
    Orquesta todo el proceso de generación de la hoja de indicadores "LLM-ready":
    carga los datos, calcula indicadores técnicos multi-temporales (1m, 15m, 60m),
    señales heurísticas, racionales explicativos y guarda el resultado final en CSV.

    Args:
        out_path (str, optional):
            Ruta del archivo CSV de salida. Default "indicators_sheet_human.csv".

    Returns:
        pd.DataFrame:
            DataFrame final con todos los indicadores, señales y racionales listos
            para consumo humano o modelos LLM. Incluye columnas de OHLC, EMAs,
            Bandas de Bollinger multi-temporales, flags de tendencia, señales,
            niveles de confianza y explicaciones textuales.
    """
    # 0) Descargar los datos de Alpaca
    get_spy_data("SPY_last60d.csv",request,False)
    # 1) CARGA Y VALIDACIÓN
    df_raw = pd.read_csv(csv_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'], utc=True, errors='coerce')
    df = df_raw.dropna(subset=['date']).set_index('date').sort_index()[
                                                ['SPY_open','SPY_high','SPY_low','SPY_close']]

    # Filtra últimas 4 semanas
    end = df.index.max()
    start = end - pd.Timedelta(days=28)
    df = df.loc[start:end].copy()
    assert not df.empty, "No hay datos en las últimas 4 semanas"

    # 2) FEATURES EN 1 MINUTO
    df['ema9']  = ema(df['SPY_close'], P.ema_fast)
    df['ema20'] = ema(df['SPY_close'], P.ema_mid)
    df['ema40'] = ema(df['SPY_close'], P.ema_slow)

    df['dist_ema9_pct']  = (df['SPY_close']/df['ema9']  - 1.0)
    df['dist_ema20_pct'] = (df['SPY_close']/df['ema20'] - 1.0)
    df['ema9_above_ema20'] = (df['ema9'] > df['ema20']).astype(int)

    # 3) CONTEXTO 15M / 60M
    df15 = resample_ohlc(df, '15min').dropna()
    df60 = resample_ohlc(df, '60min').dropna()
    ctx15 = add_bb_ctx(df15, '15').shift(1)
    ctx60 = add_bb_ctx(df60, '60').shift(1)
    up15, dn15, any15 = open_flags(ctx15, '15')
    up60, dn60, any60 = open_flags(ctx60, '60')

    # Forward-fill al índice 1m
    df = df.join(ctx15.reindex(df.index, method='ffill'))
    df = df.join(ctx60.reindex(df.index, method='ffill'))
    df['bb_open_up_15']  = up15.reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bb_open_dn_15']  = dn15.reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bb_open_any_15'] = any15.reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bb_open_up_60']  = up60.reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bb_open_dn_60']  = dn60.reindex(df.index, method='ffill').fillna(0).astype(int)
    df['bb_open_any_60'] = any60.reindex(df.index, method='ffill').fillna(0).astype(int)

    # Posición relativa dentro de la banda
    df['bb_pos_15'] = band_pos(df['SPY_close'], df['bb_mid_15'], df['bb_up_15'], df['bb_lo_15'])
    df['bb_pos_60'] = band_pos(df['SPY_close'], df['bb_mid_60'], df['bb_up_60'], df['bb_lo_60'])

    # 4) TENDENCIAS Y TOQUES
    slope = (df['ema20'] - df['ema20'].shift(P.slope_window)).abs() / df['ema20']
    df['trend_up']   = ((df['ema20'] > df['ema40']) & (slope > P.slope_min)).astype(int)
    df['trend_down'] = ((df['ema20'] < df['ema40']) & (slope > P.slope_min)).astype(int)

    prev_close = df['SPY_close'].shift(1)
    prev_low   = df['SPY_low'].shift(1)
    prev_high  = df['SPY_high'].shift(1)
    ema9_prev  = df['ema9'].shift(1)
    df['touch_ema9_up'] = ((prev_low <= ema9_prev * P.touch_tol_up) & 
                           (df['SPY_close'] > ema9_prev) & 
                           (df['SPY_close'] > prev_close)).fillna(False)
    df['touch_ema9_dn'] = ((prev_high >= ema9_prev * P.touch_tol_dn) & 
                           (df['SPY_close'] < ema9_prev) & 
                           (df['SPY_close'] < prev_close)).fillna(False)

    # 5) SEÑALES Y CONFIANZAS
    df['long_signal']  = df.apply(long_ready, axis=1)
    df['short_signal'] = df.apply(short_ready, axis=1)

    df['conf_long']  = df.apply(conf_long, axis=1)
    df['conf_short'] = df.apply(conf_short, axis=1)

    r_long, r_short, state = zip(*df.apply(make_llm_rationale, axis=1))
    df['rationale_long']  = list(r_long)
    df['rationale_short'] = list(r_short)
    df['explain_state']   = list(state)

    sig, conf = zip(*df.apply(pick_signal, axis=1))
    df['signal_now'] = list(sig)
    df['confidence_0_1'] = list(conf)

    # 6) SELECCIÓN DE COLUMNAS
    cols = [
        # OHLC
        'SPY_open','SPY_high','SPY_low','SPY_close',
        # EMAs 1m
        'ema9','ema20','ema40','ema9_above_ema20','dist_ema9_pct','dist_ema20_pct',
        # Bollinger 15m
        'bb_mid_15','bb_up_15','bb_lo_15','bb_w_15','bb_pos_15','bb_open_any_15','bb_open_up_15','bb_open_dn_15',
        # Bollinger 60m
        'bb_mid_60','bb_up_60','bb_lo_60','bb_w_60','bb_pos_60','bb_open_any_60','bb_open_up_60','bb_open_dn_60',
        # Tendencias/Toques
        'trend_up','trend_down','touch_ema9_up','touch_ema9_dn',
        # Señales y explicabilidad
        'long_signal','short_signal','conf_long','conf_short','signal_now','confidence_0_1',
        'rationale_long','rationale_short','explain_state'
    ]
    sheet = df[cols].dropna(subset=['ema9','bb_mid_15','bb_mid_60']).copy()

    # 7) Guardar el csv en la respectiva carpeta
    sheet.to_csv(os.path.join(inputs_dir,out_path), index=True)

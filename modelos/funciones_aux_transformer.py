"""
Módulo que contiene las funciones necesarias para orquestar la ejecución
del modelo transformer para identificar la acción a predecir del precio.                                    
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
import numpy as np
import pandas as pd

# Funciones auxiliares para el modelo transformer

# Función auxiliar para determinar el ATR
def atr_wilder_exact(tr: pd.Series, n: int = 14) -> pd.Series:
    """
    Calcula el Average True Range (ATR) siguiendo la metodología de Wilder.
    Implementación:
      - Se busca la primera posición donde hay exactamente `n` valores
        contiguos para calcular la semilla como la SMA de esos `n` valores.
      - A partir de la semilla se aplica la recursión de Wilder:
            ATR_t = ATR_{t-1} - ATR_{t-1}/n + TR_t/n
    La función es robusta ante índices no únicos y preserva el índice
    original de la serie de entrada.
    Args:
        tr (pd.Series): Serie del True Range (TR). Puede tener cualquier índice
            (incluso no único); se convertirá internamente a float64.
        n (int): Periodo para la SMA semilla y la recursión (por defecto 14).
    Returns:
        pd.Series: Serie de ATR con el mismo índice que `tr`. Las posiciones
        anteriores a la semilla contienen NaN. Si no hay suficientes datos
        para calcular la semilla se devuelve una serie de NaN del mismo tamaño.
    """
    tr = tr.astype("float64")
    idx = tr.index

    if not isinstance(n, int) or n < 1:
        raise ValueError("n debe ser un entero positivo mayor o igual a 1")

    # Posición del primer punto con SMA(n) disponible
    count = tr.rolling(n, min_periods=n).count().to_numpy()
    # primera posición donde hay n valores (== n)
    if not np.any(count == n):
        # no hay suficientes datos
        return pd.Series(np.nan, index=idx)

    seed_pos = int(np.argmax(count == n))            # posición entera
    seed_val = float(tr.iloc[seed_pos - n + 1 : seed_pos + 1].mean())

    # buffer de salida
    atr = np.full(tr.shape[0], np.nan, dtype="float64")
    atr[seed_pos] = seed_val

    # recursión Wilder
    for i in range(seed_pos + 1, tr.shape[0]):
        atr[i] = atr[i - 1] - (atr[i - 1] / n) + (tr.iloc[i] / n)

    return pd.Series(atr, index=idx)

# Función auxiliar para calcular los indicadores técnicos a la serie temporal
def agregar_indicadores(df: pd.DataFrame, prefix: str = "SPY") -> pd.DataFrame:
    """
    Calcula indicadores técnicos sin data leakage (todo en t usa datos <= t-1).

    Args:
        - df: DataFrame que contiene los datos básicos del activo al que se le
        estimarán los indicadores. Debe incluir las columnas date, open,
        high, low, close y volume.
        - prefix: Nombre que se le asignará al activo en cuestión.

    Incluye:
    - Precios básicos (open, high, low, close) y HL2 (t-1)
    - Retorno log/pct de 1m (t-1)
    - EMAs (5,24,50,200) y distancias %
    - RSI(14) con Wilder
    - MACD (12,26) + señal(9) + histograma (t-1)
    - ATR(14) con Wilder
    - Bandas de Bollinger (SMA20, STD20) → %B y Z-score
    - Estocástico %K/%D (t-1)
    - Volumen y Volumen Relativo(24)
    - Indicadores adicionales: ROC, Williams %R, CCI, OBV, MFI, %B de Keltner,
        Volatilidad histórica (12/24) y retorno acumulado 30m.
    """
    df = df.copy()

    C1 = df["SPY_close"].shift(1)
    H1 = df["SPY_high"].shift(1)
    L1 = df["SPY_low"].shift(1)
    O1 = df["SPY_open"].shift(1)
    V1 = df["SPY_volume"].shift(1)

    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]

    # Precios básicos
    out["SPY_open"] = O1
    out["SPY_high"] = H1
    out["SPY_low"]  = L1
    out["SPY_close"] = C1

    # Retorno 1m
    out[f"{prefix}_return_1m"] = df["SPY_close"].pct_change().shift(1)

    # HL2
    out[f"{prefix}_hl2"] = (H1 + L1) / 2

    # EMAs
    ema5   = C1.ewm(span=5,  adjust=False).mean()
    ema24  = C1.ewm(span=24, adjust=False).mean()
    ema50  = C1.ewm(span=50, adjust=False).mean()
    ema200 = C1.ewm(span=200, adjust=False).mean()
    out[f"{prefix}_EMA_5"]   = ema5
    out[f"{prefix}_EMA_24"]  = ema24
    out[f"{prefix}_EMA_50"]  = ema50
    out[f"{prefix}_EMA_200"] = ema200

    eps = 1e-8

    # Diferencias porcentuales en EMAs
    out[f"{prefix}_EMA5_diff_pct"]    = (C1 - ema5)   / (ema5 + eps)
    out[f"{prefix}_EMA24_diff_pct"]   = (C1 - ema24)  / (ema24 + eps)
    out[f"{prefix}_EMA50_diff_pct"]   = (C1 - ema50)  / (ema50 + eps)
    out[f"{prefix}_EMA200_diff_pct"]  = (C1 - ema200) / (ema200 + eps)

    # RSI(14) estilo Wilder
    delta = df["SPY_close"].diff().shift(1)
    gain  = delta.clip(lower=0.0)
    loss  = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + eps)
    out[f"{prefix}_RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema_fast = C1.ewm(span=12, adjust=False).mean()
    ema_slow = C1.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    out[f"{prefix}_MACD_12_26"]     = macd
    out[f"{prefix}_MACD_signal_9m"] = macd_sig
    out[f"{prefix}_MACD_hist"]      = macd - macd_sig

    # ATR(14) con Wilder (requiere atr_wilder_exact)
    prev_close = df["SPY_close"].shift(2)
    tr1 = H1 - L1
    tr2 = (H1 - prev_close).abs()
    tr3 = (L1 - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out[f"{prefix}_ATR_14"] = atr_wilder_exact(tr, n=14)

    # Bandas de Bollinger
    sma20 = C1.rolling(window=20, min_periods=20).mean()
    std20 = C1.rolling(window=20, min_periods=20).std()
    upper = sma20 + 2*std20
    lower = sma20 - 2*std20
    out[f"{prefix}_BB_percentB"] = (C1 - lower) / (upper - lower + eps)
    out[f"{prefix}_BB_z"] = (C1 - sma20) / (std20 + eps)

    # Estocástico %K/%D
    low14  = L1.rolling(window=14, min_periods=14).min()
    high14 = H1.rolling(window=14, min_periods=14).max()
    stoch_k = 100.0 * (C1 - low14) / (high14 - low14 + eps)
    stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
    out[f"{prefix}_stoch_k"] = stoch_k
    out[f"{prefix}_stoch_d"] = stoch_d

    # Volumen y volumen relativo
    out[f"{prefix}_volume"] = V1
    vol_mean_24 = V1.rolling(window=24, min_periods=24).mean()
    out[f"{prefix}_vol_rel_24"] = V1 / (vol_mean_24 + eps)

    # --- Indicadores adicionales ---

    # ROC de 12 y 20
    out[f"{prefix}_ROC_12"] = (C1 - C1.shift(12)) / (C1.shift(12) + eps)
    out[f"{prefix}_ROC_20"] = (C1 - C1.shift(20)) / (C1.shift(20) + eps)

    # Williams %R (14)
    highest14 = H1.rolling(window=14, min_periods=14).max()
    lowest14  = L1.rolling(window=14, min_periods=14).min()
    out[f"{prefix}_WilliamsR_14"] = -100.0 * (highest14 - C1) / (highest14 - lowest14 + eps)

    # CCI (20)
    tp = (H1 + L1 + C1) / 3
    ma_tp = tp.rolling(window=20, min_periods=20).mean()
    mad = (tp - ma_tp).abs().rolling(window=20, min_periods=20).mean()
    out[f"{prefix}_CCI_20"] = (tp - ma_tp) / (0.015 * mad + eps)

    # OBV
    direction = np.sign(C1 - C1.shift(1))
    obv_step = np.where(direction > 0, V1,
                        np.where(direction < 0, -V1, 0.0))
    obv = pd.Series(obv_step).cumsum()
    out[f"{prefix}_OBV"] = obv

    # MFI (14)
    tp_shift1 = tp
    tp_shift2 = tp.shift(1)
    mf = tp_shift1 * V1
    positive_mf = mf.where(tp_shift1 > tp_shift2, 0.0)
    negative_mf = mf.where(tp_shift1 < tp_shift2, 0.0)
    sum_pos_mf = positive_mf.rolling(window=14, min_periods=14).sum()
    sum_neg_mf = negative_mf.rolling(window=14, min_periods=14).sum()
    mfi_ratio = sum_pos_mf / (sum_neg_mf + eps)
    out[f"{prefix}_MFI_14"] = 100.0 - (100.0 / (1 + mfi_ratio))

    # %B de Keltner
    ema_tp20 = tp.ewm(span=20, adjust=False).mean()
    atr14 = out[f"{prefix}_ATR_14"]
    kel_upper = ema_tp20 + 2 * atr14
    kel_lower = ema_tp20 - 2 * atr14
    out[f"{prefix}_Keltner_percentB"] = (C1 - kel_lower) / (kel_upper - kel_lower + eps)

    # Volatilidad histórica
    ret = out[f"{prefix}_return_1m"]
    out[f"{prefix}_volatility_std12"] = ret.rolling(window=12, min_periods=12).std()
    out[f"{prefix}_volatility_std24"] = ret.rolling(window=24, min_periods=24).std()

    # Retorno acumulado 30m
    out[f"{prefix}_cumret_30m"] = (C1 / (C1.shift(30) + eps)) - 1.0

    return out

def sanitize_data(X, clip: float = 10.0) -> np.ndarray:
    """
    Normaliza y recorta valores numéricos en una entrada array-like.
    Esta función está pensada para preparar entradas antes de pasarlas a un
    modelo. Realiza dos pasos sencillos pero importantes:
      1. Reemplaza NaN por 0.0 y valores infinitos por el valor de `clip` con
         signo correspondiente.
      2. Recorta (clipping) todos los valores al rango [-clip, clip].
    Args:
        X (array-like): Entrada numérica (por ejemplo, np.ndarray, pd.Series o list)
            que puede contener NaN o infinitos.
        clip (float): Valor absoluto máximo permitido. Valores mayores en
            magnitud se recortarán a +/- `clip`. Por defecto 10.0.
    Returns:
        np.ndarray: Array numpy de tipo float con NaN/Inf reemplazados y con
        todos los valores limitados en el intervalo [-clip, clip].
    """
    arr = np.asarray(X, dtype=float)
    # Reemplaza NaN/Inf conservando el límite `clip`
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
    # Asegura que ningún valor supere la magnitud `clip`
    arr = np.clip(arr, -clip, clip)
    return arr

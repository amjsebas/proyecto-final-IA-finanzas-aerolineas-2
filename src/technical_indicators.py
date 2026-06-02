"""
src/technical_indicators.py

Indicadores tecnicos clasicos para el Fase 3 del proyecto.

Convencion: todas las funciones devuelven una serie de senales en [-1, +1] donde
+1 = maximo bullish, -1 = maximo bearish, 0 = neutral.

Entrada estandar: DataFrame con columnas ['Open','High','Low','Close','Volume']
indexado por fecha (DatetimeIndex), un ticker a la vez.

Las funciones no modifican el input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import ta  # libreria 'ta' para indicadores estandar
    HAS_TA = True
except ImportError:
    HAS_TA = False


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _clip(s: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
    return s.clip(lo, hi)


# ---------------------------------------------------------------------------
# MOMENTUM
# ---------------------------------------------------------------------------

def rsi_signal(close: pd.Series, period: int = 14,
               low: float = 30.0, high: float = 70.0) -> pd.Series:
    """
    RSI con umbrales 30/70.
    Devuelve +1 si sobreventa (RSI < low), -1 si sobrecompra (RSI > high), 0 si no.
    Bandera de continuidad: cuanto mas lejos del umbral, mas fuerte la senal.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # Senal continua: lineal entre los umbrales
    sig = pd.Series(0.0, index=close.index)
    sig = sig.where(rsi >= low,  (low  - rsi) / low)        # 0..1 por debajo del 30
    sig = sig.where(rsi <= high, -(rsi - high) / (100 - high))  # 0..-1 por encima del 70
    return _clip(sig)


def stochastic_signal(high: pd.Series, low: pd.Series, close: pd.Series,
                      k_period: int = 14, d_period: int = 3,
                      lo: float = 20.0, hi: float = 80.0) -> pd.Series:
    """
    Stochastic %K. +1 si %K < 20, -1 si %K > 80, lineal entre.
    """
    lowest  = low.rolling(k_period).min()
    highest = high.rolling(k_period).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(d_period).mean()

    sig = pd.Series(0.0, index=close.index)
    sig = sig.where(d >= lo, (lo - d) / lo)
    sig = sig.where(d <= hi, -(d - hi) / (100 - hi))
    return _clip(sig)


def roc_signal(close: pd.Series, period: int = 10, threshold_pct: float = 5.0) -> pd.Series:
    """
    Rate of Change a 'period' dias. +1 si ROC > threshold%, -1 si ROC < -threshold%.
    """
    roc = (close / close.shift(period) - 1) * 100  # %
    sig = roc / threshold_pct
    return _clip(sig)


# ---------------------------------------------------------------------------
# TENDENCIA
# ---------------------------------------------------------------------------

def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """
    MACD. +1 si MACD > linea de senal, -1 en otro caso.
    Magnitud = tanh((macd - signal_line) normalizado por su std rolling).
    """
    ema_fast = close.ewm(span=fast,  adjust=False).mean()
    ema_slow = close.ewm(span=slow,  adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    # Normalizar por la volatilidad rolling del histograma
    norm = hist.rolling(60).std().replace(0, np.nan)
    sig = np.tanh(hist / (norm + 1e-9))
    return _clip(sig)


def sma_crossover_signal(close: pd.Series,
                         fast: int = 20, mid: int = 50, slow: int = 200) -> pd.Series:
    """
    Cruce de medias moviles. +1 si SMA(fast) > SMA(mid), -1 si <.
    Modificador: si tambien SMA(mid) > SMA(slow), refuerza (tendencia larga alineada).
    """
    s_fast = close.rolling(fast).mean()
    s_mid  = close.rolling(mid).mean()
    s_slow = close.rolling(slow).mean()

    short_signal = np.sign(s_fast - s_mid)        # -1 o +1
    long_align   = np.sign(s_mid  - s_slow)       # alineacion de largo plazo

    sig = 0.6 * short_signal + 0.4 * long_align
    return _clip(sig)


def adx_strength(high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14, threshold: float = 25.0) -> pd.Series:
    """
    ADX como MODIFICADOR de fuerza de tendencia, no como direccion.
    Devuelve un valor en [0, 1] que mide cuan fuerte es la tendencia.
    Se usa como multiplicador en el signal combiner.
    """
    plus_dm  = (high.diff()).where(lambda x: x > -low.diff(), 0).clip(lower=0)
    minus_dm = (-low.diff()).where(lambda x: x > high.diff(), 0).clip(lower=0)

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di  = 100 * plus_dm.rolling(period).mean()  / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()

    # Score 0..1: 0 si ADX < threshold, 1 si ADX >= 50
    strength = ((adx - threshold) / (50 - threshold)).clip(0, 1).fillna(0)
    return strength


# ---------------------------------------------------------------------------
# VOLATILIDAD
# ---------------------------------------------------------------------------

def bollinger_signal(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    Bandas de Bollinger. +1 cerca de banda inferior, -1 cerca de banda superior.
    %B = (close - lower) / (upper - lower) en [0, 1] (puede salir un poco).
    Senal = 1 - 2*%B, es decir +1 en banda inferior, -1 en banda superior.
    """
    mid = close.rolling(period).mean()
    sd  = close.rolling(period).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    sig = 1 - 2 * pct_b
    return _clip(sig)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
    """
    ATR como porcentaje del precio. NO es una senal direccional - se usa como
    feature de regimen (volatilidad) en el modelo causal y en position sizing.
    """
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close


# ---------------------------------------------------------------------------
# VOLUMEN
# ---------------------------------------------------------------------------

def obv_signal(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    OBV (On-Balance Volume) vs su media movil. +1 si OBV por encima de la media,
    -1 si por debajo, magnitud proporcional a la desviacion.
    """
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    ma  = obv.rolling(window).mean()
    sd  = obv.rolling(window).std().replace(0, np.nan)
    z   = (obv - ma) / sd
    return _clip(np.tanh(z))


def volume_spike(volume: pd.Series, window: int = 20,
                 spike_threshold: float = 2.0) -> pd.Series:
    """
    Volume spike: volumen actual relativo a media de 'window' dias.
    Devuelve un MODIFICADOR (no direccional): 0 si volumen normal, ~1 si extremo.
    Para usar como peso o como flag de "algo esta pasando".
    """
    ratio = volume / volume.rolling(window).mean()
    spike = ((ratio - 1) / (spike_threshold - 1)).clip(0, 1).fillna(0)
    return spike


# ---------------------------------------------------------------------------
# AGREGACION POR CATEGORIA (para el signal_combiner)
# ---------------------------------------------------------------------------

def compute_all_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores para un ticker (DataFrame OHLCV).
    Devuelve un DataFrame con una columna por indicador, mismo indice.
    """
    o, h, l, c, v = (ohlcv['Open'], ohlcv['High'], ohlcv['Low'],
                     ohlcv['Close'], ohlcv['Volume'])
    out = pd.DataFrame(index=ohlcv.index)

    # momentum
    out['rsi']        = rsi_signal(c)
    out['stochastic'] = stochastic_signal(h, l, c)
    out['roc']        = roc_signal(c)

    # tendencia
    out['macd']         = macd_signal(c)
    out['sma_cross']    = sma_crossover_signal(c)
    out['adx_strength'] = adx_strength(h, l, c)  # modificador en [0,1]

    # volatilidad
    out['bollinger'] = bollinger_signal(c)
    out['atr_pct']   = atr_pct(h, l, c)          # feature de regimen

    # volumen
    out['obv']          = obv_signal(c, v)
    out['volume_spike'] = volume_spike(v)        # modificador en [0,1]

    return out


def aggregate_by_category(ind: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los indicadores en las 4 categorias del signal_combiner.
    Cada categoria es el promedio de sus indicadores direccionales (en [-1,+1]).
    """
    agg = pd.DataFrame(index=ind.index)
    agg['momentum_signal']   = ind[['rsi', 'stochastic', 'roc']].mean(axis=1)
    agg['trend_signal']      = ind[['macd', 'sma_cross']].mean(axis=1)
    agg['volatility_signal'] = ind[['bollinger']].mean(axis=1)
    agg['volume_signal']     = ind[['obv']].mean(axis=1)
    # Modificadores se preservan tal cual
    agg['adx_signal']        = ind['adx_strength']
    agg['volume_spike']      = ind['volume_spike']
    agg['atr_pct']           = ind['atr_pct']
    return agg


if __name__ == "__main__":
    # Smoke test rapido
    import yfinance as yf
    df = yf.download("AAL", start="2024-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    ind = compute_all_indicators(df)
    agg = aggregate_by_category(ind)
    print(agg.tail())
    print("\nRangos por columna (deben quedar mayormente en [-1, +1]):")
    print(agg.agg(['min', 'max', 'mean']).round(3))

"""
src/custom_signals.py

Senales propias del equipo. Tres senales originales con justificacion financiera
clara, distintas de los indicadores tecnicos clasicos.

Cada senal se normaliza a [-1, +1] siguiendo la convencion del proyecto.

Justificaciones (para incluir en el reporte):

1. MEAN_REVERSION_ZSCORE
   Hipotesis: el precio de una accion individual tiende a revertir hacia su media
   movil de 20 dias cuando se aleja > 2 sigmas. Sirve como contra-tendencia
   intra-mes, util para capturar overreaction a noticias o eventos macro.

2. MULTI_TIMEFRAME_MOMENTUM
   Hipotesis: el momentum genuino se confirma cuando las ventanas de 5, 20 y 60
   dias apuntan en la misma direccion. Combina rapidez (5d) con persistencia
   (60d), filtrando whipsaws y senales falsas de ventanas cortas.

3. RELATIVE_STRENGTH_VS_JETS
   Hipotesis: lo que importa en stock picking no es si AAL sube, sino si AAL
   le gana al ETF JETS. Aerolineas tienen alta correlacion intra-sector; la
   senal de outperformance relativa es mas informativa que el retorno absoluto
   para un long/short sector-neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Mean-reversion via z-score
# ---------------------------------------------------------------------------

def mean_reversion_zscore(close: pd.Series, window: int = 20,
                          cap_sigma: float = 3.0) -> pd.Series:
    """
    Z-score del precio vs su media movil de 'window' dias.

    z = (close - rolling_mean) / rolling_std
    senal = -tanh(z / cap_sigma)

    El signo negativo es por la hipotesis de reversion: cuando z > 0
    (precio caro vs su media), la senal es bajista; cuando z < 0
    (barato), alcista. tanh acota suavemente a [-1, +1].
    """
    mu = close.rolling(window).mean()
    sd = close.rolling(window).std().replace(0, np.nan)
    z  = (close - mu) / sd
    sig = -np.tanh(z / cap_sigma)
    return sig.clip(-1, 1)


# ---------------------------------------------------------------------------
# 2. Multi-timeframe momentum
# ---------------------------------------------------------------------------

def multi_timeframe_momentum(close: pd.Series,
                             windows=(5, 20, 60),
                             scale_pct: float = 5.0) -> pd.Series:
    """
    Composite de retornos a 5, 20 y 60 dias.

    Cada ventana se normaliza dividiendo entre 'scale_pct' (en %, por ejemplo
    5% = mueve la senal a +/- 1 cuando el retorno de esa ventana llega a +/-5%)
    y luego se hace tanh. El composite es el promedio de los 3, lo que penaliza
    senales contradictorias entre ventanas.
    """
    contribs = []
    for w in windows:
        ret = (close / close.shift(w) - 1) * 100  # %
        contribs.append(np.tanh(ret / scale_pct))
    return pd.concat(contribs, axis=1).mean(axis=1).clip(-1, 1)


# ---------------------------------------------------------------------------
# 3. Relative strength vs benchmark (JETS)
# ---------------------------------------------------------------------------

def relative_strength(close: pd.Series, benchmark_close: pd.Series,
                      window: int = 20, scale_pct: float = 3.0) -> pd.Series:
    """
    Relative strength = retorno del ticker - retorno del benchmark, en 'window' dias.

    Senal = tanh((stock_ret - bench_ret) / scale_pct).
    +1 si el ticker le gano al benchmark por scale_pct%, -1 si perdio por igual.
    """
    common = close.index.intersection(benchmark_close.index)
    c = close.reindex(common)
    b = benchmark_close.reindex(common)

    stock_ret = (c / c.shift(window) - 1) * 100
    bench_ret = (b / b.shift(window) - 1) * 100
    excess = stock_ret - bench_ret
    sig = np.tanh(excess / scale_pct)
    return sig.reindex(close.index).clip(-1, 1)


# ---------------------------------------------------------------------------
# Agregacion al combiner
# ---------------------------------------------------------------------------

def compute_custom_signals(close: pd.Series,
                           benchmark_close: pd.Series | None = None) -> pd.DataFrame:
    """
    Calcula las 3 senales propias para un ticker y devuelve un DataFrame.
    Si no se pasa el benchmark, la senal de relative_strength queda en 0.
    """
    out = pd.DataFrame(index=close.index)
    out['mr_zscore']    = mean_reversion_zscore(close)
    out['mtf_momentum'] = multi_timeframe_momentum(close)
    if benchmark_close is not None:
        out['rel_strength'] = relative_strength(close, benchmark_close)
    else:
        out['rel_strength'] = 0.0
    return out


def aggregate_custom_signal(custom_df: pd.DataFrame) -> pd.Series:
    """
    Combina las 3 senales propias en una sola serie. Promedio simple,
    pero con peso menor para mean-reversion (suele ser ruidosa intradiariamente).
    """
    weights = {'mr_zscore': 0.25, 'mtf_momentum': 0.45, 'rel_strength': 0.30}
    s = sum(custom_df[c] * w for c, w in weights.items() if c in custom_df.columns)
    return s.clip(-1, 1).rename('custom_signal')


if __name__ == "__main__":
    import yfinance as yf
    px = yf.download(["AAL", "JETS"], start="2024-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)["Close"]
    aal = px["AAL"]; jets = px["JETS"]
    cs = compute_custom_signals(aal, jets)
    print("Senales propias - ultimas filas:")
    print(cs.tail())
    print("\nCombinada:")
    print(aggregate_custom_signal(cs).tail())

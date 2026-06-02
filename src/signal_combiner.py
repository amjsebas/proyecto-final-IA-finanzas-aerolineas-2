"""
src/signal_combiner.py

Fase 5 del proyecto: combina todas las senales en una unica senal de trading.

Senales de entrada (todas en [-1, +1]):
- sentiment_signal  (Fase 4, output del sentiment hibrido)
- momentum_signal   (Fase 3, agregado de RSI/Stochastic/ROC)
- trend_signal      (Fase 3, agregado de MACD/SMA cross)
- volatility_signal (Fase 3, Bollinger)
- volume_signal     (Fase 3, OBV)
- custom_signal     (Fase 3, agregado de las 3 propias)

Modificadores:
- adx_strength en [0, 1]  -> refuerza la senal cuando hay tendencia clara
- volume_spike  en [0, 1] -> opcional, no usado en el combiner base

Salida:
- signal_raw         (combinacion ponderada, continuo en [-1, +1])
- signal_discrete    (BUY / HOLD / SELL con umbrales +-0.25)
- confidence         (|signal_raw|, para position sizing)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pesos base (Fase 5 de la rubrica)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "sentiment_signal":   0.15,
    "momentum_signal":    0.40,
    "trend_signal":       0.10,
    "volatility_signal":  0.10,
    "volume_signal":      0.13,
    "custom_signal":      0.07,
    # adx_strength se usa como modificador multiplicativo (5%), no se suma.
}


def validate_weights(weights: Dict[str, float], tol: float = 1e-6) -> None:
    """Verifica que la suma sea ~1.0 (sin contar el modificador ADX)."""
    s = sum(weights.values())
    if abs(s - 1.0) > tol:
        raise ValueError(f"Los pesos deben sumar 1.0; suman {s:.4f}")


# ---------------------------------------------------------------------------
# Combinacion principal
# ---------------------------------------------------------------------------

def combine_signals(df: pd.DataFrame,
                    weights: Dict[str, float] | None = None,
                    adx_col: str = "adx_signal",
                    adx_boost: float = 0.05,
                    buy_threshold: float = 0.25,
                    sell_threshold: float = -0.25) -> pd.DataFrame:
    """
    Combina las senales en una senal raw, aplica el modificador ADX y discretiza.

    Parametros:
        df: DataFrame con columnas de senales (las que correspondan en weights).
        weights: pesos por familia. Si None, usa DEFAULT_WEIGHTS.
        adx_col: columna con ADX strength en [0, 1] (opcional).
        adx_boost: cuanto refuerza la senal cuando ADX = 1 (0.05 = +5%).
        buy_threshold / sell_threshold: umbrales para BUY/SELL.

    Devuelve el df de entrada con columnas anadidas:
        signal_raw, signal_discrete, confidence.
    """
    weights = weights or DEFAULT_WEIGHTS
    validate_weights(weights)

    out = df.copy()

    # Asegurar que las columnas existen; si no, asumir 0
    for col in weights:
        if col not in out.columns:
            out[col] = 0.0

    # Combinacion lineal ponderada
    raw = np.zeros(len(out))
    for col, w in weights.items():
        raw += w * out[col].fillna(0).values

    # Modificador ADX: amplifica cuando hay tendencia fuerte
    if adx_col in out.columns:
        adx = out[adx_col].fillna(0).clip(0, 1).values
        raw = raw * (1 + adx_boost * adx)

    out["signal_raw"] = np.clip(raw, -1, 1)

    # Discretizar
    cond_buy  = out["signal_raw"] >  buy_threshold
    cond_sell = out["signal_raw"] <  sell_threshold
    out["signal_discrete"] = np.select(
        [cond_buy, cond_sell],
        ["BUY", "SELL"],
        default="HOLD",
    )

    # Confianza para position sizing
    out["confidence"] = out["signal_raw"].abs()

    return out


# ---------------------------------------------------------------------------
# Experimentos de pesos (rubrica pide al menos 3 configuraciones)
# ---------------------------------------------------------------------------

WEIGHT_CONFIGS: Dict[str, Dict[str, float]] = {
    # Base: la propuesta original de la rubrica
    "base": dict(DEFAULT_WEIGHTS),

    # Tilt momentum: mas peso al momentum, menos sentiment
    "momentum_heavy": {
        "sentiment_signal":   0.05,
        "momentum_signal":    0.55,
        "trend_signal":       0.15,
        "volatility_signal":  0.10,
        "volume_signal":      0.10,
        "custom_signal":      0.05,
    },

    # Tilt sentiment: el experimento del proyecto - subir sentiment a 30%
    "sentiment_heavy": {
        "sentiment_signal":   0.30,
        "momentum_signal":    0.30,
        "trend_signal":       0.10,
        "volatility_signal":  0.10,
        "volume_signal":      0.10,
        "custom_signal":      0.10,
    },

    # Equal weight: control / sanity check
    "equal": {
        "sentiment_signal":   1/6,
        "momentum_signal":    1/6,
        "trend_signal":       1/6,
        "volatility_signal":  1/6,
        "volume_signal":      1/6,
        "custom_signal":      1/6,
    },
}


def run_weight_experiments(df: pd.DataFrame,
                           configs: Dict[str, Dict[str, float]] | None = None,
                           **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Corre combine_signals con varias configuraciones de peso.
    Devuelve dict {nombre_config: DataFrame con signal_raw y signal_discrete}.
    """
    configs = configs or WEIGHT_CONFIGS
    out = {}
    for name, w in configs.items():
        out[name] = combine_signals(df, weights=w, **kwargs)
    return out


# ---------------------------------------------------------------------------
# Resumen de senales
# ---------------------------------------------------------------------------

def summarize_signals(df: pd.DataFrame, signal_col: str = "signal_discrete") -> pd.Series:
    """
    Conteo y % por categoria BUY/HOLD/SELL. Para incluir en el reporte.
    """
    counts = df[signal_col].value_counts()
    pct = (counts / len(df) * 100).round(1)
    return pd.concat([counts.rename("n"), pct.rename("pct")], axis=1)


if __name__ == "__main__":
    # Smoke test con datos simulados
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "sentiment_signal":  np.random.uniform(-1, 1, n),
        "momentum_signal":   np.random.uniform(-1, 1, n),
        "trend_signal":      np.random.uniform(-1, 1, n),
        "volatility_signal": np.random.uniform(-1, 1, n),
        "volume_signal":     np.random.uniform(-1, 1, n),
        "custom_signal":     np.random.uniform(-1, 1, n),
        "adx_signal":        np.random.uniform(0, 1, n),
    })
    out = combine_signals(df)
    print("Distribucion de senales discretas:")
    print(summarize_signals(out))
    print("\nExperimentos de pesos:")
    experiments = run_weight_experiments(df)
    for name, e in experiments.items():
        sells = (e["signal_discrete"] == "SELL").mean()
        buys  = (e["signal_discrete"] == "BUY").mean()
        print(f"  {name:18s}  BUY {buys:.1%}  SELL {sells:.1%}")

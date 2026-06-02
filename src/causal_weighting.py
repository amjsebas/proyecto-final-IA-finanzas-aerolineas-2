"""
src/causal_weighting.py

Convierte los edges causales estimados (Fase 5.5) en pesos dinamicos para el
signal_combiner. La logica preserva la estructura original de pesos fijos pero
los modula segun el regimen de mercado.

Formula central:
    w_tilde_j(t) = w_j_base * [1 + gamma * tanh(edge_j(t) / scale)]
    w_j_causal(t) = w_tilde_j(t) / sum_k |w_tilde_k(t)|

Asi:
- Si una senal muestra efecto positivo en el regimen actual -> su peso sube.
- Si muestra efecto debil o adverso -> su peso baja (pero NO se invierte
  automaticamente — solo se atenua, con un piso de 0).
- Los pesos siempre se renormalizan para sumar 1 en valor absoluto.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calculo de pesos dinamicos
# ---------------------------------------------------------------------------

def compute_causal_weights(df: pd.DataFrame,
                           signal_cols: List[str],
                           base_weights: Dict[str, float],
                           gamma: float = 1.0,
                           edge_scale: float | None = None,
                           eps: float = 1e-8) -> pd.DataFrame:
    """
    Construye las columnas causal_weight_<signal> a partir de las columnas
    causal_edge_<signal> producidas por CausalEffectsEstimator.

    base_weights : pesos originales de la Fase 5 (deben sumar ~1.0).
    gamma        : intensidad del ajuste causal. 0 = sin ajuste, 1 = ajuste pleno.
    edge_scale   : escala para normalizar los efectos antes del tanh.
                   Si None, se usa la desviacion mediana absoluta (MAD robusta).
    """
    out = df.copy()
    edge_cols = [f"causal_edge_{s}" for s in signal_cols]

    # Escala robusta global (sobre todas las senales) si no se da
    if edge_scale is None:
        all_edges = out[edge_cols].values.flatten()
        med = np.nanmedian(all_edges)
        edge_scale = max(np.nanmedian(np.abs(all_edges - med)), eps)

    raw_cols = []
    for signal in signal_cols:
        edge_col = f"causal_edge_{signal}"
        raw_col  = f"raw_causal_weight_{signal}"
        base_w   = base_weights.get(signal, 0.0)

        edge = out[edge_col].fillna(0).values
        multiplier = 1.0 + gamma * np.tanh(edge / (edge_scale + eps))
        # Piso en 0 (atenuar nunca implica voltear la senal); techo en 2x.
        multiplier = np.clip(multiplier, 0.0, 2.0)

        out[raw_col] = base_w * multiplier
        raw_cols.append(raw_col)

    # Renormalizacion fila a fila para que los pesos sumen 1 en valor absoluto.
    denom = out[raw_cols].abs().sum(axis=1).replace(0, np.nan)
    for signal in signal_cols:
        raw_col    = f"raw_causal_weight_{signal}"
        weight_col = f"causal_weight_{signal}"
        out[weight_col] = (out[raw_col] / denom).fillna(0)

    return out


# ---------------------------------------------------------------------------
# Combinacion con pesos dinamicos
# ---------------------------------------------------------------------------

def combine_signals_with_causal_weights(df: pd.DataFrame,
                                        signal_cols: List[str],
                                        buy_threshold: float = 0.25,
                                        sell_threshold: float = -0.25) -> pd.DataFrame:
    """
    Aplica la combinacion con pesos causales y produce signal_raw_causal,
    signal_discrete_causal y confidence_causal.
    """
    out = df.copy()
    raw = np.zeros(len(out))

    for signal in signal_cols:
        sig_v   = out[signal].fillna(0).values
        wt_v    = out[f"causal_weight_{signal}"].fillna(0).values
        raw    += sig_v * wt_v

    out["signal_raw_causal"]      = np.clip(raw, -1, 1)
    out["signal_discrete_causal"] = np.select(
        [out["signal_raw_causal"] >  buy_threshold,
         out["signal_raw_causal"] <  sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )
    out["confidence_causal"] = out["signal_raw_causal"].abs()
    return out


# ---------------------------------------------------------------------------
# Resumen para reporte
# ---------------------------------------------------------------------------

def weight_summary_by_regime(df: pd.DataFrame, signal_cols: List[str],
                             regime_col: str = "adx",
                             regime_threshold: float = 25.0) -> pd.DataFrame:
    """
    Promedio del peso causal por senal, segmentado por regimen de mercado
    (ej. tendencia fuerte vs lateral via ADX). Sirve para la tabla de la
    rubrica: "Best regime" por senal.
    """
    if regime_col not in df.columns:
        return pd.DataFrame()
    strong = df[df[regime_col] >= regime_threshold]
    weak   = df[df[regime_col] <  regime_threshold]

    rows = []
    for signal in signal_cols:
        col = f"causal_weight_{signal}"
        if col not in df.columns:
            continue
        rows.append({
            "signal":         signal,
            "mean_weight":    df[col].mean(),
            "weight_strong":  strong[col].mean() if len(strong) else np.nan,
            "weight_weak":    weak[col].mean()   if len(weak)   else np.nan,
            "weight_delta":   (strong[col].mean() - weak[col].mean())
                              if (len(strong) and len(weak)) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("weight_delta", ascending=False)


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    n = 200
    signals = ["sentiment_signal", "momentum_signal", "trend_signal"]
    df = pd.DataFrame({s: np.random.uniform(-1, 1, n) for s in signals})
    for s in signals:
        df[f"causal_edge_{s}"] = np.random.normal(0, 0.005, n)
    df["adx"] = np.random.uniform(0, 50, n)

    base = {"sentiment_signal": 0.15, "momentum_signal": 0.40, "trend_signal": 0.10}
    # Renormalizar a 1 para el smoke test
    s = sum(base.values()); base = {k: v/s for k, v in base.items()}

    df2 = compute_causal_weights(df, signals, base, gamma=1.0)
    df3 = combine_signals_with_causal_weights(df2, signals)
    print("Distribucion causal:")
    print(df3["signal_discrete_causal"].value_counts())
    print()
    print("Pesos medios por senal:")
    for s in signals:
        print(f"  {s}: base={base[s]:.3f}  medio_causal={df3[f'causal_weight_{s}'].mean():.3f}")
    print()
    print("Por regimen (ADX):")
    print(weight_summary_by_regime(df3, signals).round(4))

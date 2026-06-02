"""
src/evaluation.py

Fases 7 y 8 del proyecto:
  - Fase 7: Metricas de clasificacion (confusion matrix, ROC-AUC, KS).
  - Fase 8: Framework de riesgo (calibracion, position sizing, stop-loss).

Asume que recibe un DataFrame con columnas:
    'signal_discrete'   en {BUY, HOLD, SELL}
    'signal_raw'        continuo en [-1, +1] (usado como score para ROC-AUC y KS)
    'confidence'        en [0, 1]
    'next_return'       retorno del siguiente periodo
    'volatility_20d'    para position sizing
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)


# ---------------------------------------------------------------------------
# Ground truth: convertir retorno futuro en label BUY/HOLD/SELL
# ---------------------------------------------------------------------------

def make_ground_truth(returns: pd.Series, threshold_pct: float = 1.0) -> pd.Series:
    """
    Convierte next_return en label BUY/HOLD/SELL usando un umbral.
    threshold_pct en %, asi 1.0 = +/-1%.
    """
    thr = threshold_pct / 100
    return pd.Series(np.where(returns >  thr, "BUY",
                     np.where(returns < -thr, "SELL", "HOLD")),
                     index=returns.index, name="ground_truth")


# ---------------------------------------------------------------------------
# Confusion matrix y metricas multi-clase
# ---------------------------------------------------------------------------

LABELS = ["BUY", "HOLD", "SELL"]


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series,
                          out_path: str | None = None,
                          title: str = "Confusion Matrix (pred vs real)"):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS,
                cbar=False, ax=ax)
    # Anade % entre parentesis debajo
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j + 0.5, i + 0.75, f"({cm_pct[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=8, color="grey")
    ax.set_xlabel("Prediccion (signal_discrete)")
    ax.set_ylabel("Real (siguiente periodo)")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  -> guardado: {out_path}")
    plt.show()


def classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Tabla de metricas multi-clase para reporte:
      Accuracy, Precision/Recall/F1 por clase, F1-Macro.
    """
    return {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Precision (BUY)":   precision_score(y_true, y_pred, labels=["BUY"], average="macro", zero_division=0),
        "Recall    (BUY)":   recall_score(y_true,   y_pred, labels=["BUY"], average="macro", zero_division=0),
        "Precision (SELL)":  precision_score(y_true, y_pred, labels=["SELL"], average="macro", zero_division=0),
        "Recall    (SELL)":  recall_score(y_true,   y_pred, labels=["SELL"], average="macro", zero_division=0),
        "F1 (BUY)":          f1_score(y_true, y_pred, labels=["BUY"], average="macro", zero_division=0),
        "F1 (SELL)":         f1_score(y_true, y_pred, labels=["SELL"], average="macro", zero_division=0),
        "F1 Macro":          f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# ROC-AUC y KS (problema binario)
# ---------------------------------------------------------------------------

def binarize_movement(returns: pd.Series, threshold_pct: float = 0.0) -> pd.Series:
    """1 si el retorno > threshold, 0 si no. Para ROC y KS."""
    thr = threshold_pct / 100
    return (returns > thr).astype(int)


def plot_roc(y_binary: pd.Series, score: pd.Series, out_path: str | None = None,
             title: str = "ROC Curve - prediccion de movimiento positivo"):
    fpr, tpr, _ = roc_curve(y_binary, score)
    auc = roc_auc_score(y_binary, score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  -> guardado: {out_path}")
    plt.show()
    return auc


def ks_statistic(y_binary: pd.Series, score: pd.Series) -> float:
    """
    KS = max separacion vertical entre las distribuciones CDF de scores
    para clase positiva vs negativa. > 0.2 = aceptable, > 0.4 = bueno.
    """
    pos = score[y_binary == 1].sort_values().values
    neg = score[y_binary == 0].sort_values().values
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    all_scores = np.sort(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(pos, all_scores, side="right") / len(pos)
    cdf_neg = np.searchsorted(neg, all_scores, side="right") / len(neg)
    return float(np.abs(cdf_pos - cdf_neg).max())


# ---------------------------------------------------------------------------
# Calibracion (Fase 8): reliability diagram
# ---------------------------------------------------------------------------

def calibration_data(confidence: pd.Series, hit: pd.Series,
                     n_bins: int = 10) -> pd.DataFrame:
    """
    hit: 1 si la prediccion fue correcta, 0 si no. Devuelve un DataFrame con
    confianza media y accuracy real por bin.
    """
    df = pd.DataFrame({"conf": confidence.values, "hit": hit.values}).dropna()
    df["bin"] = pd.cut(df["conf"], bins=np.linspace(0, 1, n_bins + 1),
                       include_lowest=True)
    agg = df.groupby("bin", observed=True).agg(
        bin_mean_conf=("conf", "mean"),
        actual_accuracy=("hit", "mean"),
        n=("hit", "size"),
    ).reset_index(drop=True)
    return agg


def plot_calibration(confidence: pd.Series, hit: pd.Series,
                     n_bins: int = 10, out_path: str | None = None):
    cal = calibration_data(confidence, hit, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Perfectamente calibrado")
    ax.plot(cal["bin_mean_conf"], cal["actual_accuracy"],
            "o-", color="#1f77b4", lw=2, label="Modelo")
    # Tamano del bin como tamano del punto
    sizes = (cal["n"] / cal["n"].max() * 200).clip(20, 200)
    ax.scatter(cal["bin_mean_conf"], cal["actual_accuracy"], s=sizes,
               color="#1f77b4", alpha=0.4)
    ax.set_xlabel("Confianza promedio del bin")
    ax.set_ylabel("Accuracy real")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram (calibracion del modelo)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  -> guardado: {out_path}")
    plt.show()
    return cal


# ---------------------------------------------------------------------------
# Risk framework (Fase 8): position sizing, stops, confidence filter
# ---------------------------------------------------------------------------

def position_size(confidence: pd.Series, volatility: pd.Series,
                  max_position_pct: float = 0.20,
                  target_vol: float = 0.20) -> pd.Series:
    """
    Tamano de posicion = max_position * factor_confianza * factor_volatilidad.

    factor_confianza   = confidence (en [0, 1])
    factor_volatilidad = min(1, target_vol / volatility_anualizada)
                          (reduce posicion en activos con vol > target)

    volatility: serie de vol_20d anualizada (igual que la columna del pipeline).
    """
    conf_f = confidence.fillna(0).clip(0, 1)
    vol_f  = (target_vol / volatility.replace(0, np.nan)).clip(upper=1).fillna(0)
    return (max_position_pct * conf_f * vol_f).clip(0, max_position_pct)


def confidence_filter(df: pd.DataFrame, threshold: float = 0.60,
                      signal_col: str = "signal_discrete") -> pd.DataFrame:
    """
    Marca como HOLD cualquier senal con confidence < threshold.
    Devuelve copia con columna 'signal_filtered'.
    """
    out = df.copy()
    out["signal_filtered"] = out[signal_col].where(
        out["confidence"] >= threshold, "HOLD"
    )
    return out


def stop_loss_rules(df: pd.DataFrame,
                    sentiment_flip_threshold: float = 1.0,
                    confidence_drop_threshold: float = 0.40) -> pd.DataFrame:
    """
    Reglas de stop-loss basadas en el modelo (no en precio):

    1. SENTIMENT FLIP: si el sentiment cambia de signo abruptamente
       (|delta_sentiment| >= sentiment_flip_threshold), cerrar posicion.
    2. CONFIDENCE DROP: si la confianza cae por debajo del threshold, reducir
       exposicion (marcar como HOLD).

    Anade columnas: 'stop_flip', 'stop_low_conf', 'signal_with_stops'.
    """
    out = df.copy().sort_values(["ticker", "date"])
    if "sentiment_score" in out.columns:
        out["sent_delta"] = (out
                             .groupby("ticker")["sentiment_score"]
                             .diff().abs().fillna(0))
        out["stop_flip"]  = (out["sent_delta"] >= sentiment_flip_threshold).astype(int)
    else:
        out["stop_flip"]  = 0
    out["stop_low_conf"]  = (out["confidence"] < confidence_drop_threshold).astype(int)

    out["signal_with_stops"] = np.where(
        (out["stop_flip"] == 1) | (out["stop_low_conf"] == 1),
        "HOLD",
        out["signal_discrete"],
    )
    return out


# ---------------------------------------------------------------------------
# Comparacion de modelos (Fase 5.5: base vs grid vs causal)
# ---------------------------------------------------------------------------

def compare_models(model_results: Dict[str, pd.DataFrame],
                   benchmark_bt: pd.DataFrame,
                   initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Tabla comparativa Sharpe / Sortino / MaxDD / Alpha / WinRate / ProfitFactor
    para varios modelos. La rubrica pide reportar A (base), B (grid), C (causal).

    model_results: dict {nombre_modelo: backtest_df con columnas
                          'date', 'daily_return', 'equity'}.
    """
    from backtesting import (  # import diferido para evitar circular
        total_return, sharpe_ratio, sortino_ratio, max_drawdown,
        win_rate, profit_factor,
    )
    rows = []
    bench_ret = total_return(benchmark_bt["equity"], initial_capital)
    for name, bt in model_results.items():
        rows.append({
            "Modelo":         name,
            "Total Return":   total_return(bt["equity"], initial_capital),
            "Alpha":          total_return(bt["equity"], initial_capital) - bench_ret,
            "Sharpe":         sharpe_ratio(bt["daily_return"]),
            "Sortino":        sortino_ratio(bt["daily_return"]),
            "Max Drawdown":   max_drawdown(bt["equity"]),
            "Win Rate":       win_rate(bt["daily_return"]),
            "Profit Factor":  profit_factor(bt["daily_return"]),
        })
    return pd.DataFrame(rows).set_index("Modelo")


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    n = 500
    df = pd.DataFrame({
        "signal_discrete": np.random.choice(["BUY", "HOLD", "SELL"], n,
                                            p=[0.3, 0.4, 0.3]),
        "signal_raw":      np.random.uniform(-1, 1, n),
        "confidence":      np.random.uniform(0, 1, n),
        "next_return":     np.random.normal(0.001, 0.02, n),
        "volatility_20d":  np.abs(np.random.normal(0.25, 0.05, n)),
    })
    gt = make_ground_truth(df["next_return"], threshold_pct=1.0)
    print("Metricas clasificacion:")
    for k, v in classification_metrics(gt, df["signal_discrete"]).items():
        print(f"  {k:20s}: {v:.3f}")
    y_bin = binarize_movement(df["next_return"])
    print(f"\nKS: {ks_statistic(y_bin, df['signal_raw']):.3f}")

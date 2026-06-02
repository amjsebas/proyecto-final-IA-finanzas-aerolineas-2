"""
src/backtesting.py

Fase 6 del proyecto: backtesting vectorizado y rolling causal backtest.

Asume que recibe un DataFrame con columnas:
    ['date', 'ticker', 'signal_discrete', 'next_return']
donde:
    signal_discrete ∈ {BUY, HOLD, SELL}
    next_return     = retorno del siguiente periodo (NO el del periodo actual,
                      para evitar look-ahead)

Convencion de posiciones:
    BUY  -> +1 (long)
    HOLD ->  0 (cash)
    SELL -> -1 (short)

Limitaciones explicitas (a documentar en el reporte):
- Sin costos de transaccion ni slippage
- Sin impacto de mercado
- Asume liquidez perfecta a precio de cierre
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Backtest vectorizado
# ---------------------------------------------------------------------------

SIGNAL_TO_POSITION = {"BUY": 1, "HOLD": 0, "SELL": -1}


def discrete_to_position(s: pd.Series) -> pd.Series:
    return s.map(SIGNAL_TO_POSITION).fillna(0).astype(int)


def backtest_strategy(df: pd.DataFrame,
                      signal_col: str = "signal_discrete",
                      return_col: str = "next_return",
                      initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Calcula la curva de equity de la estrategia.

    Devuelve un DataFrame con columnas:
        date, position, daily_return, equity, cum_return.

    Si hay varios tickers, el portfolio asume equal-weight entre tickers con
    senal activa cada dia.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["position"] = discrete_to_position(out[signal_col])

    # Si hay varios tickers, agregamos a nivel fecha: promedio de pos*ret
    # (equivalente a equal-weight de los tickers con senal activa)
    by_date = (out
               .assign(pnl=lambda d: d["position"] * d[return_col])
               .groupby("date")
               .agg(active_n=("position", lambda x: (x != 0).sum()),
                    daily_return=("pnl", "mean")))
    by_date["daily_return"] = by_date["daily_return"].fillna(0)

    by_date["equity"]     = initial_capital * (1 + by_date["daily_return"]).cumprod()
    by_date["cum_return"] = by_date["equity"] / initial_capital - 1
    return by_date.reset_index()


def backtest_buy_and_hold(prices: pd.Series,
                          initial_capital: float = 100_000) -> pd.DataFrame:
    """
    Curva de equity de buy-and-hold del benchmark (e.g. JETS).
    Espera una serie de precios indexada por fecha.
    """
    px = prices.dropna().sort_index()
    daily = px.pct_change().fillna(0)
    equity = initial_capital * (1 + daily).cumprod()
    return pd.DataFrame({
        "date":         px.index,
        "daily_return": daily.values,
        "equity":       equity.values,
        "cum_return":   (equity / initial_capital - 1).values,
    })


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def annualization_factor(periods_per_year: int = 252) -> float:
    return np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int = 252) -> float:
    r = returns.dropna() - risk_free / periods_per_year
    if r.std() == 0 or len(r) < 2:
        return np.nan
    return r.mean() / r.std() * annualization_factor(periods_per_year)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods_per_year: int = 252) -> float:
    r = returns.dropna() - risk_free / periods_per_year
    downside = r[r < 0]
    if len(downside) < 2 or downside.std() == 0:
        return np.nan
    return r.mean() / downside.std() * annualization_factor(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    e = equity.dropna()
    if len(e) < 2:
        return np.nan
    peaks = e.cummax()
    dd = (e - peaks) / peaks
    return dd.min()


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    r = r[r != 0]
    if len(r) == 0:
        return np.nan
    return (r > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    r = returns.dropna()
    gains  = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def total_return(equity: pd.Series, initial_capital: float) -> float:
    if len(equity.dropna()) == 0:
        return np.nan
    return float(equity.dropna().iloc[-1] / initial_capital - 1)


def n_trades(df_with_position: pd.DataFrame) -> int:
    """Cambios de posicion. Cada vez que la posicion cambia, hay un trade."""
    pos = df_with_position.get("position", pd.Series(dtype=int))
    return int((pos.diff().fillna(0) != 0).sum())


def turnover(df_with_position: pd.DataFrame) -> float:
    """Turnover = sum|delta_position| / N. En [0, 2] aprox."""
    pos = df_with_position.get("position", pd.Series(dtype=int))
    if len(pos) < 2:
        return np.nan
    return float(pos.diff().abs().mean())


def metrics_summary(strategy_bt: pd.DataFrame, benchmark_bt: pd.DataFrame,
                    initial_capital: float = 100_000) -> Dict[str, float]:
    """Tabla de metricas para reporte. Devuelve dict {nombre: valor}."""
    s = strategy_bt; b = benchmark_bt
    out = {
        "Total Return":     total_return(s["equity"], initial_capital),
        "Benchmark Return": total_return(b["equity"], initial_capital),
        "Alpha":            total_return(s["equity"], initial_capital)
                          - total_return(b["equity"], initial_capital),
        "Sharpe":           sharpe_ratio(s["daily_return"]),
        "Sortino":          sortino_ratio(s["daily_return"]),
        "Max Drawdown":     max_drawdown(s["equity"]),
        "Benchmark MaxDD":  max_drawdown(b["equity"]),
        "Win Rate":         win_rate(s["daily_return"]),
        "Profit Factor":    profit_factor(s["daily_return"]),
    }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curve(strategy_bt: pd.DataFrame, benchmark_bt: pd.DataFrame,
                      title: str = "Equity Curve",
                      out_path: str | None = None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(strategy_bt["date"], strategy_bt["equity"],
                 label="Estrategia", color="#1f77b4", lw=1.6)
    axes[0].plot(benchmark_bt["date"], benchmark_bt["equity"],
                 label="Buy & Hold (JETS)", color="#aaaaaa", lw=1.4, ls="--")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].set_title(title)
    axes[0].legend(loc="upper left"); axes[0].grid(alpha=0.3)

    # Drawdown
    peaks = strategy_bt["equity"].cummax()
    dd    = (strategy_bt["equity"] - peaks) / peaks * 100
    axes[1].fill_between(strategy_bt["date"], dd, 0, color="#c62828", alpha=0.4)
    axes[1].set_ylabel("Drawdown (%)"); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel("Fecha")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  -> guardado: {out_path}")
    plt.show()


def plot_period_returns(strategy_bt: pd.DataFrame, freq: str = "QE",
                        out_path: str | None = None):
    """Retornos por trimestre o mes. freq: 'QE' (trimestre) o 'ME' (mes)."""
    s = strategy_bt.copy()
    s["date"] = pd.to_datetime(s["date"])
    periodic = s.set_index("date")["daily_return"].resample(freq).apply(
        lambda x: (1 + x).prod() - 1
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2e7d32" if v > 0 else "#c62828" for v in periodic.values]
    ax.bar(periodic.index.astype(str), periodic.values * 100, color=colors)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Retorno por periodo (%)")
    ax.set_title(f"Retornos por {'trimestre' if freq=='QE' else 'mes'}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Rolling causal backtest (anti-leakage)
# ---------------------------------------------------------------------------

def rolling_causal_backtest(df: pd.DataFrame,
                            causal_model_factory,
                            weighting_fn,
                            combine_fn,
                            signal_cols: List[str],
                            base_weights: Dict[str, float],
                            train_window: int = 126,
                            test_window: int = 21) -> pd.DataFrame:
    """
    Entrena el modelo causal en una ventana de training y predice senales en
    la ventana de test inmediatamente siguiente. Avanza hasta agotar las fechas.

    causal_model_factory : funcion sin argumentos que devuelve un nuevo
                           CausalEffectsEstimator listo para .fit().
    weighting_fn         : funcion (df, signal_cols, base_weights) -> df con
                           causal_weight_* columns. Tipicamente
                           causal_weighting.compute_causal_weights.
    combine_fn           : funcion (df, signal_cols) -> df con
                           signal_discrete_causal. Tipicamente
                           causal_weighting.combine_signals_with_causal_weights.
    """
    dates = sorted(df["date"].unique())
    results = []
    start = train_window
    while start + test_window < len(dates):
        train_dates = dates[start - train_window:start]
        test_dates  = dates[start:start + test_window]

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df  = df[df["date"].isin(test_dates)].copy()

        try:
            model = causal_model_factory()
            model.fit(train_df)
            test_df = model.predict_effects(test_df)
            test_df = weighting_fn(test_df, signal_cols, base_weights)
            test_df = combine_fn(test_df, signal_cols)
            results.append(test_df)
        except Exception as e:
            print(f"  WARN rolling window {dates[start].date()}: {e}")

        start += test_window

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # Smoke test
    np.random.seed(0)
    dates = pd.bdate_range("2024-01-01", "2024-12-31")
    n = len(dates)
    df = pd.DataFrame({
        "date":             list(dates) * 2,
        "ticker":           ["AAA"] * n + ["BBB"] * n,
        "signal_discrete":  np.random.choice(["BUY", "HOLD", "SELL"], size=2 * n,
                                             p=[0.3, 0.4, 0.3]),
        "next_return":      np.random.normal(0.0005, 0.02, 2 * n),
    })
    bt = backtest_strategy(df)
    bh_px = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n)),
                      index=dates)
    bh = backtest_buy_and_hold(bh_px)
    print("Metricas:")
    for k, v in metrics_summary(bt, bh).items():
        print(f"  {k:20s}: {v:+.4f}" if isinstance(v, float) else f"  {k}: {v}")

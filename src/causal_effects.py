"""
src/causal_effects.py

Fase 5.5 del proyecto: estimacion de Heterogeneous Treatment Effects (HTE)
para cada familia de senales, usando CausalForestDML de econml.

Pregunta causal:
    tau_j(x) = E[Y(1) - Y(0) | X = x]

donde:
    Y = retorno futuro (por default next-day, opcionalmente normalizado por ATR%)
    T_j = activacion de la senal j (long o short)
    X = contexto de mercado (ATR%, ADX, RSI, volume z-score, etc.)

El template proviene de la rubrica del curso; aqui se adapta para:
- Manejo de muestras pequenas (min_samples default reducido a 50, con warning).
- Separacion explicita de tratamientos long vs short por senal.
- Edge causal aditivo: long * tau_long - short * tau_short.
- Documentacion de los supuestos identificadores.

Nota metodologica: esto es causal inference OBSERVACIONAL. Se asume que los
confounders relevantes estan observados en X. No es prueba causal definitiva.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CausalConfig:
    """Configuracion del estimador. Defaults pensados para una muestra moderada."""
    outcome_col: str = "next_return"
    min_samples: int = 50            # rubrica sugiere 250 - reducido para muestra real
    random_state: int = 42
    n_estimators: int = 400
    min_samples_leaf: int = 5         # rubrica sugiere 20 - reducido para muestra real
    cv: int = 3


# ---------------------------------------------------------------------------
# Estimador
# ---------------------------------------------------------------------------

class CausalEffectsEstimator:
    """
    Para cada senal en signal_cols se entrenan DOS modelos causales:
      - long  : tratamiento = max(signal, 0)   (activacion bullish)
      - short : tratamiento = max(-signal, 0)  (activacion bearish)

    Despues, predict_effects devuelve tau_long, tau_short y el edge causal
    consolidado por senal para cada fila.

    Outcome recomendado:
        next_return  o  next_return / atr_pct  (recomendado para comparar tickers).
    """

    def __init__(self,
                 signal_cols: List[str],
                 context_cols: List[str],
                 config: CausalConfig | None = None):
        self.signal_cols  = signal_cols
        self.context_cols = context_cols
        self.config       = config or CausalConfig()
        self.models_: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [self.config.outcome_col] + self.signal_cols + self.context_cols
        out = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        return out

    def _fit_single(self, df: pd.DataFrame, treatment_col: str, name: str):
        Y = df[self.config.outcome_col].values
        T = df[treatment_col].values
        X = df[self.context_cols].values

        if len(df) < self.config.min_samples:
            warnings.warn(
                f"{name}: {len(df)} muestras < min_samples ({self.config.min_samples}). "
                f"Las estimaciones tendran mucha varianza."
            )

        if not ECONML_AVAILABLE:
            raise ImportError("econml no esta instalado. Instala con: pip install econml")

        model_y = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model_t = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state + 1,
            n_jobs=-1,
        )

        est = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=self.config.n_estimators,
            min_samples_leaf=self.config.min_samples_leaf,
            discrete_treatment=False,
            cv=self.config.cv,
            random_state=self.config.random_state,
        )
        est.fit(Y=Y, T=T, X=X)
        self.models_[name] = est
        return est

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        df = self._clean_frame(df)
        for signal in self.signal_cols:
            long_col  = f"{signal}_treat_long"
            short_col = f"{signal}_treat_short"
            df[long_col]  = df[signal].clip(lower=0)
            df[short_col] = (-df[signal]).clip(lower=0)
            self._fit_single(df, long_col,  name=f"{signal}_long")
            self._fit_single(df, short_col, name=f"{signal}_short")
        return self

    def predict_effects(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Anade al df las columnas:
            tau_<signal>_long, tau_<signal>_short, causal_edge_<signal>.

        El edge causal consolidado es:
            edge_j = T_long * tau_long  -  T_short * tau_short

        Es positivo cuando la activacion va en el sentido del efecto esperado
        (long con tau_long > 0, o short con tau_short < 0).
        """
        out = df.copy()
        X = (out[self.context_cols]
             .replace([np.inf, -np.inf], np.nan)
             .fillna(0).values)

        for signal in self.signal_cols:
            long_model  = self.models_.get(f"{signal}_long")
            short_model = self.models_.get(f"{signal}_short")
            if long_model is None or short_model is None:
                raise ValueError(f"Modelo para '{signal}' no entrenado.")

            tau_long  = long_model.effect(X,  T0=0, T1=1)
            tau_short = short_model.effect(X, T0=0, T1=1)

            sig_long  = out[signal].clip(lower=0).fillna(0).values
            sig_short = (-out[signal]).clip(lower=0).fillna(0).values

            out[f"tau_{signal}_long"]    = tau_long
            out[f"tau_{signal}_short"]   = tau_short
            out[f"causal_edge_{signal}"] = sig_long * tau_long - sig_short * tau_short

        return out

    def ate_summary(self) -> pd.DataFrame:
        """
        Average Treatment Effect (ATE) por senal y direccion, para la tabla
        que pide la rubrica:
            'sentiment_signal  +0.0012  -0.0008  ...'
        """
        rows = []
        for signal in self.signal_cols:
            long_model  = self.models_.get(f"{signal}_long")
            short_model = self.models_.get(f"{signal}_short")
            if long_model is None:
                continue
            try:
                ate_long  = float(long_model.ate())
                ate_short = float(short_model.ate())
            except Exception:
                ate_long, ate_short = np.nan, np.nan
            rows.append({
                "signal":    signal,
                "ate_long":  ate_long,
                "ate_short": ate_short,
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test con datos sinteticos
    np.random.seed(0)
    n = 300
    X = np.random.randn(n, 3)
    sig = np.random.uniform(-1, 1, n)
    # Outcome con efecto causal real cuando contexto X[:,0] > 0
    y = 0.01 * sig * (X[:, 0] > 0) + 0.005 * np.random.randn(n)

    df = pd.DataFrame({
        "next_return":  y,
        "my_signal":    sig,
        "atr_pct":      X[:, 0],
        "adx":          X[:, 1],
        "rsi":          X[:, 2],
    })
    est = CausalEffectsEstimator(
        signal_cols=["my_signal"],
        context_cols=["atr_pct", "adx", "rsi"],
    )
    est.fit(df)
    out = est.predict_effects(df)
    print("Columnas anadidas:", [c for c in out.columns if c.startswith(("tau_", "causal_"))])
    print(est.ate_summary())

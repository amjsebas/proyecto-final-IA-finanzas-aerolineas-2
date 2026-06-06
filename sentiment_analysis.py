"""
src/sentiment_analysis.py

Fase 4 del proyecto: sentiment de noticias con LLM (Gemini) y FinBERT.

Tres responsabilidades:
1. FinBERT score por headline (probabilidades + sentiment_score = P_pos - P_neg).
2. Gemini con prompt structured-output: event_type + sentiment_score + confidence.
3. Comparacion LLM vs FinBERT + sentiment hibrido (LLM si confianza alta, FinBERT fallback).

Tambien expone una funcion para auto-clasificar noticias sin etiquetar (usando Gemini)
para crecer la muestra del modeling table.
"""

from __future__ import annotations

import json
import time
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------

_FINBERT = None  # cache global del pipeline

def _load_finbert():
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline
        _FINBERT = pipeline("text-classification",
                            model="ProsusAI/finbert", top_k=None)
    return _FINBERT


def score_finbert(headlines: list[str]) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      finbert_p_pos, finbert_p_neg, finbert_p_neu,
      finbert_sentiment (= p_pos - p_neg, en [-1, +1]),
      finbert_confidence (= max de las 3 probabilidades).
    """
    fb = _load_finbert()
    raw = fb(list(headlines), truncation=True)
    rows = []
    for sl in raw:
        d = {s["label"].lower(): s["score"] for s in sl}
        rows.append({
            "finbert_p_pos": d.get("positive", 0.0),
            "finbert_p_neg": d.get("negative", 0.0),
            "finbert_p_neu": d.get("neutral", 0.0),
        })
    df = pd.DataFrame(rows)
    df["finbert_sentiment"]  = df["finbert_p_pos"] - df["finbert_p_neg"]
    df["finbert_confidence"] = df[["finbert_p_pos", "finbert_p_neg", "finbert_p_neu"]].max(axis=1)
    return df


# ---------------------------------------------------------------------------
# Gemini (LLM)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Eres un analista financiero experto en aerolineas estadounidenses.
Para cada headline, devuelve EXCLUSIVAMENTE un objeto JSON valido con:

{
  "event_type": "<una de: earnings_beat, earnings_miss, guidance_up, guidance_cut, fuel_cost, regulatory_risk, litigation, buyback, dividend, margin_pressure, ma_activity, mgmt_change, macro_shock, war_risk, demand, institutional_investment, irrelevant>",
  "sentiment_score": <float entre -1.0 (muy negativo) y +1.0 (muy positivo), considera contexto financiero>,
  "confidence": <float entre 0.0 y 1.0>,
  "reasoning": "<una frase corta justificando>"
}

Reglas clave:
- 'Company cuts 5,000 jobs' puede ser POSITIVO si mejora margenes.
- Headlines que ya describen el movimiento de precio ('stock falls', 'shares jump',
  'Why the Stock Is Rising') son LEAKAGE - marcalos como irrelevant con sentiment_score=0.0.
- Eventos sectoriales (varias aerolineas mencionadas) son validos con event_type
  macro_shock, fuel_cost, demand o war_risk.
- NO incluyas texto fuera del JSON, ni markdown, ni fences de codigo.
"""


def _configure_gemini(api_key: str, model_name: str = "gemini-1.5-flash"):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def classify_with_gemini(headline: str, ticker: str, model, max_retries: int = 3) -> dict:
    prompt = f"{SYSTEM_PROMPT}\n\nTicker: {ticker}\nHeadline: {headline}"
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0,
                                   "response_mime_type": "application/json"},
            )
            return json.loads(resp.text)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"event_type": "irrelevant",
                        "sentiment_score": 0.0,
                        "confidence": 0.0,
                        "reasoning": f"ERROR: {e}"}
            time.sleep(2 ** attempt)


def score_gemini(df: pd.DataFrame, api_key: str,
                 model_name: str = "gemini-2.5-flash",
                 sleep_between: float = 0.0,
                 progress_every: int = 10,
                 checkpoint_path: str | None = None,
                 checkpoint_every: int = 50) -> pd.DataFrame:
    """
    Aplica Gemini fila a fila. Devuelve un DataFrame con columnas:
      llm_event_type, llm_sentiment_score, llm_confidence, llm_reasoning.

    Parametros nuevos:
        sleep_between    : 0.0 por default (tier pagado no necesita rate-limit artificial).
                           Subir solo si te tira 429s seguidos.
        checkpoint_path  : si se pasa una ruta CSV, guarda progreso cada checkpoint_every filas.
                           Si la ruta ya existe al iniciar, RETOMA desde donde quedo.
        checkpoint_every : cuantas filas procesar antes de guardar el checkpoint.

    Ejemplo:
        score_gemini(df, api_key="...", checkpoint_path="/content/saved/gemini_ckpt.csv")
    """
    import os as _os

    model = _configure_gemini(api_key, model_name)

    # Resume desde checkpoint si existe
    rows: list[dict] = []
    start_idx = 0
    if checkpoint_path and _os.path.exists(checkpoint_path):
        ckpt = pd.read_csv(checkpoint_path)
        rows = ckpt.to_dict(orient="records")
        start_idx = len(rows)
        print(f"  Resume desde checkpoint: {start_idx} filas ya procesadas")

    df_iter = df.reset_index(drop=True).iloc[start_idx:]
    for i, r in df_iter.iterrows():
        rows.append(classify_with_gemini(r["headline"], r["ticker"], model))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  Gemini: {i+1}/{len(df)}")
        # Checkpoint periodico
        if checkpoint_path and (i + 1) % checkpoint_every == 0:
            _os.makedirs(_os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        if sleep_between > 0:
            time.sleep(sleep_between)

    # Save final checkpoint
    if checkpoint_path:
        _os.makedirs(_os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    out = pd.DataFrame(rows)
    out.columns = [f"llm_{c}" for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Sentiment hibrido + comparacion
# ---------------------------------------------------------------------------

def hybrid_sentiment(df: pd.DataFrame, llm_confidence_threshold: float = 0.5) -> pd.DataFrame:
    """
    Para cada fila escoge la senal de Gemini si su confianza es >= threshold,
    y si no, usa FinBERT como fallback.

    Espera que df ya tenga las columnas finbert_* y (opcional) llm_*.
    Anade: sentiment_score, confidence, event_type_final, sentiment_source.
    """
    out = df.copy()
    has_llm = "llm_sentiment_score" in out.columns

    if has_llm:
        use_llm = (out["llm_confidence"].fillna(0) >= llm_confidence_threshold)
        out["sentiment_score"] = np.where(use_llm,
                                          out["llm_sentiment_score"],
                                          out["finbert_sentiment"])
        out["confidence"] = np.where(use_llm,
                                     out["llm_confidence"],
                                     out["finbert_confidence"])
        out["event_type_final"] = np.where(use_llm,
                                           out["llm_event_type"],
                                           out.get("event_type", "unknown"))
        out["sentiment_source"] = np.where(use_llm, "gemini", "finbert")
    else:
        out["sentiment_score"]  = out["finbert_sentiment"]
        out["confidence"]       = out["finbert_confidence"]
        out["event_type_final"] = out.get("event_type", "unknown")
        out["sentiment_source"] = "finbert"
    return out


def compare_finbert_vs_gemini(df: pd.DataFrame) -> dict:
    """
    Calcula correlaciones y % de coincidencia en signo. Devuelve un dict para
    incluir en el reporte / Signal Card.
    """
    from scipy.stats import pearsonr, spearmanr

    if "llm_sentiment_score" not in df.columns:
        return {"available": False}

    fb = df["finbert_sentiment"].values
    lm = df["llm_sentiment_score"].values
    mask = ~(pd.isna(fb) | pd.isna(lm))
    fb, lm = fb[mask], lm[mask]

    pear,  _ = pearsonr(fb, lm)  if len(fb) > 1 else (np.nan, np.nan)
    spear, _ = spearmanr(fb, lm) if len(fb) > 1 else (np.nan, np.nan)
    sign_agree = (np.sign(fb) == np.sign(lm)).mean() if len(fb) else np.nan

    return {
        "available": True,
        "n": int(len(fb)),
        "pearson": float(pear),
        "spearman": float(spear),
        "sign_agreement_pct": float(sign_agree),
    }


# ---------------------------------------------------------------------------
# Auto-clasificacion de noticias sin etiquetar
# ---------------------------------------------------------------------------

def auto_classify_unlabeled(df_unlabeled: pd.DataFrame, api_key: str,
                            model_name: str = "gemini-2.5-flash",
                            confidence_floor: float = 0.5,
                            checkpoint_path: str | None = "/content/saved/gemini_autoclass_ckpt.csv",
                            sleep_between: float = 0.0) -> pd.DataFrame:
    """
    Toma noticias sin clasificacion manual (use_in_model vacio) y las clasifica
    con Gemini. Marca como use_in_model='yes' las que el LLM considera relevantes
    (event_type != irrelevant y confidence >= floor).

    Sirve para crecer la muestra del modeling table de ~43 a ~200+.

    checkpoint_path : guarda progreso periodicamente y RETOMA desde donde quedo
                      si el archivo existe. Default lo pone en /content/saved/ que
                      sobrevive runtime restarts en Colab. Pasa None para desactivar.
    sleep_between   : 0.0 por default (tier pagado). Sube a 0.5+ solo en tier gratis.
    """
    if df_unlabeled.empty:
        return df_unlabeled

    print(f"Auto-clasificando {len(df_unlabeled)} headlines con Gemini...")
    llm = score_gemini(df_unlabeled, api_key=api_key, model_name=model_name,
                       sleep_between=sleep_between,
                       checkpoint_path=checkpoint_path)
    out = pd.concat([df_unlabeled.reset_index(drop=True), llm], axis=1)

    # Marca para uso en el modelo
    relevant = (out["llm_event_type"].astype(str).str.lower() != "irrelevant") & \
               (out["llm_confidence"].fillna(0) >= confidence_floor)
    out["use_in_model"]    = np.where(relevant, "yes", "no")
    out["event_type"]      = out["llm_event_type"]
    out["relevance_level"] = np.where(relevant, "direct", "noise")
    out["notes"]           = "Auto-clasificado con Gemini"
    print(f"  -> {relevant.sum()} marcadas use_in_model=yes ({relevant.mean():.1%})")
    return out

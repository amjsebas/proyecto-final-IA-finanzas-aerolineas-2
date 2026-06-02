# Handoff para revisión — Proyecto final IA Finanzas

**De:** Sebastián
**Para:** José (review) — copia a Mauricio
**Fecha:** 01 de junio de 2026 | **Entrega:** 08 de junio de 2026

---

## Estado en una línea

Ya está armado **todo el código** del sistema multi-factor (las 8 fases del rubric) y el notebook maestro que orquesta el pipeline end-to-end. **Faltan tres cosas:** correr el notebook en Colab con la API key de Gemini, escribir el reporte PDF y armar la presentación PPTX.

---

## Lo que ya está en el repo

```
proyecto_final_IA_finanzas/
├── README.md                 ← portada del proyecto, cómo correrlo
├── SIGNALS.md                ← catálogo de las 13 señales con justificación
├── HANDOFF.md                ← este documento
├── requirements.txt          ← versiones pineadas (reproducibilidad)
├── config.py.example         ← plantilla; cada uno genera su config.py local con su API key
├── .gitignore
├── data/
│   ├── raw/airline_news_for_model.csv    ← seed dataset (43 headlines clasificados)
│   └── processed/                        ← outputs del pipeline
├── src/                      ← 9 módulos, 2,106 líneas de código Python
│   ├── data_collection.py      Fase 2 — yfinance + finvizfinance + price controls
│   ├── technical_indicators.py Fase 3 — RSI, MACD, Stoch, ROC, Bollinger, ATR, ADX, OBV, SMA cross, volume spike
│   ├── custom_signals.py       Fase 3 — 3 señales propias (mean-rev z-score, MTF momentum, rel strength)
│   ├── sentiment_analysis.py   Fase 4 — FinBERT + Gemini structured-output + híbrido + auto-clasificación
│   ├── signal_combiner.py      Fase 5 — combinación ponderada + BUY/HOLD/SELL + 4 configs de pesos
│   ├── causal_effects.py       Fase 5.5 — CausalForestDML wrapper (HTE long/short separados)
│   ├── causal_weighting.py     Fase 5.5 — pesos dinámicos por régimen
│   ├── backtesting.py          Fase 6 — backtest vectorizado + Sharpe/Sortino/MaxDD + rolling causal
│   └── evaluation.py           Fases 7-8 — confusion matrix, ROC-AUC, KS, calibración, position sizing, stops
└── notebooks/
    └── analisis_exploratorio.ipynb   ← 40 celdas, corre las 8 fases end-to-end
```

---

## Cómo funciona el pipeline (1 párrafo)

Se carga el seed CSV (43 headlines clasificados a mano), se augmenta con noticias scrapeadas de yfinance y finvizfinance, y las augmentadas se **auto-clasifican con Gemini** para crecer la muestra. Las noticias clasificadas se pasan por **FinBERT y Gemini en paralelo**, y se usa una señal híbrida: Gemini cuando su confianza es ≥0.5, FinBERT como fallback. Se aplica **timestamp alignment anti-leakage** (after-close → siguiente día hábil) y se agrega a nivel ticker-fecha con features avanzados (recency-weighted, confidence-weighted, sector sentiment separado, relative sentiment). En paralelo se descargan precios de yfinance y se calculan los 10 indicadores técnicos + 3 propias por ticker. Todo se mergea en un **modeling table** que entra al **signal combiner** con los pesos base del rubric (40% momentum + 15% sentiment + ...). Encima se entrena el **CausalForestDML** que estima τ(x) por señal y régimen, y ajusta los pesos dinámicamente. El backtest vectorizado evalúa tres modelos (A=base, B=sentiment_heavy, C=causal) vs buy-and-hold del JETS, y se sacan métricas de clasificación (confusion matrix, ROC-AUC, KS) + calibración + position sizing.

---

## Decisiones clave que tomamos (para discutir)

1. **Anti-leakage:** asumimos noticias after-close y las desplazamos al siguiente día hábil. Decisión conservadora — nunca introduce look-ahead, aunque a veces "perdemos" un día de señal.
2. **Sentiment híbrido Gemini + FinBERT:** Gemini entiende contexto financiero (cuts/raises/guidance), FinBERT es estable como backstop. Si Gemini no está disponible (sin API key), el pipeline degrada a FinBERT-only sin romperse.
3. **Auto-clasificación con Gemini de noticias augmentadas:** así el dataset pasa de ~43 a ~200+ headlines y la Fase 5.5 (causal HTE) tiene suficientes datos para entrenar.
4. **Sectoriales separados:** las noticias que mencionan al sector (no a un ticker individual) se agregan como `sector_sentiment` y producen `relative_sentiment = stock - sector`. Evita doble conteo y captura más señal que el sentiment crudo.
5. **Pesos causales con piso en 0:** si una señal muestra efecto adverso, se atenúa pero NO se invierte automáticamente. Mantiene la estructura conceptual de las señales originales.
6. **Reporte de resultados como exploratorio:** con n=43-200 obs, los IC/p-values no van a ser estadísticamente significativos. Se reporta como hallazgo válido (resultado nulo bien argumentado), no como falla.

---

## Lo que falta (en orden de prioridad)

1. **Correr el notebook completo en Colab.** Lo voy a hacer yo el día 6 con mi API key de Gemini ya configurada. Tarda ~10 minutos. Produce los PNGs y CSVs finales en `outputs/`.
2. **Escribir el reporte PDF** (5-8 páginas en español, estructura del rubric). Lo estoy armando en paralelo; te lo paso para review antes de subirlo.
3. **Armar la presentación PPTX** (15 slides, 15 min). Mismo flujo: te lo paso para review.

---

## Cómo correr el pipeline (para que lo pruebes tú también si quieres)

```
1. Saca tu API key de Gemini gratis en https://aistudio.google.com/app/apikey
2. Clona el repo: git clone https://github.com/amjsebas/proyecto-final-IA-finanzas-aerolineas-2.git
3. cp config.py.example config.py
4. Edita config.py y pega tu GEMINI_API_KEY
5. Abre notebooks/analisis_exploratorio.ipynb en Colab (o local)
6. Runtime → Run all. Tarda ~10 min.
```

---

## Lo que te pido que revises

- **`SIGNALS.md`** — la justificación financiera de cada señal. Si crees que falta alguna o algún parámetro está mal calibrado, dime.
- **`src/signal_combiner.py`** — los pesos base (40/15/13/10/10/7/5). El rubric los sugiere pero podemos discutirlos. Hay 4 configs en `WEIGHT_CONFIGS`.
- **Decisión #2 (sentiment híbrido)** — el threshold de confianza para usar Gemini vs FinBERT está en 0.5. Si crees que debe ser más alto o más bajo, dime el porqué.
- **Decisión #3 (auto-clasificación)** — esto crece la muestra pero hace el dataset menos "limpio" (etiquetado por LLM, no humano). Para el reporte hay que ser explícitos sobre esta decisión.
- **Notebook celdas 19-30** (precios y modeling table) — verifica que los merges están bien hechos y que el target `future_excess_return_5d` se calcula correctamente con `shift(-5)` (sin look-ahead).
- **Notebook celdas 25-28** (causal HTE) — esta es la pieza más nueva del rubric. Vale la pena que la entiendas bien para la presentación.

Cualquier issue, mándame Slack/WhatsApp o abre un issue en el repo. Si todo te checa, dame thumbs up para seguir con reporte y presentación.

— Sebas

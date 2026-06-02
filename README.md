# Sistema Multi-Factor de Senales de Trading con LLMs

**Curso:** Modelos de Inteligencia Artificial para Finanzas — EGADE Business School  
**Profesor:** Luis Angel Lozano Medina  
**Entrega:** 08 de junio de 2026  
**Equipo:** Mauricio Jazo, Sebastian Aceves, Jose Hernandez
**Universo:** Aerolineas US (AAL, DAL, UAL, LUV, JBLU, ULCC) | **Benchmark:** ETF JETS

## Que hace este sistema

Genera senales discretas BUY / HOLD / SELL para 6 tickers de aerolineas combinando:

1. **Indicadores tecnicos clasicos** (Fase 3): RSI, MACD, Stochastic, ROC, Bollinger, ATR, ADX, OBV, cruces SMA, volume spike.
2. **Senales propias** (Fase 3): z-score de reversion a la media, momentum multi-timeframe, fuerza relativa vs JETS.
3. **Sentiment con LLM** (Fase 4): Gemini 1.5 Flash con prompt estructurado + FinBERT (ProsusAI) como benchmark.
4. **Combinacion ponderada** (Fase 5): 73% tecnicos + 15% sentiment + 7% propias + 5% modificador ADX.
5. **Causal HTE** (Fase 5.5): CausalForestDML estima el efecto causal de cada senal por regimen de mercado y ajusta los pesos dinamicamente.
6. **Backtesting** (Fase 6): vectorizado vs buy-and-hold, equity curve, Sharpe/Sortino/MaxDD.
7. **Clasificacion** (Fase 7): confusion matrix, ROC-AUC, KS sobre la senal discreta.
8. **Riesgo** (Fase 8): calibracion, position sizing por confianza y volatilidad, stop-loss.

## Estructura

```
proyecto_final_IA_finanzas/
├── data/
│   ├── raw/                 (CSVs de noticias y precios; no se versionan)
│   └── processed/           (modeling table, signals, features)
├── src/
│   ├── data_collection.py     yfinance + finvizfinance + Google News RSS
│   ├── technical_indicators.py 10 indicadores en [-1, +1]
│   ├── custom_signals.py      3 senales propias con justificacion
│   ├── sentiment_analysis.py  FinBERT + Gemini + comparacion
│   ├── signal_combiner.py     pesos fijos + BUY/HOLD/SELL
│   ├── causal_effects.py      CausalForestDML wrapper (Fase 5.5)
│   ├── causal_weighting.py    pesos dinamicos por regimen
│   ├── backtesting.py         vectorizado + rolling causal
│   └── evaluation.py          metricas de clasificacion + riesgo
├── notebooks/
│   └── analisis_exploratorio.ipynb   pipeline end-to-end
├── outputs/                          PNGs + CSVs generados
├── config.py.example                 plantilla de credenciales
├── requirements.txt
└── README.md
```

## Como correrlo

```bash
# 1. Ambiente
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Credenciales
cp config.py.example config.py
# editar config.py y pegar GEMINI_API_KEY

# 3. Correr el notebook maestro
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

Alternativamente, subir el notebook a Google Colab y correrlo ahi (todas las
dependencias se instalan automaticamente en la primera celda).

## Decisiones de diseno relevantes

- **Anti-leakage:** asumimos noticias publicadas after-close → asignamos al siguiente
  dia habil. Es la decision mas conservadora; nunca introduce leakage.
- **Sentiment hibrido:** Gemini si confianza > 0.5, FinBERT como fallback. Aprovecha
  que el LLM entiende contexto financiero (cuts/raises/guidance) y FinBERT es estable.
- **Sectoriales separados:** noticias que mencionan al sector (no a un ticker) se
  agregan como `sector_sentiment` y producen `relative_sentiment = stock - sector`.
- **Auto-augmentacion:** las noticias scrapeadas con yfinance/finvizfinance se
  clasifican automaticamente con Gemini para crecer la muestra a >200 obs.
- **Causal observacional:** los efectos τ_j(x) son condicionales a los confounders
  observados (X). No es prueba causal definitiva — se reporta como inferencia
  observacional con confounders explicitos.

## Resultados

Los numeros se generan al correr el notebook. La Signal Card final y el reporte PDF
los incluyen automaticamente.

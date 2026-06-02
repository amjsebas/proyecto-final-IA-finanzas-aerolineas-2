# Catalogo de senales del sistema

Documento maestro de las 13 senales utilizadas por el sistema. Para cada una se especifica:
**que mide**, **parametros**, **regla de activacion** (cuando dispara BUY o SELL),
y **referencia** o intuicion financiera.

Todas las senales se normalizan a **[-1, +1]** donde +1 es el sesgo bullish maximo,
-1 el bearish maximo y 0 es neutral. Los modificadores (ADX, volume spike) viven en
**[0, 1]** y multiplican o ponderan, no apuntan en una direccion.

---

## Senales tecnicas clasicas (Fase 3 — Indicadores)

### 1. RSI (Relative Strength Index)
- **Categoria:** Momentum
- **Que mide:** Velocidad y magnitud de los cambios de precio en una ventana. Detecta condiciones de sobrecompra (precio subio demasiado rapido) o sobreventa (cayo demasiado rapido).
- **Parametros:** ventana 14 dias; umbrales 30 (sobreventa) / 70 (sobrecompra).
- **Regla:** Mas alla del umbral 30 -> +1 lineal (bullish, esperamos rebote). Mas alla del 70 -> -1 lineal (bearish, esperamos correccion).
- **Referencia:** J. Welles Wilder (1978), *New Concepts in Technical Trading Systems*.

### 2. Stochastic Oscillator (%K, %D)
- **Categoria:** Momentum
- **Que mide:** Donde esta el cierre actual relativo al rango high-low de los ultimos 14 dias. Es el primo del RSI pero suaviza con una segunda media movil (%D).
- **Parametros:** K period 14, D period 3; umbrales 20 / 80.
- **Regla:** %D < 20 -> +1 lineal. %D > 80 -> -1 lineal.
- **Referencia:** George Lane (anos 50). Mejor en mercados laterales que en tendencias fuertes.

### 3. ROC (Rate of Change)
- **Categoria:** Momentum
- **Que mide:** Porcentaje de cambio del precio en una ventana corta. Captura aceleracion.
- **Parametros:** ventana 10 dias; umbral 5%.
- **Regla:** ROC > +5% -> +1. ROC < -5% -> -1. Entre, escalado lineal.
- **Intuicion:** Mas directo que RSI; es simplemente el retorno a 10 dias normalizado.

### 4. MACD (Moving Average Convergence Divergence)
- **Categoria:** Tendencia
- **Que mide:** Diferencia entre dos EMAs (12 y 26 dias) vs su linea de senal (EMA-9 del MACD). El histograma captura aceleracion de la tendencia.
- **Parametros:** fast=12, slow=26, signal=9 (default Gerald Appel).
- **Regla:** histograma > 0 -> bullish (escalado por tanh de su z-score rolling). Histograma < 0 -> bearish.
- **Referencia:** Gerald Appel (1970s). Estandar de la industria para confirmacion de tendencia.

### 5. Cruce de Medias Moviles (SMA 20/50/200)
- **Categoria:** Tendencia
- **Que mide:** Alineacion de tendencia de corto (20d), medio (50d) y largo plazo (200d). El "Golden Cross" (50d cruza al alza al 200d) es la senal alcista clasica.
- **Parametros:** SMA(20), SMA(50), SMA(200).
- **Regla:** 0.6 * sign(SMA20 - SMA50) + 0.4 * sign(SMA50 - SMA200). Resultado en [-1, +1].
- **Intuicion:** Si las tres medias estan alineadas al alza, la tendencia esta confirmada en multiples horizontes.

### 6. ADX (Average Directional Index) — Modificador
- **Categoria:** Tendencia (fuerza, no direccion)
- **Que mide:** Cuan FUERTE es la tendencia, sin importar la direccion. Un ADX > 25 indica tendencia clara; < 20 indica mercado lateral.
- **Parametros:** ventana 14 dias; umbral 25.
- **Regla:** Devuelve un escalar en [0, 1] que MULTIPLICA la senal combinada. ADX = 25 -> 0. ADX = 50 -> 1.
- **Uso:** En el signal_combiner, refuerza la senal cuando hay tendencia clara (boost de +5%). No produce BUY/SELL por si solo.
- **Referencia:** Wilder (1978), mismo libro que el RSI.

### 7. Bollinger Bands
- **Categoria:** Volatilidad
- **Que mide:** Distancia del precio actual a su media movil de 20 dias, expresada en desviaciones estandar (banda superior = +2 sigma, inferior = -2 sigma). Captura "ruptura" o "compresion" de volatilidad.
- **Parametros:** ventana 20, n_std=2.
- **Regla:** %B = (close - lower) / (upper - lower). Senal = 1 - 2*%B. Cerca de banda inferior -> +1 (esperamos reversion al alza). Cerca de superior -> -1.
- **Referencia:** John Bollinger (1980s).

### 8. ATR (Average True Range) — Feature de regimen
- **Categoria:** Volatilidad (magnitud, no direccion)
- **Que mide:** Rango promedio de movimiento diario de los ultimos 14 dias, expresado como % del precio. NO es direccional — sirve como (a) feature de contexto para el modelo causal, (b) input para position sizing en Fase 8.
- **Parametros:** ventana 14 dias.
- **Uso:** ATR% alto -> reducir tamano de posicion. ATR% como X en el modelo causal HTE.

### 9. OBV (On-Balance Volume)
- **Categoria:** Volumen
- **Que mide:** Acumulado de volumen firmado por el signo del retorno diario. Si la divergencia entre precio y OBV es grande, la tendencia pierde sustento.
- **Parametros:** comparado contra su SMA(20) via z-score.
- **Regla:** OBV por encima de su media -> +tanh(z). Por debajo -> -tanh(z).
- **Referencia:** Joseph Granville (1963).

### 10. Volume Spike — Modificador
- **Categoria:** Volumen (intensidad, no direccion)
- **Que mide:** Volumen del dia relativo a su media de 20 dias. Spike >= 2x marca "algo esta pasando".
- **Parametros:** ventana 20 dias; threshold 2x.
- **Regla:** Devuelve un escalar en [0, 1] que se puede usar como peso de confianza extra. No produce BUY/SELL por si solo.

---

## Senales propias del equipo (Fase 3 — Custom, minimo 2)

### 11. Mean-reversion Z-Score
- **Categoria:** Propia (reversion a la media)
- **Hipotesis:** El precio de una accion individual tiende a revertir hacia su media movil de 20 dias cuando se aleja mas de 2 sigmas. Util para capturar **overreaction a noticias** o eventos macro.
- **Parametros:** ventana 20 dias; cap a 3 sigmas; transformacion tanh.
- **Formula:** z = (close - rolling_mean_20) / rolling_std_20; senal = -tanh(z / 3).
- **Regla:** Precio caro vs su media (z > 0) -> senal bajista. Precio barato (z < 0) -> alcista.
- **Justificacion para el reporte:** Aerolineas tienen alta volatilidad intra-mes por eventos (fuel, weather, earnings). Esta senal captura el "ruido" que tiende a corregirse.

### 12. Multi-Timeframe Momentum
- **Categoria:** Propia (momentum compuesto)
- **Hipotesis:** El momentum **genuino** se confirma cuando las ventanas de 5, 20 y 60 dias apuntan en la misma direccion. Senales contradictorias entre ventanas indican whipsaws.
- **Parametros:** ventanas 5d / 20d / 60d; scale 5%; promedio simple.
- **Formula:** Promedio de tanh(retorno_w / 5%) para w en {5, 20, 60}.
- **Regla:** Ventanas alineadas alza -> hacia +1. Mixtas -> cerca de 0.
- **Justificacion:** Combina rapidez (5d) con persistencia (60d). Filtra mejor que un solo timeframe.

### 13. Relative Strength vs JETS (sector-neutral)
- **Categoria:** Propia (fuerza relativa)
- **Hipotesis:** En stock-picking sector-neutral lo que importa no es si AAL sube, sino si **AAL le gana a JETS**. Aerolineas tienen correlacion intra-sector ~0.7+, asi que el retorno absoluto esta contaminado por beta sectorial.
- **Parametros:** ventana 20 dias; scale 3% de excess return.
- **Formula:** senal = tanh((stock_ret_20d - JETS_ret_20d) / 3%).
- **Regla:** AAL outperform JETS por 3% en 20d -> +1. Underperform por 3% -> -1.
- **Justificacion:** Mas informativa que el retorno absoluto para estrategias long/short dentro del sector. Es exactamente lo que se calcula al final del backtest (excess return vs JETS) — usar la version trailing como input es coherente.

---

## Senal de sentiment (Fase 4 — Noticias)

### Sentiment hibrido Gemini + FinBERT
- **Categoria:** Sentiment de noticias (no tecnica)
- **Engine:** Gemini 1.5 Flash con prompt structured-output (event_type + sentiment_score + confidence) como motor principal; FinBERT (ProsusAI) como fallback cuando la confianza del LLM cae por debajo de 0.5.
- **Que mide:** Sentiment del headline a nivel financiero. Gemini extrae **event_type** estructurado (earnings_beat, guidance_cut, fuel_cost, etc.) ademas del score numerico.
- **Definicion de score:** sentiment_score en [-1, +1]; el promedio diario por ticker se agrega ademas como recency-weighted y confidence-weighted.
- **Anti-leakage:** noticias publicadas after-close se asignan al siguiente dia habil. Headlines que ya describen movimiento de precio ("Stock Is Rising") se marcan como `irrelevant` con score 0.
- **Justificacion:** Gemini entiende contexto financiero (ej. "company cuts 5,000 jobs" puede ser positivo si mejora margenes). FinBERT actua como ancla estable cuando el LLM duda.

---

## Combinacion final (Fase 5 — Signal Combiner)

| Familia              | Peso base | Senales agrupadas |
|----------------------|-----------|--------------------|
| Sentiment            | 15%       | sentiment_score hibrido |
| Momentum             | 40%       | RSI, Stochastic, ROC |
| Tendencia            | 10%       | MACD, SMA cross |
| Volatilidad          | 10%       | Bollinger |
| Volumen              | 13%       | OBV |
| Propias              | 7%        | Mean-rev Z-score, MTF momentum, Relative strength |
| Modificador ADX      | 5%        | Multiplicativo (no aditivo) |

**Senal raw** = suma ponderada * (1 + 0.05 * ADX_strength), clip a [-1, +1].
**Senal discreta** = BUY si raw > 0.25, SELL si raw < -0.25, HOLD en medio.
**Confidence** = |raw|, usado para position sizing en Fase 8.

Configuraciones alternativas exploradas (Fase 5 — al menos 3 experimentos):
- `base` (la de la rubrica)
- `momentum_heavy` (55% momentum, 5% sentiment)
- `sentiment_heavy` (30% sentiment, 30% momentum)
- `equal` (1/6 a cada familia, sanity check)

---

## Ajuste causal de pesos (Fase 5.5 — HTE)

Los pesos base se ajustan dinamicamente por regimen de mercado usando
**CausalForestDML** de econml. Para cada senal estimamos el efecto causal heterogeneo
τ_j(x) sobre el retorno futuro, condicionado al contexto (ATR%, ADX, RSI, volume z-score,
news count, sentiment confidence, vol rolling, retorno rolling).

Si la senal muestra efecto positivo en un regimen, su peso se amplifica via
tanh(edge / scale); si muestra efecto debil o adverso, su peso se reduce
(sin invertirla automaticamente — solo se atenua).

**Nota metodologica:** es causal inference observacional. Se asume que los confounders
relevantes estan en X. No prueba causal definitiva.

"""
src/data_collection.py

Recoleccion de datos para el pipeline:
- Precios OHLCV (yfinance)
- Noticias (yfinance.Ticker.news + finvizfinance + opcional Alpha Vantage)
- Features de control de precio (returns rolling, volatilidad, volumen relativo,
  retorno del benchmark)

Las funciones son puras: no dependen del notebook ni de variables globales.
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import yfinance as yf

try:
    from finvizfinance.quote import finvizfinance
    HAS_FINVIZ = True
except ImportError:
    HAS_FINVIZ = False


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------

def fetch_prices(tickers: Iterable[str], start: str, end: str | None = None,
                 auto_adjust: bool = True) -> dict:
    """
    Descarga OHLCV con yfinance para una lista de tickers.

    Devuelve dict {ticker: DataFrame} donde cada DataFrame tiene
    columnas ['Open','High','Low','Close','Volume'] indexado por fecha.
    """
    end = end or (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tickers = list(tickers)

    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=auto_adjust, progress=False, group_by="ticker")

    out = {}
    for tk in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[tk].copy()
            else:
                df = data.copy()
                df.columns = data.columns
            df = df.dropna(how="all")
            df.index = pd.to_datetime(df.index).tz_localize(None)
            out[tk] = df
        except KeyError:
            print(f"  WARN: sin datos para {tk}")
    return out


# ---------------------------------------------------------------------------
# Noticias - yfinance (esquema nuevo, fixed)
# ---------------------------------------------------------------------------

def fetch_yfinance_news(tickers: Iterable[str],
                        company_names: dict | None = None) -> pd.DataFrame:
    """
    yfinance.Ticker.news cambio su esquema; ahora los campos viven bajo
    item['content']. Esta funcion maneja ambos formatos.
    """
    rows = []
    company_names = company_names or {}

    for tk in tickers:
        try:
            news = yf.Ticker(tk).news or []
            for n in news:
                content = n.get("content", {}) or n
                title = content.get("title") or n.get("title")
                if not title:
                    continue

                # Url y publisher (ambos esquemas)
                url = None
                cu = content.get("canonicalUrl") or content.get("clickThroughUrl")
                if isinstance(cu, dict):
                    url = cu.get("url")
                url = url or n.get("link", "")

                provider = content.get("provider") or {}
                publisher = (provider.get("displayName")
                             if isinstance(provider, dict) else "") or n.get("publisher", "")

                # Fecha
                pub = content.get("pubDate") or content.get("displayTime") or n.get("providerPublishTime")
                try:
                    if isinstance(pub, (int, float)):
                        dt = pd.to_datetime(pub, unit="s", utc=True)
                    else:
                        dt = pd.to_datetime(pub, utc=True)
                    date_str = dt.tz_convert(None).strftime("%Y-%m-%d")
                except Exception:
                    continue

                rows.append({
                    "date": date_str,
                    "ticker": tk,
                    "company": company_names.get(tk, tk),
                    "headline": title,
                    "source": publisher,
                    "url": url,
                    "raw_source": "yfinance",
                })
            time.sleep(0.4)
        except Exception as e:
            print(f"  yfinance {tk}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Noticias - finvizfinance
# ---------------------------------------------------------------------------

def fetch_finviz_news(tickers: Iterable[str],
                      company_names: dict | None = None) -> pd.DataFrame:
    """
    finvizfinance scrapea la tabla de noticias de Finviz por ticker.
    Suele tener mas historico que yfinance.
    """
    if not HAS_FINVIZ:
        print("  finvizfinance no instalado; skipping.")
        return pd.DataFrame()

    rows = []
    company_names = company_names or {}

    for tk in tickers:
        try:
            stock = finvizfinance(tk)
            news_df = stock.ticker_news()
            for _, r in news_df.iterrows():
                try:
                    dt = pd.to_datetime(r["Date"])
                except Exception:
                    continue
                rows.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "ticker": tk,
                    "company": company_names.get(tk, tk),
                    "headline": r.get("Title", ""),
                    "source": r.get("Source", ""),
                    "url": r.get("Link", ""),
                    "raw_source": "finvizfinance",
                })
            time.sleep(0.6)
        except Exception as e:
            print(f"  finviz {tk}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Consolidacion
# ---------------------------------------------------------------------------

def consolidate_news(*frames: pd.DataFrame, dedup_keys=("ticker", "headline")) -> pd.DataFrame:
    """
    Concatena varios DataFrames de noticias y deduplica por (ticker, headline).
    Anade columnas vacias para el flujo del pipeline (relevance_level, use_in_model, etc.).
    """
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if df.empty:
        return df
    df = df.dropna(subset=["headline"])
    df = df.drop_duplicates(subset=list(dedup_keys)).reset_index(drop=True)

    for col, default in [("relevance_level", ""), ("use_in_model", ""),
                         ("event_type", ""), ("notes", "Auto-recolectado")]:
        if col not in df.columns:
            df[col] = default
    return df


# ---------------------------------------------------------------------------
# Features de control de precio
# ---------------------------------------------------------------------------

def compute_price_controls(prices: dict, benchmark_ticker: str,
                           windows=(5, 20)) -> pd.DataFrame:
    """
    Calcula los controles que pide el PDF para no confundir senal de noticias
    con momentum/size/sector:

    past_5d_return, past_20d_return, volatility_20d,
    volume_change_20d, market_return_20d.

    Devuelve un DataFrame long con columnas ['ticker', 'date', <features>].
    """
    if benchmark_ticker not in prices:
        raise ValueError(f"Falta el benchmark {benchmark_ticker} en los precios")

    bench = prices[benchmark_ticker]["Close"]
    bench_ret_20 = bench.pct_change(20)

    rows = []
    for tk, df in prices.items():
        if tk == benchmark_ticker:
            continue
        c = df["Close"]; v = df["Volume"]
        r1 = c.pct_change()
        feat = {
            "past_5d_return":    c.pct_change(5),
            "past_20d_return":   c.pct_change(20),
            "volatility_20d":    r1.rolling(20).std() * np.sqrt(252),
            "volume_change_20d": v / v.rolling(20).mean(),
            "market_return_20d": bench_ret_20.reindex(c.index),
        }
        long = pd.DataFrame(feat)
        long["ticker"] = tk
        long["date"]   = long.index
        rows.append(long.reset_index(drop=True))

    return pd.concat(rows, ignore_index=True)[
        ["ticker", "date", "past_5d_return", "past_20d_return",
         "volatility_20d", "volume_change_20d", "market_return_20d"]
    ]


# ---------------------------------------------------------------------------
# Save / Load helpers
# ---------------------------------------------------------------------------

def save_prices_to_csv(prices: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for tk, df in prices.items():
        df.to_csv(out_dir / f"{tk}.csv")
    print(f"OK - {len(prices)} tickers guardados en {out_dir}")


def load_prices_from_csv(in_dir: str | Path) -> dict:
    in_dir = Path(in_dir)
    out = {}
    for f in sorted(in_dir.glob("*.csv")):
        tk = f.stem
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        out[tk] = df
    return out


if __name__ == "__main__":
    UNIVERSE  = ["AAL", "DAL", "UAL", "LUV", "JBLU", "ULCC"]
    BENCHMARK = "JETS"
    px = fetch_prices(UNIVERSE + [BENCHMARK], start="2024-01-01")
    print(f"Precios: {len(px)} tickers")
    controls = compute_price_controls(px, BENCHMARK)
    print(f"Controles: {controls.shape}")
    print(controls.head())

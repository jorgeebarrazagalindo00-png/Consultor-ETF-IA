# ============================================================================
#  FASE 3 — RECOLECCIÓN Y PREPARACIÓN DE DATOS  (CRISP-DM: Preparación de datos)
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Este script cubre las 4 tareas de la Fase 3 de su cronograma:
#    Tarea 7  -> Identificación de fuentes de datos      (ver comentarios)
#    Tarea 8  -> Obtención de datos históricos OHLCV      (SECCIÓN 2)
#    Tarea 9  -> Obtención de noticias financieras        (SECCIÓN 3)
#    Tarea 10 -> Limpieza, normalización e integración    (SECCIÓN 4)
#
#  CÓMO EJECUTARLO:
#   1. Ábranlo en Spyder, Jupyter o VSCode (ya vi que tienen Anaconda instalado).
#   2. Instalen las librerías que falten:  pip install yfinance pandas requests
#   3. Corran el script completo. Va a crear una carpeta "data/" con los CSV.
#
#  NOTA IMPORTANTE: este script se ejecuta en SU computadora, no en este chat,
#  porque este entorno no tiene acceso a internet para consultar APIs
#  financieras (política de la organización). Ustedes sí lo tienen.
# ============================================================================

import os
import time
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
FECHA_INICIO = "2021-01-01"     # ajusten el rango histórico que necesiten
FECHA_FIN = None                # None = hasta hoy

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clave gratuita de Alpha Vantage para la Tarea 9 (noticias + sentimiento).
# Sáquenla gratis en: https://www.alphavantage.co/support/#api-key
# NUNCA escriban la clave aquí en texto plano (Instrucción #5 del cuestionario).
# Se lee desde la variable de entorno ALPHA_VANTAGE_API_KEY. Si no está
# configurada, el script se salta la Tarea 9 automáticamente.
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")


# ============================================================================
# TAREA 7 — IDENTIFICACIÓN Y RECOPILACIÓN DE FUENTES DE DATOS
# ============================================================================
# Fuentes elegidas (documentar esto también en el informe / hoja Metodología):
#
#   - Datos estructurados (precios OHLCV):
#       Yahoo Finance, vía la librería `yfinance`. Gratis, sin necesidad de
#       API key, y es la misma fuente que usa la mayoría de proyectos
#       académicos de este tipo (equivalente a lo que el Grupo 13 llamó
#       "datos históricos IBKR/Kaggle").
#
#   - Datos no estructurados (noticias financieras):
#       Alpha Vantage News & Sentiment API. Es gratuita (con límite de
#       25 solicitudes/día en el plan free) y trae una ventaja enorme:
#       YA calcula un sentiment score por noticia, así que les puede servir
#       de insumo directo para la Fase 4 (Sentiment Score) sin tener que
#       programar el modelo de NLP desde cero si no les alcanza el tiempo.


# ============================================================================
# TAREA 8 — OBTENCIÓN DE DATOS HISTÓRICOS DE LOS ETF (OHLCV)
# ============================================================================
def descargar_precios_historicos():
    import yfinance as yf

    print("Descargando precios históricos (OHLCV)...")
    resultados = {}
    for ticker in ETFS:
        print(f"  -> {ticker}")
        df = yf.download(ticker, start=FECHA_INICIO, end=FECHA_FIN, progress=False)

        # Las versiones recientes de yfinance devuelven las columnas en dos
        # niveles (por ejemplo ('Close', 'SPY')) incluso al descargar un solo
        # ticker. Los "aplanamos" a un solo nivel (nos quedamos con 'Close',
        # 'Open', 'High', 'Low', 'Volume') para que el resto del script
        # pueda seguir usando nombres de columna simples como "Date".
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df["Ticker"] = ticker
        resultados[ticker] = df
        # Guardar un CSV individual por ETF (insumo crudo, "raw")
        ruta = os.path.join(OUTPUT_DIR, f"raw_precios_{ticker}.csv")
        df.to_csv(ruta, index=False)
        print(f"     guardado en {ruta} ({len(df)} filas)")

    return resultados


# ============================================================================
# TAREA 9 — OBTENCIÓN DE NOTICIAS Y PUBLICACIONES FINANCIERAS
# ============================================================================
def descargar_noticias():
    import requests

    if not ALPHA_VANTAGE_API_KEY:
        print("\n[Tarea 9] No se configuró ALPHA_VANTAGE_API_KEY: se omite la "
              "descarga de noticias. Consigan su clave gratis en "
              "https://www.alphavantage.co/support/#api-key y péguenla arriba.")
        return None

    print("\nDescargando noticias financieras con sentimiento (Alpha Vantage)...")
    todas = []
    for ticker in ETFS:
        print(f"  -> {ticker}")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": ALPHA_VANTAGE_API_KEY,
            "limit": 200,
        }
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        items = data.get("feed", [])
        for item in items:
            todas.append({
                "ticker_consultado": ticker,
                "fecha": item.get("time_published"),
                "fuente": item.get("source"),
                "titulo": item.get("title"),
                "url": item.get("url"),
                "resumen": item.get("summary"),
                "sentiment_score": item.get("overall_sentiment_score"),
                "sentiment_label": item.get("overall_sentiment_label"),
            })
        # Alpha Vantage free tier: máx. 5 llamadas/min -> pausa de seguridad
        time.sleep(15)

    df_noticias = pd.DataFrame(todas)
    ruta = os.path.join(OUTPUT_DIR, "raw_noticias.csv")
    df_noticias.to_csv(ruta, index=False)
    print(f"  guardado en {ruta} ({len(df_noticias)} noticias)")
    return df_noticias


# ============================================================================
# TAREA 10 — LIMPIEZA, NORMALIZACIÓN E INTEGRACIÓN INICIAL
# ============================================================================
def limpiar_e_integrar(precios_por_etf, noticias_df):
    print("\nLimpiando y normalizando precios...")
    partes = []
    for ticker, df in precios_por_etf.items():
        df = df.copy()
        # Normalizar nombres de columnas
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
        # Salvaguarda: si por alguna razón la columna de fecha no se llama
        # exactamente "Date" (por ejemplo si vino como parte de un índice con
        # otro nombre), renombramos la primera columna asumiendo que es la
        # fecha, que es como siempre la deja yfinance tras reset_index().
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})
        # Asegurar tipo fecha
        df["Date"] = pd.to_datetime(df["Date"])
        # Quitar duplicados y filas sin precio de cierre
        df = df.drop_duplicates(subset=["Date"]).dropna(subset=["Close"])
        df = df.sort_values("Date")
        partes.append(df)

    precios_integrado = pd.concat(partes, ignore_index=True)
    ruta_precios = os.path.join(OUTPUT_DIR, "clean_precios_ETFs.csv")
    precios_integrado.to_csv(ruta_precios, index=False)
    print(f"  precios integrados guardados en {ruta_precios} "
          f"({len(precios_integrado)} filas, {precios_integrado['Ticker'].nunique()} ETFs)")

    if noticias_df is not None and not noticias_df.empty:
        print("Limpiando y normalizando noticias...")
        noticias_df = noticias_df.drop_duplicates(subset=["url"])
        noticias_df = noticias_df.dropna(subset=["titulo"])
        noticias_df["fecha"] = pd.to_datetime(
            noticias_df["fecha"], format="%Y%m%dT%H%M%S", errors="coerce"
        )
        ruta_noticias = os.path.join(OUTPUT_DIR, "clean_noticias.csv")
        noticias_df.to_csv(ruta_noticias, index=False)
        print(f"  noticias integradas guardadas en {ruta_noticias} "
              f"({len(noticias_df)} filas)")

    return precios_integrado


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    precios = descargar_precios_historicos()
    noticias = descargar_noticias()
    dataset_final = limpiar_e_integrar(precios, noticias)

    print("\n✅ Fase 3 completada. Archivos listos en la carpeta 'data/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print("   -", f)
    print("\nEstos archivos limpios (clean_precios_ETFs.csv y clean_noticias.csv) "
          "son el insumo directo para la Fase 4 (Feature Engineering y Sentiment Score).")

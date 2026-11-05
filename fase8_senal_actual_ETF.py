# ============================================================================
#  FASE 8 (preparación) — SEÑAL ACTUAL PARA EL DASHBOARD PRESCRIPTIVO
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Este script no es una fase nueva del cronograma: es un paso corto que
#  prepara el insumo que le faltaba al Dashboard Prescriptivo. Las Fases 5
#  y 6 evaluaron el modelo con datos HISTÓRICOS (backtest). Pero un
#  Dashboard Prescriptivo, como se vio en la exposición del Grupo 13, tiene
#  que mostrar una recomendación ACCIONABLE para HOY: señal, probabilidad,
#  nivel de riesgo y el porqué. Eso es justo lo que genera este script.
#
#  Usa data/modelo_final.joblib (el modelo de la Fase 4, entrenado con el
#  100% del historial disponible -es el modelo correcto para usar en
#  producción, a diferencia del modelo "limpio" de la Fase 5, que era solo
#  para el backtest-) sobre la fila MÁS RECIENTE de cada ETF.
#
#  Requiere los archivos que genera la Fase 4:
#    data/dataset_maestro_ETF.csv, modelo_final.joblib, scaler.joblib,
#    features.joblib, label_encoder.joblib (si el ganador fue XGBoost)
#
#  Se ejecuta en su computadora (Anaconda Prompt), igual que las fases anteriores.
# ============================================================================

import os
import numpy as np
import pandas as pd
import joblib

DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset_maestro_ETF.csv")
MODELO_PATH = os.path.join(DATA_DIR, "modelo_final.joblib")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(DATA_DIR, "features.joblib")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.joblib")


def calcular_nivel_de_riesgo(dataset, filas_recientes):
    """Ubica la volatilidad más reciente de cada ETF dentro de su propio
    historial (percentil) y la traduce a Bajo / Medio / Alto. Un ETF con
    volatilidad reciente alta respecto a su propia historia implica más
    incertidumbre en la señal, aunque el modelo diga BUY."""
    niveles, percentiles = [], []
    for _, fila in filas_recientes.iterrows():
        historial = dataset.loc[dataset["Ticker"] == fila["Ticker"], "Volatilidad_20d"]
        percentil = (historial < fila["Volatilidad_20d"]).mean()
        percentiles.append(percentil)
        if percentil < 0.33:
            niveles.append("Bajo")
        elif percentil < 0.66:
            niveles.append("Medio")
        else:
            niveles.append("Alto")
    return niveles, percentiles


def construir_justificacion(fila, percentil_vol):
    """Texto corto y legible con el porqué de la señal, usando los mismos
    indicadores que ve el modelo. Es el insumo para la columna "Justificación"
    del Dashboard Prescriptivo."""
    return (
        f"RSI_14={fila['RSI_14']:.1f}, "
        f"MACD={fila['MACD']:.3f} vs señal={fila['MACD_signal']:.3f}, "
        f"Sentiment={fila['Sentiment_score']:.2f}, "
        f"volatilidad en percentil {percentil_vol*100:.0f}% de su propio historial."
    )


if __name__ == "__main__":
    print("Cargando el modelo final de la Fase 4 y el dataset maestro...")
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    features = joblib.load(FEATURES_PATH)
    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)

    es_xgboost = type(modelo).__name__ == "XGBClassifier"
    label_encoder = joblib.load(LABEL_ENCODER_PATH) if es_xgboost else None

    print("Tomando la fila más reciente de cada ETF (la señal 'de hoy')...")
    filas_recientes = (
        dataset.sort_values("Date").groupby("Ticker").tail(1).reset_index(drop=True)
    )
    for _, fila in filas_recientes.iterrows():
        print(f"  {fila['Ticker']}: última fecha disponible = {fila['Date'].date()}")

    X = scaler.transform(filas_recientes[features].values)
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)

    if es_xgboost:
        clases = label_encoder.inverse_transform(modelo.classes_)
        senales = label_encoder.inverse_transform(pred)
    else:
        clases = modelo.classes_
        senales = pred

    print("\nCalculando el nivel de riesgo (percentil de volatilidad reciente)...")
    niveles_riesgo, percentiles_vol = calcular_nivel_de_riesgo(dataset, filas_recientes)

    resultado = filas_recientes[["Ticker", "Date"]].copy()
    resultado["Señal"] = senales
    resultado["Confianza_modelo_%"] = (proba.max(axis=1) * 100).round(1)
    for i, clase in enumerate(clases):
        resultado[f"Prob_{clase}_%"] = (proba[:, i] * 100).round(1)
    resultado["Nivel_de_riesgo"] = niveles_riesgo
    resultado["Justificacion"] = [
        construir_justificacion(filas_recientes.iloc[i], percentiles_vol[i])
        for i in range(len(filas_recientes))
    ]

    print("\n=== SEÑAL ACTUAL POR ETF (Dashboard Prescriptivo) ===")
    for _, r in resultado.iterrows():
        print(f"\n{r['Ticker']} ({r['Date'].date()}): {r['Señal']}  "
              f"(confianza {r['Confianza_modelo_%']:.1f}%, riesgo {r['Nivel_de_riesgo']})")
        print(f"   {r['Justificacion']}")

    ruta = os.path.join(DATA_DIR, "senal_actual_ETF.csv")
    resultado.to_csv(ruta, index=False)
    print(f"\n✅ Señal actual guardada en {ruta}")
    print("Este archivo, junto con backtest_resultados.csv y kpis_finales_ETF.csv, "
          "es lo que se carga en Streamlit para el Dashboard Prescriptivo.")

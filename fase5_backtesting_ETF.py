# ============================================================================
#  FASE 5 — MODELAMIENTO Y BACKTESTING   (CRISP-DM: Evaluación)
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Cubre las 4 tareas de la Fase 5 de su cronograma:
#    Tarea 15 -> División temporal train/test para el backtest (anti-leakage)
#    Tarea 16 -> Reentrenamiento del modelo ganador SOLO con el período de train
#    Tarea 17 -> Generación de señales BUY/HOLD/SELL y simulación de la estrategia
#    Tarea 18 -> Comparación contra el benchmark (Buy & Hold)
#
#  Requiere los archivos que genera la Fase 4:
#    data/dataset_maestro_ETF.csv
#    data/modelo_final.joblib
#    data/scaler.joblib
#    data/features.joblib
#    data/label_encoder.joblib   (solo existe si el modelo ganador fue XGBoost)
#
#  POR QUÉ ESTE SCRIPT NO USA "modelo_final.joblib" DIRECTAMENTE PARA PREDECIR:
#  ese modelo se entrenó con el 100% de los datos (a propósito, para dejarlo
#  lo más completo posible de cara al uso real). Si lo usáramos para "predecir"
#  sobre esos mismos datos, el backtest sería trampa: el modelo ya habría visto
#  las respuestas. Por eso aquí se toma SOLO la arquitectura del modelo ganador
#  (misma clase, mismos hiperparámetros) y se reentrena desde cero usando
#  únicamente el primer 80% cronológico de cada ETF. El 20% final -que el
#  modelo nunca vio- es el período real de backtest.
#
#  INSTALAR ANTES DE CORRER (además de lo ya instalado en la Fase 4):
#    pip install matplotlib
#
#  Se ejecuta en su computadora (Anaconda Prompt), igual que las Fases 3 y 4.
# ============================================================================

import os
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")  # genera el gráfico como archivo PNG, sin necesitar ventana
import matplotlib.pyplot as plt

DATA_DIR = "data"
DATASET_PATH = os.path.join(DATA_DIR, "dataset_maestro_ETF.csv")
MODELO_PATH = os.path.join(DATA_DIR, "modelo_final.joblib")
FEATURES_PATH = os.path.join(DATA_DIR, "features.joblib")
LABEL_ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.joblib")

TEST_FRACTION = 0.20      # último 20% cronológico de cada ETF = período de backtest
CAPITAL_INICIAL = 10_000  # capital ficticio de partida, solo para graficar la curva


# ============================================================================
# TAREA 15 — DIVISIÓN TEMPORAL TRAIN / TEST (ANTI-LEAKAGE)
# ============================================================================
def dividir_train_test(dataset):
    """Para cada ETF por separado, toma el primer TEST_FRACTION% (cronológico)
    como período de entrenamiento y el resto final como período de backtest.
    Se hace por ETF (y no con una fecha global) porque así funciona aunque
    algún ETF tenga un historial un poco más corto que los demás."""
    train_partes, test_partes = [], []
    for ticker, grupo in dataset.groupby("Ticker"):
        grupo = grupo.sort_values("Date").reset_index(drop=True)
        corte = int(len(grupo) * (1 - TEST_FRACTION))
        train_partes.append(grupo.iloc[:corte])
        test_partes.append(grupo.iloc[corte:])

    train_df = pd.concat(train_partes, ignore_index=True).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_df = pd.concat(test_partes, ignore_index=True).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"  Train: {len(train_df)} filas (usadas para reentrenar el modelo)")
    print(f"  Test / Backtest: {len(test_df)} filas (el modelo NUNCA las vio)")
    for ticker in sorted(dataset["Ticker"].unique()):
        fechas_test = test_df.loc[test_df["Ticker"] == ticker, "Date"]
        if len(fechas_test) > 0:
            print(f"    {ticker}: backtest desde {fechas_test.min()} hasta {fechas_test.max()}")

    return train_df, test_df


# ============================================================================
# TAREA 16 — REENTRENAMIENTO DEL MODELO GANADOR SOLO CON EL TRAIN
# ============================================================================
def reentrenar_modelo_limpio(train_df, features):
    """Clona la clase y los hiperparámetros del modelo ganador de la Fase 4
    (RandomForest, LogisticRegression o XGBoost) y lo entrena desde cero,
    usando SOLO el período de entrenamiento (nunca el de backtest)."""
    from sklearn.preprocessing import StandardScaler

    modelo_referencia = joblib.load(MODELO_PATH)
    nombre_clase = type(modelo_referencia).__name__
    es_xgboost = nombre_clase == "XGBClassifier"

    print(f"  Modelo ganador de la Fase 4: {nombre_clase}")
    print("  Reentrenando una copia nueva, solo con datos de entrenamiento...")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].values)
    y_train = train_df["Label"].values

    # Se clona la MISMA clase y los MISMOS hiperparámetros que ganaron en la
    # Fase 4 (no se vuelve a comparar contra los otros modelos: eso ya se
    # decidió en la Fase 4; aquí solo se reentrena limpio, sin fuga de datos).
    modelo_limpio = type(modelo_referencia)(**modelo_referencia.get_params())

    label_encoder = None
    if es_xgboost:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        y_train_num = label_encoder.transform(y_train)
        modelo_limpio.fit(X_train, y_train_num)
    else:
        modelo_limpio.fit(X_train, y_train)

    return modelo_limpio, scaler, label_encoder, es_xgboost


# ============================================================================
# TAREA 17 — GENERACIÓN DE SEÑALES Y SIMULACIÓN DE LA ESTRATEGIA
# ============================================================================
def generar_senales_y_simular(modelo, scaler, label_encoder, es_xgboost, test_df, features):
    test_df = test_df.copy()
    X_test = scaler.transform(test_df[features].values)

    pred = modelo.predict(X_test)
    if es_xgboost:
        pred = label_encoder.inverse_transform(pred)
    test_df["Senal"] = pred

    print("  Distribución de señales generadas en el período de backtest:")
    print(test_df["Senal"].value_counts().to_string())

    # Retorno realmente ocurrido de un día para el siguiente (NO el retorno a
    # HORIZONTE_DIAS que se usó para construir el label): esto es lo que de
    # verdad se "gana" o "pierde" si se sigue la señal un día de trading.
    test_df = test_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test_df["Close_siguiente"] = test_df.groupby("Ticker")["Close"].shift(-1)
    test_df["retorno_1d"] = test_df["Close_siguiente"] / test_df["Close"] - 1

    # Se descarta la última fila de cada ETF (no tiene "día siguiente" dentro del backtest)
    test_df = test_df.dropna(subset=["retorno_1d"]).reset_index(drop=True)

    # Posición según la señal: BUY = comprado (+1), SELL = vendido en corto (-1),
    # HOLD = fuera del mercado (0). Es una simplificación académica razonable
    # para medir si la señal del modelo tiene valor direccional.
    mapa_posicion = {"BUY": 1, "HOLD": 0, "SELL": -1}
    test_df["Posicion"] = test_df["Senal"].map(mapa_posicion)
    test_df["retorno_estrategia"] = test_df["Posicion"] * test_df["retorno_1d"]

    return test_df


# ============================================================================
# TAREA 18 — COMPARACIÓN CONTRA EL BENCHMARK (BUY & HOLD)
# ============================================================================
def calcular_kpis(retornos_diarios):
    """Calcula KPIs financieros estándar a partir de una serie de retornos
    diarios de portafolio (ya sea de la estrategia o del benchmark)."""
    retornos_diarios = pd.Series(retornos_diarios).dropna()
    equity = CAPITAL_INICIAL * (1 + retornos_diarios).cumprod()

    retorno_total = equity.iloc[-1] / CAPITAL_INICIAL - 1
    media = retornos_diarios.mean()
    std = retornos_diarios.std()
    # Sharpe simplificado (tasa libre de riesgo = 0), anualizado con 252 días de trading
    sharpe = (media / std) * np.sqrt(252) if std > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    win_rate = (retornos_diarios > 0).mean()

    return {
        "Retorno total (%)": retorno_total * 100,
        "Sharpe Ratio (anualizado)": sharpe,
        "Max Drawdown (%)": max_drawdown * 100,
        "Win Rate (%)": win_rate * 100,
    }, equity


def comparar_contra_benchmark(test_df):
    print("\nConstruyendo el portafolio de la ESTRATEGIA (equal-weight, 5 ETFs)...")
    port_estrategia = (
        test_df.groupby("Date")["retorno_estrategia"].mean().rename("retorno_estrategia")
    )
    print("Construyendo el portafolio BENCHMARK (Buy & Hold, equal-weight, 5 ETFs)...")
    port_benchmark = (
        test_df.groupby("Date")["retorno_1d"].mean().rename("retorno_benchmark")
    )

    comparacion = pd.concat([port_estrategia, port_benchmark], axis=1).sort_index()

    kpis_estrategia, equity_estrategia = calcular_kpis(comparacion["retorno_estrategia"])
    kpis_benchmark, equity_benchmark = calcular_kpis(comparacion["retorno_benchmark"])

    print("\n=== KPIs del período de backtest (fuera de muestra) ===")
    print(f"{'KPI':30s} | {'Estrategia (modelo)':20s} | {'Benchmark (Buy&Hold)':20s}")
    for k in kpis_estrategia:
        print(f"{k:30s} | {kpis_estrategia[k]:20.3f} | {kpis_benchmark[k]:20.3f}")

    # Guardar resultados detallados + gráfico de la curva de equity
    comparacion["equity_estrategia"] = equity_estrategia
    comparacion["equity_benchmark"] = equity_benchmark
    ruta_csv = os.path.join(DATA_DIR, "backtest_resultados.csv")
    comparacion.to_csv(ruta_csv)
    print(f"\nResultados diarios guardados en {ruta_csv}")

    plt.figure(figsize=(10, 5))
    plt.plot(equity_estrategia.index, equity_estrategia.values, label="Estrategia (modelo)")
    plt.plot(equity_benchmark.index, equity_benchmark.values, label="Benchmark (Buy & Hold)")
    plt.title("Backtest fuera de muestra: Estrategia vs. Benchmark")
    plt.xlabel("Fecha")
    plt.ylabel(f"Valor del portafolio (capital inicial = ${CAPITAL_INICIAL:,})")
    plt.legend()
    plt.tight_layout()
    ruta_png = os.path.join(DATA_DIR, "backtest_equity_curve.png")
    plt.savefig(ruta_png, dpi=150)
    print(f"Gráfico de la curva de equity guardado en {ruta_png}")

    return kpis_estrategia, kpis_benchmark


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    print("Cargando dataset maestro y artefactos de la Fase 4...")
    dataset = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    features = joblib.load(FEATURES_PATH)

    print("\n[Tarea 15] Dividiendo en train (80%) y test/backtest (20%), por ETF...")
    train_df, test_df = dividir_train_test(dataset)

    print("\n[Tarea 16] Reentrenando el modelo ganador solo con el período de train...")
    modelo, scaler, label_encoder, es_xgboost = reentrenar_modelo_limpio(train_df, features)

    print("\n[Tarea 17] Generando señales y simulando la estrategia en el período de backtest...")
    test_df = generar_senales_y_simular(modelo, scaler, label_encoder, es_xgboost, test_df, features)

    print("\n[Tarea 18] Comparando la estrategia contra el benchmark Buy & Hold...")
    kpis_estrategia, kpis_benchmark = comparar_contra_benchmark(test_df)

    print("\n✅ Fase 5 completada. Revisa la carpeta 'data/':")
    print("   - backtest_resultados.csv  (retornos y curva de equity día por día)")
    print("   - backtest_equity_curve.png (gráfico: Estrategia vs. Benchmark)")

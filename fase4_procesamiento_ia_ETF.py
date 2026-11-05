# ============================================================================
#  FASE 4 — PROCESAMIENTO E IA   (CRISP-DM: Modelado)
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Cubre las 4 tareas de la Fase 4 de su cronograma:
#    Tarea 11 -> Procesamiento de textos mediante NLP/IA      (SECCIÓN 2)
#    Tarea 12 -> Cálculo del sentimiento (Sentiment Score)    (SECCIÓN 2)
#    Tarea 13 -> Integración de datos financieros y sentimiento (SECCIÓN 3)
#    Tarea 14 -> Desarrollo del modelo predictivo             (SECCIÓN 4)
#
#  Requiere los archivos que genera la Fase 3:
#    data/clean_precios_ETFs.csv
#    data/clean_noticias.csv   (opcional; si no existe, el sentimiento queda en 0)
#
#  INSTALAR ANTES DE CORRER:
#    pip install pandas numpy scikit-learn xgboost vaderSentiment joblib
#
#  Se ejecuta en su computadora (Spyder / Jupyter / VSCode), igual que el
#  script de la Fase 3.
# ============================================================================

import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
PRECIOS_PATH = os.path.join(DATA_DIR, "clean_precios_ETFs.csv")
NOTICIAS_PATH = os.path.join(DATA_DIR, "clean_noticias.csv")

# Parámetros del label (variable objetivo) — ajustables
HORIZONTE_DIAS = 5        # a cuántos días mira hacia adelante para definir la señal
UMBRAL_SUBIDA = 0.015     # +1.5% en HORIZONTE_DIAS -> BUY
UMBRAL_BAJADA = -0.015    # -1.5% en HORIZONTE_DIAS -> SELL

# Ventana de tolerancia (días calendario, solo hacia ATRÁS en el tiempo) para
# asociar el Sentiment Score de la noticia más reciente a una fecha de precio.
# Antes se exigía coincidencia EXACTA de fecha (noticia publicada ese mismo
# día calendario), lo cual es más estricto de lo que tiene sentido financiero
# -el efecto de una noticia no desaparece de un día para otro- y además deja
# casi todo en 0 simplemente porque no sale una noticia de cada ETF cada día.
# "Solo hacia atrás" es clave para no mirar al futuro (evita fuga de
# información / look-ahead bias, el mismo principio que ya se cuida con
# TimeSeriesSplit en el entrenamiento).
VENTANA_SENTIMENT_DIAS = 5


# ============================================================================
# SECCIÓN 1 — FEATURE ENGINEERING TÉCNICO (RSI, MACD, medias móviles, volatilidad)
# ============================================================================
def calcular_indicadores_tecnicos(df):
    """Recibe un DataFrame de un solo ETF (ordenado por fecha) y le agrega
    las columnas de indicadores técnicos usadas como features del modelo."""
    df = df.sort_values("Date").copy()
    close = df["Close"]

    # RSI (14 periodos)
    delta = close.diff()
    ganancia = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    perdida = -delta.where(delta < 0, 0.0).rolling(window=14).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Medias móviles
    df["SMA_20"] = close.rolling(window=20).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # Volatilidad (desviación estándar de los retornos diarios, ventana 20d)
    retorno_diario = close.pct_change()
    df["Volatilidad_20d"] = retorno_diario.rolling(window=20).std()

    return df


# ============================================================================
# SECCIÓN 2 — TAREAS 11 y 12: NLP / IA y CÁLCULO DEL SENTIMENT SCORE
# ============================================================================
def calcular_sentiment_score(noticias_df):
    """Aplica análisis de sentimiento (NLP) a cada noticia y arma un
    Sentiment Score diario por ETF, entre -1 (muy negativo) y +1 (muy positivo)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analizador = SentimentIntensityAnalyzer()

    noticias_df = noticias_df.copy()
    noticias_df["texto"] = (
        noticias_df.get("titulo", "").fillna("") + ". " + noticias_df.get("resumen", "").fillna("")
    )

    # compound: puntaje compuesto de VADER, ya normalizado entre -1 y +1
    noticias_df["sentiment_compound"] = noticias_df["texto"].apply(
        lambda t: analizador.polarity_scores(t)["compound"]
    )

    noticias_df["fecha_dia"] = pd.to_datetime(noticias_df["fecha"]).dt.date

    sentiment_diario = (
        noticias_df.groupby(["ticker_consultado", "fecha_dia"])["sentiment_compound"]
        .mean()
        .reset_index()
        .rename(columns={
            "ticker_consultado": "Ticker",
            "fecha_dia": "Date",
            "sentiment_compound": "Sentiment_score",
        })
    )
    sentiment_diario["Date"] = pd.to_datetime(sentiment_diario["Date"])
    return sentiment_diario


# ============================================================================
# SECCIÓN 3 — TAREA 13: INTEGRACIÓN DE DATOS FINANCIEROS Y SENTIMIENTO
# ============================================================================
def construir_dataset_maestro(precios_df, sentiment_diario):
    print("Calculando indicadores técnicos por ETF...")
    partes = []
    for ticker, grupo in precios_df.groupby("Ticker"):
        partes.append(calcular_indicadores_tecnicos(grupo))
    dataset = pd.concat(partes, ignore_index=True)
    # .astype("datetime64[ns]") fuerza la misma resolución de fecha en ambos
    # lados del cruce. pandas 2.x puede parsear "Date" con distinta resolución
    # (microsegundos, segundos, ns...) según de dónde venga cada columna, y
    # merge_asof (a diferencia de un merge normal) exige que ambas coincidan
    # EXACTO o lanza MergeError "incompatible merge keys ... must be the same
    # type" aunque las dos sean fechas válidas.
    dataset["Date"] = pd.to_datetime(dataset["Date"]).astype("datetime64[ns]")

    print(f"Integrando el Sentiment Score diario (ventana de tolerancia: "
          f"{VENTANA_SENTIMENT_DIAS} días hacia atrás, sin mirar al futuro)...")
    if sentiment_diario is not None:
        # merge_asof en vez de merge exacto: para cada fila de precio, toma el
        # Sentiment_score de la noticia más reciente de ESE ticker publicada en
        # o antes de esa fecha, dentro de la ventana de tolerancia.
        # direction="backward" es lo que garantiza que nunca se usa una noticia
        # futura para una fecha pasada (anti-leakage).
        sentiment_ordenado = sentiment_diario.copy()
        sentiment_ordenado["Date"] = pd.to_datetime(
            sentiment_ordenado["Date"]
        ).astype("datetime64[ns]")
        dataset = dataset.sort_values("Date")
        sentiment_ordenado = sentiment_ordenado.sort_values("Date")
        dataset = pd.merge_asof(
            dataset, sentiment_ordenado, on="Date", by="Ticker",
            direction="backward",
            tolerance=pd.Timedelta(days=VENTANA_SENTIMENT_DIAS),
        )
    else:
        dataset["Sentiment_score"] = np.nan
    # Si no hay ninguna noticia dentro de la ventana, se asume sentimiento
    # neutro (0) en vez de perder la fila.
    dataset["Sentiment_score"] = dataset["Sentiment_score"].fillna(0.0)

    print("Construyendo el label (variable objetivo: BUY / HOLD / SELL)...")
    dataset = dataset.sort_values(["Ticker", "Date"])
    dataset["retorno_futuro"] = (
        dataset.groupby("Ticker")["Close"].shift(-HORIZONTE_DIAS) / dataset["Close"] - 1
    )

    def etiquetar(r):
        if pd.isna(r):
            return np.nan
        if r > UMBRAL_SUBIDA:
            return "BUY"
        if r < UMBRAL_BAJADA:
            return "SELL"
        return "HOLD"

    dataset["Label"] = dataset["retorno_futuro"].apply(etiquetar)

    # Se descartan filas sin indicadores completos (inicio de cada serie) o sin label
    # (las últimas HORIZONTE_DIAS filas de cada ETF, que aún no tienen futuro conocido)
    columnas_requeridas = ["RSI_14", "MACD", "MACD_signal", "SMA_20", "EMA_20",
                            "Volatilidad_20d", "Sentiment_score", "Label"]
    dataset_limpio = dataset.dropna(subset=columnas_requeridas).reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    ruta_dataset = os.path.join(DATA_DIR, "dataset_maestro_ETF.csv")
    dataset_limpio.to_csv(ruta_dataset, index=False)
    print(f"  dataset maestro guardado en {ruta_dataset} ({len(dataset_limpio)} filas)")
    print("  distribución del label:")
    print(dataset_limpio["Label"].value_counts())

    return dataset_limpio


# ============================================================================
# SECCIÓN 4 — TAREA 14: DESARROLLO DEL MODELO PREDICTIVO (con validación anti-leakage)
# ============================================================================
def entrenar_y_validar_modelos(dataset):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

    try:
        from xgboost import XGBClassifier
        tiene_xgboost = True
    except ImportError:
        tiene_xgboost = False
        print("[Aviso] xgboost no está instalado (pip install xgboost); "
              "se entrenará solo con Random Forest y Logistic Regression.")

    FEATURES = ["RSI_14", "MACD", "MACD_signal", "SMA_20", "EMA_20",
                "Volatilidad_20d", "Sentiment_score"]

    # Se ordena por fecha: CLAVE para que la validación respete el tiempo
    # (nunca se debe barajar / shuffle en series de tiempo -> eso sería leakage)
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    X = dataset[FEATURES].values
    y = dataset["Label"].values

    modelos = {
        # Nota: desde scikit-learn 1.7 el parámetro "multi_class" fue eliminado;
        # con el solver por defecto (lbfgs) y más de 2 clases, LogisticRegression
        # ya aplica automáticamente una regresión multinomial (softmax).
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    if tiene_xgboost:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_xgb = le.fit_transform(y)
        modelos["XGBoost"] = ("xgb_especial", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="mlogloss", random_state=42,
        ), le)

    # --- Validación cruzada respetando el tiempo (Walk-Forward / TimeSeriesSplit) ---
    # En vez de un K-Fold aleatorio (que "vería el futuro" y contaminaría el
    # entrenamiento), TimeSeriesSplit siempre entrena con datos pasados y
    # valida con datos futuros -> así se evita el anti-leakage.
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    print("\n=== Validación anti-leakage (TimeSeriesSplit, 5 folds) ===")
    resultados = {}
    for nombre, modelo in modelos.items():
        f1s = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if isinstance(modelo, tuple):  # caso especial XGBoost (necesita labels numéricos)
                _, xgb_model, le = modelo
                y_train, y_test = y_xgb[train_idx], y_xgb[test_idx]
                xgb_model.fit(X_train_s, y_train)
                pred = xgb_model.predict(X_test_s)
                f1 = f1_score(y_test, pred, average="macro")
            else:
                y_train, y_test = y[train_idx], y[test_idx]
                modelo.fit(X_train_s, y_train)
                pred = modelo.predict(X_test_s)
                f1 = f1_score(y_test, pred, average="macro")

            f1s.append(f1)
            print(f"  {nombre:20s} | fold {fold} | F1 Macro = {f1:.3f}")

        resultados[nombre] = np.mean(f1s)
        print(f"  {nombre:20s} | F1 Macro promedio = {np.mean(f1s):.3f}\n")

    mejor_modelo = max(resultados, key=resultados.get)
    print(f"Mejor modelo según F1 Macro promedio: {mejor_modelo} "
          f"({resultados[mejor_modelo]:.3f})")

    # --- Entrenamiento final: con TODOS los datos, para usar en la Fase 5 (backtesting) ---
    X_scaled = scaler.fit_transform(X)
    if mejor_modelo == "XGBoost":
        _, xgb_final, le = modelos["XGBoost"]
        xgb_final.fit(X_scaled, y_xgb)
        modelo_final = xgb_final
        joblib.dump(le, os.path.join(DATA_DIR, "label_encoder.joblib"))
    else:
        modelo_final = modelos[mejor_modelo]
        modelo_final.fit(X_scaled, y)

    joblib.dump(modelo_final, os.path.join(DATA_DIR, "modelo_final.joblib"))
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.joblib"))
    joblib.dump(FEATURES, os.path.join(DATA_DIR, "features.joblib"))
    print(f"\nModelo final ({mejor_modelo}) guardado en data/modelo_final.joblib, "
          f"listo para la Fase 5 (backtesting).")

    return modelo_final, resultados


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    print("Cargando datos de la Fase 3...")
    precios = pd.read_csv(PRECIOS_PATH)

    if os.path.exists(NOTICIAS_PATH):
        noticias = pd.read_csv(NOTICIAS_PATH)
        sentiment_diario = calcular_sentiment_score(noticias)
    else:
        print("[Aviso] No se encontró data/clean_noticias.csv: el Sentiment_score "
              "quedará en 0 para todas las filas. Corran primero la Tarea 9 de la Fase 3.")
        sentiment_diario = None

    dataset = construir_dataset_maestro(precios, sentiment_diario)
    entrenar_y_validar_modelos(dataset)

    print("\n✅ Fase 4 completada. Revisa la carpeta 'data/':")
    for f in sorted(os.listdir(DATA_DIR)):
        print("   -", f)

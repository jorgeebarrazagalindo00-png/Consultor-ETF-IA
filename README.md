# Radar ETF — Grupo 10

Proyecto académico de Business Analytics / BI sobre 5 ETFs (**SPY, QQQ, IWM,
GLD, TLT**), con tres reportes obligatorios construidos sobre una misma base
de datos limpia:

- **Descriptivo** — indicadores técnicos históricos (precio, medias móviles,
  RSI, MACD, volatilidad).
- **Predictivo** — modelo de Machine Learning propio (XGBoost, con
  validación temporal anti-leakage y backtesting) que predice BUY/HOLD/SELL.
- **Prescriptivo** — recomendación en lenguaje natural generada por un LLM
  (Google Gemini) a partir de la señal del modelo, nunca al revés.

**Tablero en vivo:** _(pegar aquí la URL de Streamlit Community Cloud una vez
publicado)_

---

## Arquitectura

```
Fuentes de datos            ETL / Features         Modelos              Tablero
──────────────────          ──────────────         ──────────────       ──────────
Yahoo Finance (yfinance) ─┐                     ┌─ Modelo ML (XGBoost) ─┐
                           ├─ fase3 → fase4 ────►┤                      ├─► Streamlit
Alpha Vantage (noticias) ─┘   (limpieza,         └─ LLM (Gemini, fase9)─┘   (fase8 +
                                indicadores,                                 fase9 +
                                sentimiento)                                 data/*.csv)
```

Cada fase es un script independiente, pensado para correrse en orden:

| Script | Qué hace |
|---|---|
| `fase3_recoleccion_datos_ETF.py` | Descarga precios (yfinance) y noticias con sentimiento (Alpha Vantage). |
| `fase4_procesamiento_ia_ETF.py` | Limpieza, features, entrena y compara 3 modelos (Logistic Regression, Random Forest, XGBoost) con validación temporal (`TimeSeriesSplit`). |
| `fase5_backtesting_ETF.py` | Backtesting de la estrategia del modelo ganador vs. benchmark (Buy & Hold). |
| `fase6_metricas_finales_ETF.py` | Consolida los KPIs finales (Sharpe, Drawdown, IC Spearman, etc.). |
| `fase8_senal_actual_ETF.py` | Genera la señal BUY/HOLD/SELL más reciente por ETF. |
| `fase9_llm_prescriptivo_ETF.py` | Llama a Gemini para redactar la recomendación prescriptiva a partir de la señal (nunca decide la acción, solo la explica). |
| `dashboard_app.py` | Tablero Streamlit que muestra los tres reportes a partir de los CSV generados arriba. |

---

## Cómo correrlo localmente

1. Clonen el repositorio e instalen las dependencias del pipeline completo:
   ```
   pip install -r requirements-pipeline.txt
   ```
2. Configuren sus claves como variables de entorno (nunca en archivos —
   vean `.env.example`):
   ```
   setx ALPHA_VANTAGE_API_KEY "su_clave_real"
   setx GEMINI_API_KEY "su_clave_real"
   ```
   (cierren y vuelvan a abrir la terminal para que aplique)
3. Corran las fases en orden: `fase3` → `fase4` → `fase5` → `fase6` → `fase8`
   → `fase9`.
4. Para ver el tablero localmente:
   ```
   pip install -r requirements.txt
   streamlit run dashboard_app.py
   ```

## Actualización automática de datos

Los datos y el modelo se actualizan solos, una vez al día, mediante un
workflow de GitHub Actions (`.github/workflows/actualizar_datos.yml`): corre
las fases 3 a 9 con las claves guardadas como *Secrets* de GitHub (nunca en
el código), reentrena el modelo con los datos más recientes, y guarda los
resultados de vuelta en el repositorio. El tablero público simplemente lee
esos archivos actualizados. El historial de corridas es público y
verificable en la pestaña **Actions** del repositorio.

## Seguridad

Ninguna clave de API se escribe en el código ni en este repositorio — solo
se referencian por nombre de variable de entorno (`ALPHA_VANTAGE_API_KEY`,
`GEMINI_API_KEY`). Ver `.env.example`.

## Aviso

Este es un proyecto académico. Las recomendaciones que genera el reporte
prescriptivo **no constituyen asesoría financiera**. No ejecuten operaciones
reales de compra o venta basándose en este resultado.

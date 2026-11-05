import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import date

st.set_page_config(
    page_title="Radar ETF — Grupo 10",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paleta (validada: orden categorico fijo, pares CVD-seguros; ver skill de
# dataviz). No se usan colores sueltos en ningun otro lugar del archivo.
# ---------------------------------------------------------------------------
C_PRIMARY = "#2a78d6"     # slot 1 — serie principal / estrategia
C_SECONDARY = "#eb6834"   # slot 2 — serie de referencia / benchmark
C_TERTIARY = "#1baf7a"    # slot 3 — tercera serie
C_MUTED = "#898781"       # lineas de referencia, ejes
C_GRID = "#e1e0d9"
C_SURFACE = "#ffffff"
C_TEXT = "#0b0b0b"
C_TEXT_SECONDARY = "#52514e"

C_GOOD = "#0ca30c"        # BUY / vela alcista
C_CRITICAL = "#d03b3b"    # SELL / vela bajista
C_NEUTRAL = "#6b6a65"     # HOLD (no es "problema", es neutral)

DATA_DIR = "data"
ETFS_DISPONIBLES = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}
GITHUB_REPO = "jorgeebarrazagalindo00-png/Consultor-ETF-IA"


@st.cache_data
def cargar_precios():
    ruta = os.path.join(DATA_DIR, "clean_precios_ETFs.csv")
    df = pd.read_csv(ruta, parse_dates=["Date"])
    return df


def calcular_indicadores(df):
    df = df.sort_values("Date").copy()
    close = df["Close"]

    delta = close.diff()
    ganancia = delta.clip(lower=0).rolling(window=14).mean()
    perdida = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Caso borde: si en la ventana de 14 dias no hubo NINGUNA perdida, el RSI
    # es 100 por definicion (no NaN) — la division de arriba lo deja en NaN
    # porque perdida=0. Las primeras 13 filas (ventana incompleta) siguen en
    # NaN correctamente, porque ahi "perdida" tambien es NaN (no 0).
    rsi = rsi.mask(perdida.eq(0) & ganancia.notna(), 100.0)
    df["RSI_14"] = rsi

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["SMA_20"] = close.rolling(window=20).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    retorno_diario = close.pct_change()
    df["Volatilidad_20d"] = retorno_diario.rolling(window=20).std()

    return df


def _layout_grafica(fig, titulo, y_label, rangeslider=False):
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=15, color=C_TEXT)),
        yaxis_title=y_label, xaxis_title="Fecha",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.14, font=dict(color=C_TEXT_SECONDARY)),
        margin=dict(t=55, b=40, l=40, r=20), height=360,
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=C_TEXT_SECONDARY),
        xaxis=dict(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID, rangeslider=dict(visible=rangeslider)),
        yaxis=dict(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID),
    )
    return fig


def grafica_lineas(x, series, titulo, y_label, colores=None, hlines=None):
    """series: dict {nombre: valores}, en orden. hlines: lista de (y, texto)."""
    fig = go.Figure()
    paleta_default = [C_PRIMARY, C_SECONDARY, C_TERTIARY]
    for i, (nombre, valores) in enumerate(series.items()):
        color = colores[i] if colores else paleta_default[i % len(paleta_default)]
        fig.add_trace(go.Scatter(
            x=x, y=valores, mode="lines", name=nombre,
            line=dict(width=2, color=color),
        ))
    for hl in (hlines or []):
        y, texto = hl
        fig.add_hline(
            y=y, line_color=C_MUTED, line_dash="dash", line_width=1,
            annotation_text=texto, annotation_position="top left",
            annotation_font_color=C_TEXT_SECONDARY,
        )
    return _layout_grafica(fig, titulo, y_label)


def grafica_velas(df, titulo):
    """Candlestick de Open/High/Low/Close con SMA20/EMA20 superpuestas."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Precio",
        increasing_line_color=C_GOOD, increasing_fillcolor=C_GOOD,
        decreasing_line_color=C_CRITICAL, decreasing_fillcolor=C_CRITICAL,
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SMA_20"], mode="lines", name="SMA 20",
        line=dict(width=1.5, color=C_PRIMARY),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["EMA_20"], mode="lines", name="EMA 20",
        line=dict(width=1.5, color=C_TERTIARY),
    ))
    return _layout_grafica(fig, titulo, "Precio (USD)")


def explicacion(texto):
    with st.expander("¿Cómo leer este gráfico?"):
        st.caption(texto)


def _pastilla(color, texto):
    return (
        f'<span style="background:{color}1a;color:{color};'
        f'border:1px solid {color}66;padding:3px 12px;border-radius:14px;'
        f'font-weight:600;font-size:0.82rem;white-space:nowrap;margin-right:8px;">{texto}</span>'
    )


def badge_accion(accion):
    mapa = {
        "BUY": (C_GOOD, "▲ BUY"),
        "SELL": (C_CRITICAL, "▼ SELL"),
        "HOLD": (C_NEUTRAL, "● HOLD"),
    }
    color, texto = mapa.get(accion, (C_NEUTRAL, str(accion)))
    return _pastilla(color, texto)


# ---------------------------------------------------------------------------
# Barra lateral: filtros globales (ETF, rango de fechas) compartidos entre
# pestañas — así el dashboard se siente como un solo panel, no 4 scripts.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Radar ETF — Grupo 10")
    st.caption("Proyecto de Business Analytics / BI")
    st.divider()

    etf_global = st.selectbox("ETF", ETFS_DISPONIBLES, key="etf_global")

    rango_fechas = None
    try:
        _precios_sidebar = cargar_precios()
        _fechas_etf = _precios_sidebar.loc[
            _precios_sidebar["Ticker"] == etf_global, "Date"
        ]
        _fmin, _fmax = _fechas_etf.min().date(), _fechas_etf.max().date()
        rango_fechas = st.slider(
            "Rango de fechas (Descriptivo)",
            min_value=_fmin, max_value=_fmax, value=(_fmin, _fmax),
        )
    except FileNotFoundError:
        st.caption("Corran fase3 para habilitar el filtro de fechas.")

    st.divider()
    st.caption(
        "Tres reportes: **Descriptivo** (indicadores técnicos), "
        "**Predictivo** (ML con backtesting) y **Prescriptivo** "
        "(recomendación vía LLM)."
    )

st.title("Radar ETF — Grupo 10")

try:
    _ultima_fecha_senal = pd.read_csv(os.path.join(DATA_DIR, "senal_actual_ETF.csv"))["Date"].iloc[0]
    st.caption(f"SPY · QQQ · IWM · GLD · TLT · Última señal calculada: {_ultima_fecha_senal}")
except (FileNotFoundError, IndexError):
    _ultima_fecha_senal = None
    st.caption("SPY · QQQ · IWM · GLD · TLT")

# ---------------------------------------------------------------------------
# Estado del sistema: 3 indicadores rapidos de que "esto esta vivo" — datos,
# ultima recomendacion del LLM, y el estado real del workflow en GitHub
# Actions (el badge es una imagen que GitHub actualiza solo, en vivo).
# ---------------------------------------------------------------------------
_chips = []
try:
    _fecha_senal_dt = pd.to_datetime(_ultima_fecha_senal).date()
    _dias = (date.today() - _fecha_senal_dt).days
    _color = C_GOOD if _dias <= 5 else C_CRITICAL
    _chips.append(_pastilla(_color, f"📊 Datos y señal: {_fecha_senal_dt.isoformat()}"))
except (TypeError, ValueError):
    _chips.append(_pastilla(C_MUTED, "📊 Datos: no disponible"))

try:
    _log_llm = pd.read_csv(os.path.join(DATA_DIR, "log_recomendaciones_LLM.csv"))
    _ultimo_llm = pd.to_datetime(_log_llm["timestamp"]).max()
    _dias_llm = (pd.Timestamp.now() - _ultimo_llm).days
    _color_llm = C_GOOD if _dias_llm <= 5 else C_CRITICAL
    _chips.append(_pastilla(_color_llm, f"🤖 Recomendación LLM: {_ultimo_llm.strftime('%Y-%m-%d')}"))
except (FileNotFoundError, KeyError, ValueError):
    _chips.append(_pastilla(C_MUTED, "🤖 LLM: no disponible"))

_badge_actions = (
    f'<img src="https://github.com/{GITHUB_REPO}/actions/workflows/actualizar_datos.yml/badge.svg" '
    f'alt="estado del workflow de actualizacion automatica" style="vertical-align:middle;">'
)
st.markdown(
    f'<div style="margin:2px 0 10px 0;">{"".join(_chips)}{_badge_actions}</div>',
    unsafe_allow_html=True,
)

tab_home, tab_descriptivo, tab_predictivo, tab_prescriptivo, tab_metodologia = st.tabs(
    ["Home", "Descriptivo", "Predictivo", "Prescriptivo", "Metodología"]
)

with tab_home:
    st.subheader("Resumen del portafolio")
    st.write(
        "Proyecto de Business Analytics / BI sobre 5 ETFs, con tres reportes: "
        "**Descriptivo** (indicadores tecnicos historicos), **Predictivo** "
        "(modelo de Machine Learning con backtesting) y **Prescriptivo** "
        "(recomendacion en lenguaje natural generada por un LLM)."
    )

    try:
        senal_resumen = pd.read_csv(os.path.join(DATA_DIR, "senal_actual_ETF.csv"))
    except FileNotFoundError:
        st.error(
            "No se encontro data/senal_actual_ETF.csv. "
            "Corran primero fase8_senal_actual_ETF.py desde esta misma carpeta."
        )
    else:
        st.markdown("#### Señal más reciente por ETF")
        cols = st.columns(len(ETFS_DISPONIBLES))
        orden = senal_resumen.set_index("Ticker").loc[ETFS_DISPONIBLES].reset_index()
        for col, (_, fila) in zip(cols, orden.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div style="text-align:center;">'
                        f'<div style="font-weight:700;font-size:1.05rem;color:{C_TEXT};">{fila["Ticker"]}</div>'
                        f'<div style="margin:8px 0;">{badge_accion(fila["Señal"])}</div>'
                        f'<div style="font-size:0.78rem;color:{C_TEXT_SECONDARY};">'
                        f'confianza {fila["Confianza_modelo_%"]:.0f}% · riesgo {fila["Nivel_de_riesgo"]}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
        st.caption(
            "Tip: el ETF que elijas en la barra lateral se usa en las pestañas "
            "Descriptivo y Prescriptivo."
        )

with tab_descriptivo:
    st.subheader(f"Indicadores técnicos — {etf_global}")

    try:
        precios = cargar_precios()
    except FileNotFoundError:
        st.error(
            "No se encontro data/clean_precios_ETFs.csv. "
            "Corran primero fase3_recoleccion_datos_ETF.py desde esta misma carpeta."
        )
    else:
        df = precios[precios["Ticker"] == etf_global]
        df = calcular_indicadores(df)  # sobre el historial completo (evita bordes NaN)
        if rango_fechas:
            ini, fin = rango_fechas
            df = df[(df["Date"] >= pd.Timestamp(ini)) & (df["Date"] <= pd.Timestamp(fin))]

        with st.container(border=True):
            fig1 = grafica_velas(df, f"{etf_global} · Precio (velas) y medias móviles")
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
            explicacion(
                "Cada 'vela' resume el día: apertura, cierre, máximo y mínimo. "
                "Verde = el precio subió ese día; rojo = bajó. Las líneas de "
                "color son promedios móviles (SMA/EMA) que suavizan el ruido "
                "diario para ver mejor la tendencia de fondo."
            )

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                fig2 = grafica_lineas(
                    df["Date"], {"RSI (14)": df["RSI_14"]}, "RSI (14 días)", "RSI",
                    colores=[C_PRIMARY],
                    hlines=[(70, "Sobrecompra (70)"), (30, "Sobreventa (30)")],
                )
                st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
                explicacion(
                    "El RSI mide la fuerza de los movimientos recientes del "
                    "precio, en una escala de 0 a 100. Arriba de 70 se "
                    "considera 'sobrecomprado' (posible corrección a la "
                    "baja); abajo de 30, 'sobrevendido' (posible rebote)."
                )

        with col2:
            with st.container(border=True):
                fig3 = grafica_lineas(
                    df["Date"], {"Volatilidad 20d": df["Volatilidad_20d"]},
                    "Volatilidad (20 días)", "Desv. estándar de retornos",
                    colores=[C_SECONDARY],
                )
                st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)
                explicacion(
                    "Mide qué tan dispersos están los retornos diarios en "
                    "una ventana de 20 días. Valores más altos = movimientos "
                    "de precio más bruscos e impredecibles (mayor riesgo)."
                )

        with st.container(border=True):
            fig4 = grafica_lineas(
                df["Date"], {"MACD": df["MACD"], "Señal (9)": df["MACD_signal"]},
                "MACD", "MACD", hlines=[(0, "")],
            )
            st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG)
            explicacion(
                "Compara dos promedios móviles exponenciales (12 y 26 días) "
                "para detectar cambios de tendencia. Cuando la línea MACD "
                "cruza por encima de la línea de señal suele leerse como "
                "alcista; por debajo, como bajista."
            )

with tab_predictivo:
    st.subheader("Predictivo — desempeño del modelo (XGBoost) vs. benchmark")

    try:
        kpis = pd.read_csv(os.path.join(DATA_DIR, "kpis_finales_ETF.csv"), index_col=0)
    except FileNotFoundError:
        st.error(
            "No se encontro data/kpis_finales_ETF.csv. "
            "Corran primero fase5_backtesting_ETF.py y fase6_metricas_finales_ETF.py."
        )
    else:
        def _kpi(nombre):
            return float(kpis.loc[nombre, "Estrategia (modelo)"]), float(kpis.loc[nombre, "Benchmark (Buy & Hold)"])

        with st.container(border=True):
            try:
                ret_m, ret_b = _kpi("Retorno total (%)")
                sharpe_m, sharpe_b = _kpi("Sharpe Ratio (anualizado)")
                dd_m, dd_b = _kpi("Max Drawdown (%)")

                m1, m2, m3 = st.columns(3)
                m1.metric("Retorno total", f"{ret_m:.2f}%", f"{ret_m - ret_b:+.2f} pp vs. benchmark")
                m2.metric("Sharpe Ratio", f"{sharpe_m:.2f}", f"{sharpe_m - sharpe_b:+.2f} vs. benchmark")
                m3.metric("Max Drawdown", f"{dd_m:.2f}%", f"{dd_m - dd_b:+.2f} pp vs. benchmark")
            except (KeyError, ValueError):
                pass

        with st.container(border=True):
            st.markdown("#### KPIs completos: estrategia del modelo vs. Buy & Hold")
            st.dataframe(kpis, use_container_width=True)

            try:
                ic = float(kpis.loc["IC Spearman", "Estrategia (modelo)"])
                ic_p = float(kpis.loc["IC Spearman (p-valor)", "Estrategia (modelo)"])
                if ic_p < 0.05:
                    st.success(
                        f"IC Spearman = {ic:.4f} (p = {ic_p:.4f} < 0.05) — "
                        "estadísticamente significativo: el modelo sí tiene valor "
                        "direccional real, más allá del azar."
                    )
                else:
                    st.warning(
                        f"IC Spearman = {ic:.4f} (p = {ic_p:.4f}) — no alcanza "
                        "significancia estadística al 5%."
                    )
            except (KeyError, ValueError):
                pass

    try:
        backtest = pd.read_csv(
            os.path.join(DATA_DIR, "backtest_resultados.csv"), parse_dates=["Date"]
        )
    except FileNotFoundError:
        st.error("No se encontro data/backtest_resultados.csv.")
    else:
        with st.container(border=True):
            fig5 = grafica_lineas(
                backtest["Date"],
                {
                    "Estrategia (modelo)": backtest["equity_estrategia"],
                    "Benchmark (Buy & Hold)": backtest["equity_benchmark"],
                },
                "Curva de equity: estrategia vs. benchmark (base $10,000)",
                "Valor del portafolio (USD)",
            )
            st.plotly_chart(fig5, use_container_width=True, config=PLOTLY_CONFIG)
            explicacion(
                "Simula cómo habría crecido una inversión inicial de "
                "$10,000 siguiendo las señales del modelo, comparado con "
                "simplemente comprar y mantener (buy & hold) el mismo ETF "
                "durante todo el periodo."
            )

            st.caption(
                "Nota honesta: el modelo no supera al benchmark en retorno total, "
                "pero sí en riesgo (menor drawdown, mejor relación riesgo/beneficio). "
                "El IC Spearman significativo confirma que las señales sí tienen "
                "valor predictivo real, aunque modesto — no se oculta este resultado."
            )

with tab_prescriptivo:
    st.subheader(f"Prescriptivo — recomendación para {etf_global} (vía LLM)")
    st.warning(
        "Esta recomendación la genera un sistema académico y NO constituye "
        "asesoría financiera personalizada. No ejecuten operaciones reales de "
        "compra o venta basándose únicamente en este resultado."
    )

    try:
        log = pd.read_csv(os.path.join(DATA_DIR, "log_recomendaciones_LLM.csv"))
    except FileNotFoundError:
        st.error(
            "No se encontro data/log_recomendaciones_LLM.csv. "
            "Corran primero fase9_llm_prescriptivo_ETF.py."
        )
    else:
        ultima = (
            log.sort_values("timestamp")
            .groupby("ticker")
            .tail(1)
            .set_index("ticker")
        )

        if etf_global not in ultima.index:
            st.info("Todavía no hay recomendación registrada para este ETF.")
        else:
            fila = ultima.loc[etf_global]
            origen_texto = (
                "LLM (Gemini)" if fila["origen"] == "LLM"
                else "respaldo por reglas fijas (el LLM no respondió o falló la validación)"
            )

            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(badge_accion(fila["accion_sugerida"]), unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="margin-top:10px;color:{C_TEXT_SECONDARY};font-size:0.9rem;">'
                        f'Confianza: <b style="color:{C_TEXT};">{fila["nivel_confianza"]}</b></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.caption(f"Origen: {origen_texto}  ·  Generado: {fila['timestamp']}")
                    st.markdown("**Justificación**")
                    st.write(fila["justificacion"])

                colr, cola = st.columns(2)
                with colr:
                    st.markdown("**Riesgos**")
                    for riesgo in str(fila["riesgos"]).split(" | "):
                        st.write(f"- {riesgo}")
                with cola:
                    st.markdown("**Alternativas**")
                    for alt in str(fila["alternativas"]).split(" | "):
                        st.write(f"- {alt}")

        with st.container(border=True):
            st.markdown("#### Historial completo de recomendaciones (log de auditoría)")
            st.dataframe(
                log[[
                    "timestamp", "ticker", "senal_ml", "origen",
                    "accion_sugerida", "nivel_confianza", "validado_ok",
                ]].rename(columns={
                    "timestamp": "Fecha/hora",
                    "ticker": "ETF",
                    "senal_ml": "Señal del modelo ML",
                    "origen": "Origen",
                    "accion_sugerida": "Acción sugerida",
                    "nivel_confianza": "Confianza",
                    "validado_ok": "Pasó validación",
                }),
                use_container_width=True,
                hide_index=True,
            )

with tab_metodologia:
    st.subheader("Metodología — cómo está construido este proyecto")
    st.write(
        "Cada fase es un script independiente que corre en orden. Los datos "
        "se actualizan y el modelo se reentrena automáticamente todos los "
        "días mediante un workflow de GitHub Actions — nadie tiene que "
        "correr nada manualmente para que esto se mantenga al día."
    )

    _svg_diagrama = f"""
    <svg viewBox="0 0 1180 200" style="width:100%;height:auto;max-width:1180px;" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="flecha" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="{C_MUTED}"/>
        </marker>
      </defs>

      <line x1="262" y1="90" x2="316" y2="90" stroke="{C_MUTED}" stroke-width="2" marker-end="url(#flecha)"/>
      <line x1="572" y1="90" x2="626" y2="90" stroke="{C_MUTED}" stroke-width="2" marker-end="url(#flecha)"/>
      <line x1="882" y1="90" x2="936" y2="90" stroke="{C_MUTED}" stroke-width="2" marker-end="url(#flecha)"/>

      <rect x="10" y="20" width="250" height="140" rx="10" fill="{C_SURFACE}" stroke="{C_GRID}" stroke-width="1.5"/>
      <rect x="10" y="20" width="250" height="6" rx="3" fill="{C_MUTED}"/>
      <text x="135" y="55" text-anchor="middle" font-size="15" font-weight="700" fill="{C_TEXT}">Fuentes de datos</text>
      <text x="135" y="80" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">Yahoo Finance</text>
      <text x="135" y="98" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(precios, via yfinance)</text>
      <text x="135" y="120" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">Alpha Vantage</text>
      <text x="135" y="138" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(noticias + sentimiento)</text>

      <rect x="320" y="20" width="250" height="140" rx="10" fill="{C_SURFACE}" stroke="{C_GRID}" stroke-width="1.5"/>
      <rect x="320" y="20" width="250" height="6" rx="3" fill="{C_PRIMARY}"/>
      <text x="445" y="55" text-anchor="middle" font-size="15" font-weight="700" fill="{C_TEXT}">ETL / Features</text>
      <text x="445" y="80" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">fase3 → fase4</text>
      <text x="445" y="100" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">limpieza, indicadores</text>
      <text x="445" y="118" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">tecnicos, sentimiento</text>
      <text x="445" y="136" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(merge_asof anti-leakage)</text>

      <rect x="630" y="20" width="250" height="140" rx="10" fill="{C_SURFACE}" stroke="{C_GRID}" stroke-width="1.5"/>
      <rect x="630" y="20" width="250" height="6" rx="3" fill="{C_SECONDARY}"/>
      <text x="755" y="55" text-anchor="middle" font-size="15" font-weight="700" fill="{C_TEXT}">Modelos</text>
      <text x="755" y="80" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">ML propio: XGBoost</text>
      <text x="755" y="98" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(fase4, TimeSeriesSplit)</text>
      <text x="755" y="120" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">LLM: Gemini</text>
      <text x="755" y="138" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(fase9, explica la señal)</text>

      <rect x="940" y="20" width="230" height="140" rx="10" fill="{C_SURFACE}" stroke="{C_GRID}" stroke-width="1.5"/>
      <rect x="940" y="20" width="230" height="6" rx="3" fill="{C_TERTIARY}"/>
      <text x="1055" y="55" text-anchor="middle" font-size="15" font-weight="700" fill="{C_TEXT}">Tablero</text>
      <text x="1055" y="80" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">Streamlit Community Cloud</text>
      <text x="1055" y="100" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">lee data/*.csv generados</text>
      <text x="1055" y="118" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">por fase8 y fase9</text>
      <text x="1055" y="136" text-anchor="middle" font-size="11.5" fill="{C_TEXT_SECONDARY}">(3 reportes, esta app)</text>
    </svg>
    """
    with st.container(border=True):
        st.markdown(_svg_diagrama, unsafe_allow_html=True)

    st.markdown("#### Qué hace cada script")
    _tabla_scripts = pd.DataFrame(
        [
            ("fase3_recoleccion_datos_ETF.py", "Descarga precios (yfinance) y noticias con sentimiento (Alpha Vantage)."),
            ("fase4_procesamiento_ia_ETF.py", "Limpieza, features, entrena y compara 3 modelos (Logistic Regression, Random Forest, XGBoost) con validación temporal."),
            ("fase5_backtesting_ETF.py", "Backtesting de la estrategia del modelo ganador vs. benchmark (Buy & Hold)."),
            ("fase6_metricas_finales_ETF.py", "Consolida los KPIs finales (Sharpe, Drawdown, IC Spearman, etc.)."),
            ("fase8_senal_actual_ETF.py", "Genera la señal BUY/HOLD/SELL más reciente por ETF."),
            ("fase9_llm_prescriptivo_ETF.py", "Llama a Gemini para redactar la recomendación prescriptiva a partir de la señal."),
            ("dashboard_app.py", "Este tablero: lee los CSV generados arriba y muestra los 3 reportes."),
        ],
        columns=["Script", "Qué hace"],
    )
    st.dataframe(_tabla_scripts, use_container_width=True, hide_index=True)

    st.markdown("#### Actualización automática")
    st.write(
        "Un workflow de GitHub Actions (`.github/workflows/actualizar_datos.yml`) "
        "corre las fases 3 a 9 **una vez al día**, usando las claves de API "
        "guardadas como *Secrets* de GitHub (nunca en el código), y guarda "
        "los resultados de vuelta en el repositorio. Este tablero simplemente "
        "lee esos archivos ya actualizados — no llama a ninguna API por su "
        "cuenta. El historial de corridas es público y verificable en la "
        "pestaña **Actions** del repositorio de GitHub (el estado en vivo se "
        "muestra como insignia junto al título de esta página)."
    )

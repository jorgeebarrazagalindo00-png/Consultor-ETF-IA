# ============================================================================
#  FASE 6 — ANÁLISIS Y MÉTRICAS FINALES   (CRISP-DM: Evaluación, continuación)
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Cubre las 3 tareas de la Fase 6 de su cronograma:
#    Tarea 19 -> Cálculo de los KPIs financieros que faltaban (Sortino, Profit
#                Factor, IC Spearman)
#    Tarea 20 -> Consolidación de TODOS los KPIs (Fase 4 + Fase 5 + Fase 6)
#                en una sola tabla resumen
#    Tarea 21 -> Guardar ese resumen como insumo directo para el informe
#                (Fase 7) y el Dashboard Prescriptivo (Fase 8)
#
#  Requiere:
#    data/backtest_resultados.csv   (lo genera la Fase 5)
#    fase5_backtesting_ETF.py debe estar en ESTA MISMA carpeta: la Fase 6
#    reutiliza directamente sus funciones ya probadas (dividir_train_test,
#    reentrenar_modelo_limpio, generar_senales_y_simular, calcular_kpis) en
#    vez de duplicar ese código. Al importarlas no se vuelve a ejecutar nada
#    de la Fase 5 por su cuenta: solo se reutilizan las funciones.
#
#  INSTALAR ANTES DE CORRER:
#    pip install scipy
#
#  Se ejecuta en su computadora (Anaconda Prompt), igual que las Fases 3, 4 y 5.
# ============================================================================

import os
import numpy as np
import pandas as pd
import joblib

DATA_DIR = "data"
BACKTEST_PATH = os.path.join(DATA_DIR, "backtest_resultados.csv")


# ============================================================================
# TAREA 19 — KPIs FINANCIEROS QUE FALTABAN
# ============================================================================
def calcular_sortino(retornos_diarios):
    """Como el Sharpe, pero solo penaliza la volatilidad de los días malos
    (retornos negativos). Es más justo que el Sharpe cuando a una estrategia
    no le molestan los días muy buenos, solo los muy malos."""
    retornos_diarios = pd.Series(retornos_diarios).dropna()
    media = retornos_diarios.mean()
    negativos = retornos_diarios[retornos_diarios < 0]
    downside_std = negativos.std()
    if downside_std == 0 or pd.isna(downside_std):
        return np.nan
    return (media / downside_std) * np.sqrt(252)


def calcular_profit_factor(retornos_diarios):
    """Cuánto se ganó en los días positivos por cada unidad perdida en los
    días negativos. Profit Factor > 1 significa que las ganancias superan
    a las pérdidas en magnitud total (sin importar cuántos días de cada)."""
    retornos_diarios = pd.Series(retornos_diarios).dropna()
    ganancias = retornos_diarios[retornos_diarios > 0].sum()
    perdidas = retornos_diarios[retornos_diarios < 0].sum()
    if perdidas == 0:
        return np.nan
    return ganancias / abs(perdidas)


def calcular_ic_spearman(test_df):
    """Information Coefficient (correlación de rangos de Spearman) entre la
    posición que tomó el modelo (BUY=+1, HOLD=0, SELL=-1) y el retorno real
    que tuvo el ETF al día siguiente. Mide si la señal, en general, apunta
    en la dirección correcta -independientemente de cómo se hayan encadenado
    las ganancias y pérdidas en la simulación de la Fase 5."""
    from scipy.stats import spearmanr
    datos = test_df.dropna(subset=["retorno_1d", "Posicion"])
    ic, p_valor = spearmanr(datos["Posicion"], datos["retorno_1d"])
    return ic, p_valor


# ============================================================================
# EJECUCIÓN
# ============================================================================
if __name__ == "__main__":
    print("Cargando resultados de la Fase 5 (data/backtest_resultados.csv)...")
    backtest = pd.read_csv(BACKTEST_PATH, index_col=0, parse_dates=True)

    print("\n[Tarea 19] Calculando Sortino Ratio y Profit Factor...")
    sortino_estrategia = calcular_sortino(backtest["retorno_estrategia"])
    sortino_benchmark = calcular_sortino(backtest["retorno_benchmark"])
    pf_estrategia = calcular_profit_factor(backtest["retorno_estrategia"])
    pf_benchmark = calcular_profit_factor(backtest["retorno_benchmark"])

    print("Reconstruyendo las señales detalladas del backtest de la Fase 5 "
          "(mismo train/test, mismo modelo reentrenado) para calcular el IC Spearman...")
    import fase5_backtesting_ETF as f5

    dataset = pd.read_csv(f5.DATASET_PATH, parse_dates=["Date"])
    features = joblib.load(f5.FEATURES_PATH)
    train_df, test_df = f5.dividir_train_test(dataset)
    modelo, scaler, label_encoder, es_xgboost = f5.reentrenar_modelo_limpio(train_df, features)
    test_df = f5.generar_senales_y_simular(modelo, scaler, label_encoder, es_xgboost, test_df, features)
    ic_spearman, p_valor_ic = calcular_ic_spearman(test_df)

    print("\n[Tarea 20] Consolidando TODOS los KPIs (Fase 5 + Fase 6) en una sola tabla...")
    kpis_estrategia, _ = f5.calcular_kpis(backtest["retorno_estrategia"])
    kpis_benchmark, _ = f5.calcular_kpis(backtest["retorno_benchmark"])
    kpis_estrategia["Sortino Ratio (anualizado)"] = sortino_estrategia
    kpis_benchmark["Sortino Ratio (anualizado)"] = sortino_benchmark
    kpis_estrategia["Profit Factor"] = pf_estrategia
    kpis_benchmark["Profit Factor"] = pf_benchmark

    print("\n=== RESUMEN FINAL DE KPIs (Fase 5 + Fase 6) ===")
    print(f"{'KPI':30s} | {'Estrategia (modelo)':20s} | {'Benchmark (Buy&Hold)':20s}")
    for k in kpis_estrategia:
        print(f"{k:30s} | {kpis_estrategia[k]:20.3f} | {kpis_benchmark[k]:20.3f}")

    print(f"\nIC Spearman (señal del modelo vs. retorno real del día siguiente): "
          f"{ic_spearman:.4f}  (p-valor = {p_valor_ic:.4f})")
    if p_valor_ic < 0.05:
        print("  -> Estadísticamente significativo (p < 0.05): la señal SÍ tiene "
              "algo de valor direccional real, aunque sea modesto.")
    else:
        print("  -> NO es estadísticamente significativo (p >= 0.05): no se puede "
              "afirmar con confianza que la señal tenga valor direccional real; "
              "podría deberse al azar.")

    # Guardar tabla consolidada
    resumen = pd.DataFrame({"Estrategia (modelo)": kpis_estrategia, "Benchmark (Buy & Hold)": kpis_benchmark})
    resumen.loc["IC Spearman"] = [ic_spearman, np.nan]
    resumen.loc["IC Spearman (p-valor)"] = [p_valor_ic, np.nan]
    ruta = os.path.join(DATA_DIR, "kpis_finales_ETF.csv")
    resumen.to_csv(ruta)
    print(f"\n✅ Fase 6 completada. Tabla de KPIs finales guardada en {ruta}")
    print("Este archivo es el insumo directo para la Fase 7 (informe) y la Fase 8 (Dashboard Prescriptivo).")

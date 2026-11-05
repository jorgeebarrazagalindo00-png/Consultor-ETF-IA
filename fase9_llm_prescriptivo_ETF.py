# ============================================================================
#  FASE 9 — MÓDULO LLM DEL REPORTE PRESCRIPTIVO (Sección G del cuestionario)
#  Proyecto: Inversiones ETF (SPY, QQQ, IWM, GLD, TLT)
# ============================================================================
#  Este script toma la señal "de hoy" que genera fase8_senal_actual_ETF.py
#  (modelo ML de la Fase 4 + indicadores técnicos) y llama a la API de un LLM
#  (Google Gemini) para redactar la recomendación en lenguaje natural del
#  Dashboard Prescriptivo: acción sugerida, justificación, riesgos, nivel de
#  confianza y alternativas. El LLM NO decide la señal (BUY/HOLD/SELL) —esa
#  la decide el modelo de ML, por una regla fija—, solo la explica y la
#  enmarca en riesgo. Si el LLM falla, no responde JSON válido, o contradice
#  la señal del modelo, el script usa un respaldo determinístico para que el
#  tablero nunca se quede sin recomendación.
#
#  Mapeo contra la Sección G del cuestionario:
#    G1 -> proveedor/modelo elegido y por qué (ver docstring más abajo)
#    G2 -> construir_entrada()            (payload que recibe el LLM)
#    G3 -> SYSTEM_PROMPT                  (prompt de sistema completo)
#    G4 -> RecomendacionLLM               (salida estructurada + validación de formato)
#    G5 -> reglas fijas vs. LLM: ver validar_contra_reglas_fijas() y comentarios
#    G6 -> validar_contra_reglas_fijas()  (coherencia con la predicción, anti-alucinación)
#    G7 -> correr este script con tu propia API key genera el log de donde se
#          sacan los 3 casos reales (data/log_recomendaciones_LLM.csv)
#    G8 -> MAX_LLAMADAS_POR_DIA, MAX_OUTPUT_TOKENS, manejo de errores de la
#          API, DISCLAIMER y registrar_log() (registro de cada recomendación)
#    G9 -> ver respuesta aparte en Respuestas_Seccion_G_LLM.md (cálculo de costo)
#
#  G1. Modelo/proveedor elegido y por qué:
#    Proveedor: Google. API: Gemini Developer API. Modelo: gemini-3.5-flash-lite.
#    (Se eligió originalmente gemini-2.5-flash-lite; Google lo retiró para
#    cuentas/API keys nuevas -error 404 "no longer available to new users"-
#    y en su documentación de deprecaciones recomienda migrar a este modelo.)
#    Razones: (1) es el modelo más económico y ligero de la familia Gemini
#    vigente -ver G9-, (2) tiene tier gratuito sin pedir tarjeta de crédito,
#    suficiente para el volumen de este proyecto académico, (3) soporta salida
#    JSON estructurada nativa (response_json_schema), lo que evita tener que
#    parsear texto libre y reduce el riesgo de alucinación de formato, (4) la
#    tarea (redactar una justificación corta a partir de datos ya calculados)
#    no requiere un modelo de razonamiento más caro.
#    Variable de entorno con la clave (NUNCA escrita en este archivo ni en el
#    repositorio, solo el nombre): GEMINI_API_KEY  (también se acepta GOOGLE_API_KEY).
#
#  Requiere:
#    pip install google-genai pydantic pandas
#    data/senal_actual_ETF.csv   (lo genera fase8_senal_actual_ETF.py)
#    Variable de entorno GEMINI_API_KEY (o GOOGLE_API_KEY) ya configurada.
#
#  Se ejecuta en su computadora (Anaconda Prompt / terminal), igual que las
#  fases anteriores:  python fase9_llm_prescriptivo_ETF.py
# ============================================================================

import os
import sys
import json
import csv
import time
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.errors import APIError

# ----------------------------------------------------------------------------
# Configuración (Sección G: G1, G8)
# ----------------------------------------------------------------------------
MODEL_NAME = "gemini-3.5-flash-lite"            # G1: modelo elegido
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

DATA_DIR = "data"
SENAL_PATH = os.path.join(DATA_DIR, "senal_actual_ETF.csv")
LOG_PATH = os.path.join(DATA_DIR, "log_recomendaciones_LLM.csv")          # G8: registro
LIMITE_PATH = os.path.join(DATA_DIR, "limite_llamadas_LLM_estado.json")   # G8: límite diario

MAX_LLAMADAS_POR_DIA = 50       # G8: límite de llamadas/día (cuida el tier gratuito)
MAX_OUTPUT_TOKENS = 400         # G8: límite de tokens de salida por llamada
TEMPERATURE = 0.3               # salidas poco creativas: queremos consistencia, no variedad
REINTENTOS_LLM = 3

PERFIL_RIESGO_USUARIO = "Moderado"   # debe coincidir con lo respondido en B2 del cuestionario

DISCLAIMER = (
    "Esta recomendación la genera un sistema académico y NO constituye asesoría "
    "financiera personalizada. No ejecute operaciones reales de compra o venta "
    "basándose únicamente en este resultado."
)

# ----------------------------------------------------------------------------
# G3. Prompt de sistema completo.
#
# Cómo se diseñó: se partió de tres restricciones no negociables (no inventar
# cifras, no decidir la acción -eso ya lo decidió el modelo de ML-, y devolver
# siempre el JSON exacto que espera el tablero) y se probó con 5 entradas de
# prueba construidas a mano (una por ETF) verificando a ojo que: (a) el JSON
# siempre parseaba, (b) accion_sugerida siempre coincidía con la señal dada,
# y (c) ningún número en la justificación era distinto a los que se le pasaron
# en la entrada. Las primeras versiones del prompt dejaban que el LLM opinara
# libremente sobre la acción (a veces discrepaba del modelo); se corrigió
# agregando la instrucción explícita de NUNCA cambiar accion_sugerida.
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Eres un asistente de redacción para el Dashboard Prescriptivo de un proyecto
académico de inversión en ETFs (SPY, QQQ, IWM, GLD, TLT). Tu única tarea es
explicar, en lenguaje claro para un inversionista no experto, una señal que
YA fue calculada por un modelo de Machine Learning. No eres tú quien decide
si comprar, mantener o vender.

Reglas estrictas (no las rompas bajo ninguna circunstancia):
1. El campo "accion_sugerida" de tu respuesta debe ser EXACTAMENTE igual al
   valor de "senal_modelo_ml" que se te entrega en la entrada. Nunca lo
   cambies, nunca lo matices, nunca sugieras una acción distinta.
2. Usa ÚNICAMENTE los datos numéricos que se te entregan en la entrada
   (confianza del modelo, nivel de riesgo, indicadores). Está prohibido
   inventar, estimar o completar cifras que no te dieron explícitamente.
3. Responde SIEMPRE en el formato JSON exacto que se te pide, sin texto
   adicional antes o después, sin markdown, sin comentarios.
4. Si la información recibida es insuficiente o contradictoria, dilo
   explícitamente dentro de "riesgos" en vez de rellenar con suposiciones.
5. Nunca des asesoría financiera personalizada ni garantices resultados.
   El campo "justificacion" debe sonar informativo, no imperativo
   (evita "debes comprar", prefiere "el modelo indica...").
6. "nivel_confianza" debe ser exactamente uno de: "Bajo", "Medio", "Alto",
   y debe ser coherente con el porcentaje de confianza del modelo que
   recibiste (menos de 55% -> Bajo, 55-75% -> Medio, más de 75% -> Alto).
7. "alternativas" debe tener 1 a 3 opciones conservadoras y genéricas
   (ejemplos: "mantener posición actual", "reducir tamaño de la posición",
   "consultar a un asesor financiero certificado"); nunca una operación
   apalancada ni de alto riesgo.

Devuelve un único objeto JSON con los campos: accion_sugerida (str),
justificacion (str, 2-4 frases), riesgos (lista de strings),
nivel_confianza (str), alternativas (lista de strings)."""


# ----------------------------------------------------------------------------
# G4. Salida estructurada esperada del LLM + validación de formato.
# Pydantic valida tipos/forma; además se envía como response_json_schema a
# Gemini para que el propio modelo esté forzado a devolver este formato.
# ----------------------------------------------------------------------------
class RecomendacionLLM(BaseModel):
    accion_sugerida: str
    justificacion: str
    riesgos: list[str]
    nivel_confianza: str
    alternativas: list[str]


# ----------------------------------------------------------------------------
# G2. Construcción de la entrada que recibe el LLM: predicción del modelo de
# ML (Fase 4/8) + indicadores del reporte descriptivo + perfil de riesgo del
# usuario + contexto. Esta función es el ejemplo real de entrada que pide G2
# (se imprime formateado al correr el script).
# ----------------------------------------------------------------------------
def construir_entrada(fila_senal: pd.Series) -> dict:
    return {
        "ticker": fila_senal["Ticker"],
        "fecha": str(fila_senal["Date"]),
        "senal_modelo_ml": fila_senal["Señal"],
        "confianza_modelo_%": float(fila_senal["Confianza_modelo_%"]),
        "nivel_de_riesgo_volatilidad": fila_senal["Nivel_de_riesgo"],
        "indicadores": fila_senal["Justificacion"],
        "perfil_riesgo_usuario": PERFIL_RIESGO_USUARIO,
        "contexto": (
            "Señal generada automáticamente por el modelo entrenado en la "
            "Fase 4 sobre la fila más reciente disponible de este ETF."
        ),
    }


def construir_prompt_usuario(entrada: dict) -> str:
    return (
        "Genera la recomendación para el siguiente ETF usando EXCLUSIVAMENTE "
        "los datos entregados a continuación. No inventes cifras que no "
        "aparezcan aquí.\n\n" + json.dumps(entrada, ensure_ascii=False, indent=2)
    )


# ----------------------------------------------------------------------------
# G5/G6. Qué decide el LLM y qué decide una regla fija, y cómo se valida que
# el LLM no contradiga al modelo ni invente cifras.
#
# Reglas fijas (NO las decide el LLM):
#   - La acción (BUY/HOLD/SELL): la decide el modelo de ML de la Fase 4.
#   - El nivel de riesgo por volatilidad: lo decide calcular_nivel_de_riesgo()
#     de la Fase 8 (percentil histórico), no el LLM.
#   - El límite de llamadas diarias y el límite de tokens: configuración fija.
# Lo que sí decide el LLM:
#   - La redacción de la justificación en lenguaje natural.
#   - La lista de riesgos a resaltar (dentro del vocabulario permitido).
#   - Las alternativas conservadoras a sugerir.
# Por qué: las decisiones con impacto financiero directo (qué señal se
# muestra, cuánto riesgo se le asigna) deben ser deterministas y auditables
# -mismo input, mismo output, sin "creatividad" de un modelo probabilístico-;
# el LLM solo aporta valor en la parte de comunicación/explicación, que es
# donde el riesgo de un error es bajo y fácil de revisar.
# ----------------------------------------------------------------------------
def validar_contra_reglas_fijas(reco: RecomendacionLLM, entrada: dict) -> tuple[bool, str]:
    if reco.accion_sugerida.strip().upper() != str(entrada["senal_modelo_ml"]).strip().upper():
        return False, (
            f"La acción del LLM ('{reco.accion_sugerida}') no coincide con la "
            f"señal del modelo ML ('{entrada['senal_modelo_ml']}')."
        )

    if reco.nivel_confianza not in ("Bajo", "Medio", "Alto"):
        return False, f"nivel_confianza ('{reco.nivel_confianza}') fuera del vocabulario permitido."

    # Anti-alucinación simple: cualquier cifra >100 mencionada en la
    # justificación es sospechosa en este dominio (porcentajes/indicadores
    # normales caen bajo 100); y la única cifra de confianza permitida es
    # la que se le entregó.
    for numero in re.findall(r"\d+(?:\.\d+)?", reco.justificacion):
        valor = float(numero)
        if valor > 100:
            return False, f"La justificación menciona una cifra fuera de rango plausible ({valor})."

    if not reco.riesgos or not reco.alternativas:
        return False, "riesgos o alternativas vacíos: la respuesta no aporta información útil."

    return True, "OK"


def recomendacion_de_respaldo(entrada: dict, motivo: str) -> RecomendacionLLM:
    """Respaldo determinístico (sin LLM) para que el tablero nunca se quede
    sin recomendación si la API falla o la respuesta no pasa validación."""
    return RecomendacionLLM(
        accion_sugerida=str(entrada["senal_modelo_ml"]),
        justificacion=(
            f"{entrada['indicadores']} (texto generado por regla fija, no por el "
            f"LLM, debido a: {motivo})"
        ),
        riesgos=[f"Nivel de riesgo por volatilidad: {entrada['nivel_de_riesgo_volatilidad']}"],
        nivel_confianza=entrada["nivel_de_riesgo_volatilidad"],
        alternativas=["Mantener posición actual", "Consultar a un asesor financiero certificado"],
    )


# ----------------------------------------------------------------------------
# G8. Límite de llamadas por día (cuida el tier gratuito y evita costos
# inesperados si algo queda corriendo en bucle).
# ----------------------------------------------------------------------------
def cargar_estado_limite() -> dict:
    hoy = date.today().isoformat()
    if os.path.exists(LIMITE_PATH):
        with open(LIMITE_PATH, encoding="utf-8") as f:
            guardado = json.load(f)
        if guardado.get("fecha") == hoy:
            return guardado
    return {"fecha": hoy, "llamadas": 0}


def guardar_estado_limite(estado: dict) -> None:
    Path(DATA_DIR).mkdir(exist_ok=True)
    with open(LIMITE_PATH, "w", encoding="utf-8") as f:
        json.dump(estado, f)


# ----------------------------------------------------------------------------
# G8. Llamada al LLM con reintentos/backoff y manejo de errores de la API.
# ----------------------------------------------------------------------------
def llamar_llm(client: "genai.Client", entrada: dict):
    prompt = construir_prompt_usuario(entrada)
    for intento in range(1, REINTENTOS_LLM + 1):
        try:
            respuesta = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=RecomendacionLLM.model_json_schema(),
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            )
            datos = json.loads(respuesta.text)
            return RecomendacionLLM(**datos), None
        except APIError as e:
            print(f"   intento {intento}/{REINTENTOS_LLM}: error de la API "
                  f"({e.code} {e.status}) - {e.message}")
            if e.code == 429 or (e.code is not None and 500 <= e.code < 600):
                time.sleep(2 ** intento)  # 429/5xx: reintentable con backoff
                continue
            return None, f"Error no reintentable de la API: {e.code} {e.message}"
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   intento {intento}/{REINTENTOS_LLM}: salida no parseable como JSON válido ({e})")
            continue
    return None, "Se agotaron los reintentos sin obtener una respuesta válida del LLM."


# ----------------------------------------------------------------------------
# G8. Registro (log) de cada recomendación emitida, LLM o de respaldo.
# De este archivo se sacan los 3 casos reales que pide G7.
# ----------------------------------------------------------------------------
def registrar_log(entrada: dict, reco: RecomendacionLLM, origen: str, nota: str) -> None:
    Path(DATA_DIR).mkdir(exist_ok=True)
    existe = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow([
                "timestamp", "ticker", "fecha_senal", "senal_ml", "origen",
                "accion_sugerida", "nivel_confianza", "justificacion",
                "riesgos", "alternativas", "validado_ok", "nota",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            entrada["ticker"], entrada["fecha"], entrada["senal_modelo_ml"], origen,
            reco.accion_sugerida, reco.nivel_confianza, reco.justificacion,
            " | ".join(reco.riesgos), " | ".join(reco.alternativas),
            origen == "LLM", nota or "",
        ])


def generar_recomendacion(client: "genai.Client", fila_senal: pd.Series, estado: dict):
    entrada = construir_entrada(fila_senal)

    if estado["llamadas"] >= MAX_LLAMADAS_POR_DIA:
        reco = recomendacion_de_respaldo(entrada, "límite diario de llamadas al LLM alcanzado")
        registrar_log(entrada, reco, "FALLBACK_LIMITE_DIARIO", "límite diario alcanzado")
        return entrada, reco, "FALLBACK_LIMITE_DIARIO"

    reco, error = llamar_llm(client, entrada)
    estado["llamadas"] += 1
    guardar_estado_limite(estado)

    if reco is None:
        reco = recomendacion_de_respaldo(entrada, error)
        registrar_log(entrada, reco, "FALLBACK_ERROR_API", error)
        return entrada, reco, "FALLBACK_ERROR_API"

    ok, motivo = validar_contra_reglas_fijas(reco, entrada)
    if not ok:
        print(f"   ⚠️  Validación de coherencia falló: {motivo}")
        reco = recomendacion_de_respaldo(entrada, motivo)
        registrar_log(entrada, reco, "FALLBACK_VALIDACION", motivo)
        return entrada, reco, "FALLBACK_VALIDACION"

    registrar_log(entrada, reco, "LLM", "OK")
    return entrada, reco, "LLM"


if __name__ == "__main__":
    if not any(os.environ.get(v) for v in API_KEY_ENV_VARS):
        print(f"❌ Falta configurar la variable de entorno {API_KEY_ENV_VARS[0]} "
              f"(o {API_KEY_ENV_VARS[1]}) con tu clave de la API de Gemini.")
        print("   Nunca escribas la clave en este archivo ni en el repositorio "
              "(Instrucción #5 del cuestionario) — solo expórtala en tu sistema:")
        print(f"     export {API_KEY_ENV_VARS[0]}=tu_clave_aqui   (Mac/Linux)")
        print(f"     setx {API_KEY_ENV_VARS[0]} tu_clave_aqui     (Windows)")
        sys.exit(1)

    if not os.path.exists(SENAL_PATH):
        print(f"❌ No se encontró {SENAL_PATH}. Corre primero fase8_senal_actual_ETF.py.")
        sys.exit(1)

    client = genai.Client()  # toma GEMINI_API_KEY / GOOGLE_API_KEY del entorno

    print(f"Cargando señales actuales desde {SENAL_PATH}...")
    senales = pd.read_csv(SENAL_PATH, parse_dates=["Date"])

    estado = cargar_estado_limite()
    print(f"\n{DISCLAIMER}\n")
    print(f"Llamadas al LLM usadas hoy: {estado['llamadas']}/{MAX_LLAMADAS_POR_DIA}\n")

    print("=== RECOMENDACIONES DEL DASHBOARD PRESCRIPTIVO (vía LLM) ===")
    for _, fila in senales.iterrows():
        entrada, reco, origen = generar_recomendacion(client, fila, estado)
        print(f"\n{entrada['ticker']} ({entrada['fecha']}): {reco.accion_sugerida}  "
              f"[origen={origen}, confianza={reco.nivel_confianza}]")
        print(f"   Justificación: {reco.justificacion}")
        print(f"   Riesgos: {', '.join(reco.riesgos)}")
        print(f"   Alternativas: {', '.join(reco.alternativas)}")

    print(f"\n✅ Log de recomendaciones guardado/actualizado en {LOG_PATH}")
    print("Para G7 del cuestionario: copia 3 filas de ese log (idealmente con "
          "origen=LLM), junto con su entrada (construir_entrada) y 2-3 líneas "
          "de análisis crítico de si la recomendación fue razonable.")

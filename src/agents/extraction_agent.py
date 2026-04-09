"""
extraction_agent.py — Agente 2: Extracción de cambios contractuales.

Recibe los textos parseados de ambos documentos más el mapa contextual del
ContextualizationAgent, y produce un ContractChangeOutput validado
con cada cambio (adiciones, eliminaciones, modificaciones) identificado.

Usa LangChain with_structured_output() + Pydantic model_validate() para
aplicación estricta del schema.
"""

import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import ValidationError

from src.models import ContractChangeOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un Auditor Legal Especializado en Análisis de Cambios Contractuales con certificación en compliance y gestión de riesgos.

TU FUNCIÓN EXCLUSIVA:
Utilizando el mapa contextual elaborado por el Analista Senior y los textos completos de ambos documentos, identificar con precisión quirúrgica CADA modificación introducida por la enmienda sobre el contrato original.

TIPOS DE CAMBIOS QUE DEBES DETECTAR:
- ADICIÓN: Contenido presente en la enmienda que no existía en el original (nuevas cláusulas, nuevas condiciones).
- ELIMINACIÓN: Contenido presente en el original que fue removido en la enmienda.
- MODIFICACIÓN: Contenido presente en ambos documentos pero alterado en la enmienda (cambios en plazos, montos, condiciones, partes).

REGLAS DE PRECISIÓN:
1. Usa los identificadores de sección exactos del mapa contextual y de los documentos.
2. Para MODIFICACIONES: indica explícitamente qué decía el original y qué dice la enmienda.
3. No inventes cambios — solo reporta lo que puedes verificar en el texto.
4. Sé exhaustivo: un cambio no detectado puede tener consecuencias legales graves.
5. Para campos numéricos (montos, plazos, porcentajes): indica el valor anterior y el nuevo.
6. Prioriza cambios de alto impacto: indemnizaciones, limitaciones de responsabilidad, plazos, honorarios.

CAMPOS DE SALIDA REQUERIDOS:
- "sections_changed": Lista de identificadores de secciones modificadas (usar identificadores exactos del documento).
- "topics_touched": Categorías legales/comerciales afectadas (ej: "Plazo", "Honorarios", "Terminación", "Confidencialidad", "Propiedad Intelectual").
- "summary_of_the_change": Resumen ejecutivo narrativo detallado de TODOS los cambios. Debe referenciar secciones específicas y para cada cambio indicar: qué cambió, de qué valor, a qué valor.

IMPORTANTE: Tu análisis debe ser tan preciso que un abogado pueda usarlo directamente sin releer los documentos.
"""


class ExtractionAgent:
    """
    Agente 2: Extrae y clasifica todos los cambios entre el original y la enmienda.

    Responsabilidades:
    - Recibir el mapa contextual del ContextualizationAgent
    - Identificar cada adición, eliminación y modificación
    - Producir un ContractChangeOutput validado por Pydantic

    Usa with_structured_output() para validación integrada con el LLM.
    Recurre a model_validate() si el structured output falla.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature, timeout=60)
        self.structured_llm = self.llm.with_structured_output(ContractChangeOutput, include_raw=True)

    def run(
        self,
        original_text: str,
        amendment_text: str,
        context_map: dict,
        parent_trace,
    ) -> ContractChangeOutput:
        """
        Extrae todos los cambios y retorna un ContractChangeOutput validado.

        Args:
            original_text: Texto extraído del contrato original.
            amendment_text: Texto extraído de la enmienda.
            context_map: Dict de contexto estructural del ContextualizationAgent.
            parent_trace: Trace de Langfuse para crear el span hijo.

        Returns:
            Instancia validada de ContractChangeOutput.
        """
        import json as _json

        span = parent_trace.span(
            name="extraction_agent",
            input={
                "original_text_length": len(original_text),
                "amendment_text_length": len(amendment_text),
                "context_map_sections": len(context_map.get("structure_summary", {})),
            },
        )

        start_time = time.time()

        context_str = _json.dumps(context_map, ensure_ascii=False, indent=2)

        human_content = f"""MAPA CONTEXTUAL (elaborado por el Analista Senior):
---
{context_str}
---

CONTRATO ORIGINAL:
---
{original_text}
---

ENMIENDA:
---
{amendment_text}
---

Extrae todos los cambios siguiendo el esquema requerido. Sé exhaustivo y preciso."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        try:
            # Estrategia principal: with_structured_output (validación integrada con LLM)
            raw_output = self.structured_llm.invoke(messages)
            result: ContractChangeOutput = raw_output["parsed"]
            usage = (raw_output["raw"].usage_metadata or {}) if raw_output.get("raw") else {}

            # Re-validación explícita como capa adicional de seguridad
            result = ContractChangeOutput.model_validate(result.model_dump())

            latency_ms = int((time.time() - start_time) * 1000)

            span.end(
                output={
                    "sections_changed": result.sections_changed,
                    "topics_touched": result.topics_touched,
                    "summary_preview": result.summary_of_the_change[:300],
                },
                metadata={
                    "latency_ms": latency_ms,
                    "model": "gpt-4o",
                    "sections_count": len(result.sections_changed),
                    "topics_count": len(result.topics_touched),
                    "prompt_tokens": usage.get("input_tokens"),
                    "completion_tokens": usage.get("output_tokens"),
                    "validation_status": "valid",
                },
            )

            logger.info(
                f"[ExtractionAgent] Extracción completada: "
                f"{len(result.sections_changed)} secciones, "
                f"{len(result.topics_touched)} temas en {latency_ms}ms"
            )
            return result

        except ValidationError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[ExtractionAgent] El output del LLM no cumple el schema ContractChangeOutput: {e}"
            )
            span.end(
                output={"error": str(e)},
                metadata={
                    "latency_ms": latency_ms,
                    "validation_status": "failed",
                },
            )
            raise

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[ExtractionAgent] Error inesperado: {e}")
            span.end(
                output={"error": str(e)},
                metadata={"latency_ms": latency_ms},
            )
            raise

"""
contextualization_agent.py — Agente 1: Contextualización de contratos.

Recibe el texto extraído de ambos documentos (original y enmienda) y
produce un mapa contextual estructurado consumido por el agente extractor.
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.models import ContextMap

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un Analista Senior de Contratos Legales con 15 años de experiencia en derecho corporativo internacional. Tu especialidad es comprender la estructura y el contexto de contratos complejos.

TU RESPONSABILIDAD UNICA:
Cuando se te presenten dos documentos (contrato original y su enmienda), debes construir un mapa contextual preciso que sirva de base para el analisis de cambios posterior.

TU SALIDA DEBE INCLUIR un JSON con exactamente estos campos:
- "document_type": tipo de contrato (ej: "Contrato de Licencia de Software", "Contrato de Servicios SaaS")
- "parties": lista de las partes involucradas con sus roles (ej: ["TechNova S.A. (Licenciante)", "DataBridge Soluciones S.R.L. (Licenciatario)"])
- "contract_date": fecha del contrato original
- "general_purpose": descripcion del proposito general del acuerdo en 2-3 oraciones
- "structure_summary": objeto JSON que mapea cada seccion/clausula del original a la enmienda con el formato:
  {"Clausula 1": "presente en ambos", "Clausula 7": "nueva en enmienda", "Clausula X": "eliminada en enmienda"}

RESTRICCIONES CRITICAS:
- NO extraigas cambios de contenido especificos, eso lo hace el agente de extraccion.
- NO compares textos clausula por clausula.
- Responde SIEMPRE en formato JSON valido. Sin markdown, sin texto adicional.
- Si una seccion existe en la enmienda pero no en el original, marcala como "nueva en enmienda".
- Si una seccion existe en el original pero no en la enmienda, marcala como "eliminada en enmienda".

IMPORTANTE: El agente de extraccion downstream depende de tu mapa contextual para ser preciso. Se exhaustivo.
"""


class ContextualizationAgent:
    """Construye un mapa contextual estructural a partir de ambos textos."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(ContextMap, include_raw=True)

    def run(
        self,
        original_text: str,
        amendment_text: str,
        parent_trace,
    ) -> dict:
        span = parent_trace.span(
            name="contextualization_agent",
            input={
                "original_text": original_text,
                "amendment_text": amendment_text,
                "original_text_length": len(original_text),
                "amendment_text_length": len(amendment_text),
            },
        )

        start_time = time.time()
        human_content = f"""CONTRATO ORIGINAL:
---
{original_text}
---

ENMIENDA:
---
{amendment_text}
---

Construye el mapa contextual en formato JSON."""

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        try:
            response = self.structured_llm.invoke(messages)
            parsed = response.get("parsed")
            raw_response = response.get("raw")
            usage = (raw_response.usage_metadata or {}) if raw_response else {}
            latency_ms = int((time.time() - start_time) * 1000)

            if parsed is None:
                parsing_error = response.get("parsing_error")
                raise ValueError(
                    "El agente de contextualizacion no devolvio un JSON valido "
                    f"con el schema esperado. Detalle: {parsing_error}"
                )

            if isinstance(parsed, ContextMap):
                context_map = parsed.model_dump()
            elif isinstance(parsed, dict):
                context_map = ContextMap.model_validate(parsed).model_dump()
            else:
                raise TypeError(
                    "Tipo inesperado en la salida estructurada del agente de contextualizacion: "
                    f"{type(parsed).__name__}"
                )
            span.end(
                output={
                    "document_type": context_map.get("document_type"),
                    "parties_count": len(context_map.get("parties", [])),
                    "sections_mapped": len(context_map.get("structure_summary", {})),
                    "context_map": context_map,
                },
                metadata={
                    "latency_ms": latency_ms,
                    "model": "gpt-4o",
                    "prompt_tokens": usage.get("input_tokens") if usage else None,
                    "completion_tokens": usage.get("output_tokens") if usage else None,
                    "validation_status": "valid",
                },
            )

            if not context_map.get("structure_summary"):
                logger.warning(
                    "[ContextualizationAgent] El modelo no devolvio structure_summary; se usara un mapa vacio"
                )

            logger.info(
                "[ContextualizationAgent] Mapa construido: %s - %s secciones en %sms",
                context_map.get("document_type"),
                len(context_map.get("structure_summary", {})),
                latency_ms,
            )
            return context_map

        except ValidationError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("[ContextualizationAgent] El output no cumple el schema ContextMap: %s", e)
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
            logger.error("[ContextualizationAgent] Error inesperado: %s", e)
            span.end(
                output={"error": str(e)},
                metadata={
                    "latency_ms": latency_ms,
                    "validation_status": "failed",
                },
            )
            raise

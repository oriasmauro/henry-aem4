"""
contextualization_agent.py — Agente 1: Contextualización de contratos.

Recibe el texto extraído de ambos documentos (original y enmienda) y
produce un mapa contextual estructurado (dict JSON) que describe: tipo de
documento, partes, fecha, propósito general y resumen de la estructura de cláusulas.

Este agente NO extrae cambios — solo construye contexto para el Agente 2.
"""

import json
import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un Analista Senior de Contratos Legales con 15 años de experiencia en derecho corporativo internacional. Tu especialidad es comprender la estructura y el contexto de contratos complejos.

TU RESPONSABILIDAD ÚNICA:
Cuando se te presenten dos documentos (contrato original y su enmienda), debes construir un mapa contextual preciso que sirva de base para el análisis de cambios posterior.

TU SALIDA DEBE INCLUIR un JSON con exactamente estos campos:
- "document_type": tipo de contrato (ej: "Contrato de Licencia de Software", "Contrato de Servicios SaaS")
- "parties": lista de las partes involucradas con sus roles (ej: ["TechNova S.A. (Licenciante)", "DataBridge Soluciones S.R.L. (Licenciatario)"])
- "contract_date": fecha del contrato original
- "general_purpose": descripción del propósito general del acuerdo en 2-3 oraciones
- "structure_summary": objeto JSON que mapea cada sección/cláusula del original a la enmienda con el formato:
  {"Cláusula 1": "presente en ambos", "Cláusula 7": "nueva en enmienda", "Cláusula X": "eliminada en enmienda"}

RESTRICCIONES CRÍTICAS:
- NO extraigas cambios de contenido específicos — eso lo hace el agente de extracción.
- NO compares textos cláusula por cláusula.
- Responde SIEMPRE en formato JSON válido. Sin markdown, sin texto adicional.
- Si una sección existe en la enmienda pero no en el original, márcala como "nueva en enmienda".
- Si una sección existe en el original pero no en la enmienda, márcala como "eliminada en enmienda".

IMPORTANTE: El agente de extracción downstream depende de tu mapa contextual para ser preciso. Sé exhaustivo.
"""


class ContextualizationAgent:
    """
    Agente 1: Construye un mapa contextual estructural a partir de ambos textos contractuales.

    Responsabilidades:
    - Identificar tipo de documento, partes, fecha y propósito
    - Mapear la correspondencia de secciones entre original y enmienda
    - Producir un JSON estructurado consumido por ExtractionAgent

    NO hace: extraer cambios específicos, comparar contenido de cláusulas.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        self.llm = ChatOpenAI(model=model, temperature=temperature, timeout=60)

    def run(
        self,
        original_text: str,
        amendment_text: str,
        parent_trace,
    ) -> dict:
        """
        Construye el mapa contextual a partir de ambos textos del documento.

        Args:
            original_text: Texto extraído del contrato original.
            amendment_text: Texto extraído de la enmienda.
            parent_trace: Trace de Langfuse para crear el span hijo.

        Returns:
            dict con claves: document_type, parties, contract_date,
                             general_purpose, structure_summary
        """
        span = parent_trace.span(
            name="contextualization_agent",
            input={
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
            response = self.llm.invoke(messages)
            raw_content = response.content.strip()
            latency_ms = int((time.time() - start_time) * 1000)

            # Eliminar markdown code fences si están presentes
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                raw_content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                ).strip()

            context_map = json.loads(raw_content)

            usage = response.usage_metadata if hasattr(response, "usage_metadata") else {}

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
                },
            )

            logger.info(
                f"[ContextualizationAgent] Mapa construido: {context_map.get('document_type')} — "
                f"{len(context_map.get('structure_summary', {}))} secciones mapeadas en {latency_ms}ms"
            )
            return context_map

        except json.JSONDecodeError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[ContextualizationAgent] JSON inválido en respuesta: {e}")
            logger.error(f"Respuesta cruda: {response.content[:500] if 'response' in dir() else 'N/A'}")

            # Fallback: retornar contexto mínimo para que el pipeline pueda continuar
            fallback = {
                "document_type": "Tipo no identificado",
                "parties": [],
                "contract_date": "No identificada",
                "general_purpose": "No se pudo extraer el contexto automáticamente.",
                "structure_summary": {},
                "_parsing_error": str(e),
            }

            span.end(
                output={"error": str(e), "fallback_used": True},
                metadata={"latency_ms": latency_ms},
            )
            return fallback

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[ContextualizationAgent] Error inesperado: {e}")
            span.end(
                output={"error": str(e)},
                metadata={"latency_ms": latency_ms},
            )
            raise

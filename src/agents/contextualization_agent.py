"""
contextualization_agent.py — Agent 1: Contract Contextualization.

Receives the extracted text from both documents (original + amendment) and
produces a structured context map (JSON dict) describing: document type,
parties, date, general purpose, and clause structure summary.

This agent does NOT extract changes — it only builds context for Agent 2.
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
    Agent 1: Builds a structural context map from both contract texts.

    Responsibilities:
    - Identify document type, parties, date, purpose
    - Map section correspondence between original and amendment
    - Produce structured JSON consumed by ExtractionAgent

    Does NOT: extract specific changes, compare clause content.
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
        Build context map from both document texts.

        Args:
            original_text: Extracted text from the original contract.
            amendment_text: Extracted text from the amendment.
            parent_trace: Langfuse trace to create child span under.

        Returns:
            dict with keys: document_type, parties, contract_date,
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

            # Strip markdown code fences if present
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
                },
                metadata={
                    "latency_ms": latency_ms,
                    "model": "gpt-4o",
                    "input_tokens": usage.get("input_tokens") if usage else None,
                    "output_tokens": usage.get("output_tokens") if usage else None,
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

            # Fallback: return minimal context so pipeline can continue
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

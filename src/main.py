"""
main.py — LegalMove: Autonomous Contract Comparison Pipeline.

Usage:
    python src/main.py <original_image_path> <amendment_image_path>

Example:
    python src/main.py data/test_contracts/documento_1__original.jpg \\
                       data/test_contracts/documento_1__enmienda.jpg
"""

import sys
import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Validate environment variables before any import that needs them ──────────
_REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]

def _validate_env() -> None:
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Faltan variables de entorno requeridas: {', '.join(missing)}\n"
            "Copia .env.example a .env y completa los valores."
        )

_validate_env()

# ── Imports after env validation ──────────────────────────────────────────────
from openai import OpenAI
from langfuse import Langfuse

from src.image_parser import parse_contract_image
from src.agents import ContextualizationAgent, ExtractionAgent
from src.models import ContractChangeOutput

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legalmove")


def run_pipeline(original_image_path: str, amendment_image_path: str) -> ContractChangeOutput:
    """
    Execute the full LegalMove contract comparison pipeline.

    Pipeline stages:
        1. parse_original_contract  — GPT-4o Vision on original image
        2. parse_amendment_contract — GPT-4o Vision on amendment image
        3. contextualization_agent  — Build structural context map
        4. extraction_agent         — Extract + validate all changes

    All stages emit child Langfuse spans under the root "contract-analysis" trace.

    Args:
        original_image_path: Path to the original contract image.
        amendment_image_path: Path to the amendment image.

    Returns:
        Validated ContractChangeOutput with sections, topics, and summary.
    """
    # ── Clients ───────────────────────────────────────────────────────────────
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    # ── Root trace ────────────────────────────────────────────────────────────
    trace = langfuse_client.trace(
        name="contract-analysis",
        metadata={
            "original_image": original_image_path,
            "amendment_image": amendment_image_path,
            "pipeline_version": "1.0.0",
        },
    )

    logger.info("=" * 60)
    logger.info("LegalMove — Análisis de Contratos")
    logger.info(f"  Original : {original_image_path}")
    logger.info(f"  Enmienda : {amendment_image_path}")
    logger.info("=" * 60)

    # ── Paso 1: Parse original contract ───────────────────────────────────────
    logger.info("[1/4] Parseando contrato original...")
    original_text = parse_contract_image(
        image_path=original_image_path,
        openai_client=openai_client,
        langfuse_client=langfuse_client,
        parent_trace=trace,
        span_name="parse_original_contract",
    )
    logger.info(f"      Extraídos {len(original_text)} caracteres.")

    # ── Paso 2: Parse amendment ───────────────────────────────────────────────
    logger.info("[2/4] Parseando enmienda...")
    amendment_text = parse_contract_image(
        image_path=amendment_image_path,
        openai_client=openai_client,
        langfuse_client=langfuse_client,
        parent_trace=trace,
        span_name="parse_amendment_contract",
    )
    logger.info(f"      Extraídos {len(amendment_text)} caracteres.")

    # ── Paso 3: Contextualization Agent ───────────────────────────────────────
    logger.info("[3/4] Construyendo mapa contextual (Agente 1)...")
    context_agent = ContextualizationAgent(model="gpt-4o")
    context_map = context_agent.run(
        original_text=original_text,
        amendment_text=amendment_text,
        parent_trace=trace,
    )
    logger.info(
        f"      Contexto: {context_map.get('document_type', 'N/A')} — "
        f"{len(context_map.get('structure_summary', {}))} secciones mapeadas."
    )

    # ── Paso 4: Extraction Agent ──────────────────────────────────────────────
    logger.info("[4/4] Extrayendo cambios (Agente 2)...")
    extraction_agent = ExtractionAgent(model="gpt-4o")
    result: ContractChangeOutput = extraction_agent.run(
        original_text=original_text,
        amendment_text=amendment_text,
        context_map=context_map,
        parent_trace=trace,
    )
    logger.info(
        f"      Detectados {len(result.sections_changed)} secciones modificadas, "
        f"{len(result.topics_touched)} temas afectados."
    )

    # ── Finalize trace ────────────────────────────────────────────────────────
    trace.update(
        output=result.model_dump(),
        metadata={
            "sections_changed_count": len(result.sections_changed),
            "topics_touched_count": len(result.topics_touched),
            "document_type": context_map.get("document_type"),
        },
    )
    langfuse_client.flush()
    logger.info("Traza Langfuse enviada.")

    return result


def _print_result(result: ContractChangeOutput) -> None:
    """Print the final result as formatted JSON to stdout."""
    print("\n" + "=" * 60)
    print("RESULTADO DEL ANÁLISIS")
    print("=" * 60)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print("=" * 60)


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python src/main.py <original_image> <amendment_image>")
        print()
        print("Ejemplos:")
        print("  python src/main.py data/test_contracts/documento_1__original.jpg \\")
        print("                     data/test_contracts/documento_1__enmienda.jpg")
        sys.exit(1)

    original_path = sys.argv[1]
    amendment_path = sys.argv[2]

    # Validate paths
    for path in [original_path, amendment_path]:
        if not Path(path).exists():
            print(f"Error: archivo no encontrado: {path}")
            sys.exit(1)

    result = run_pipeline(original_path, amendment_path)
    _print_result(result)


if __name__ == "__main__":
    main()

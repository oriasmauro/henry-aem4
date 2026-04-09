"""
main.py — LegalMove: Pipeline autónomo de comparación de contratos.

Uso:
    python src/main.py <ruta_imagen_original> <ruta_imagen_enmienda>

Ejemplo:
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

# ── Validar variables de entorno antes de cualquier import que las requiera ───
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

# ── Imports después de validar el entorno ─────────────────────────────────────
from openai import OpenAI
from langfuse import Langfuse

from src.image_parser import parse_contract_image
from src.agents import ContextualizationAgent, ExtractionAgent
from src.models import ContractChangeOutput

# ── Configuración de logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legalmove")


def run_pipeline(original_image_path: str, amendment_image_path: str) -> ContractChangeOutput:
    """
    Ejecuta el pipeline completo de comparación de contratos LegalMove.

    Etapas del pipeline:
        1. parse_original_contract  — GPT-4o Vision sobre imagen original
        2. parse_amendment_contract — GPT-4o Vision sobre imagen de enmienda
        3. contextualization_agent  — Construye mapa contextual estructural
        4. extraction_agent         — Extrae y valida todos los cambios

    Todas las etapas emiten spans hijos de Langfuse bajo el trace raíz "contract-analysis".

    Args:
        original_image_path: Path a la imagen del contrato original.
        amendment_image_path: Path a la imagen de la enmienda.

    Returns:
        ContractChangeOutput validado con secciones, temas y resumen.
    """
    # ── Clientes ──────────────────────────────────────────────────────────────
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    # ── Trace root ────────────────────────────────────────────────────────────
    trace = langfuse_client.trace(
        name="contract-analysis",
        metadata={
            "original_image": original_image_path,
            "amendment_image": amendment_image_path,
            "pipeline_version": "1.0.0",
        },
    )

    original_size_kb = round(Path(original_image_path).stat().st_size / 1024, 1)
    amendment_size_kb = round(Path(amendment_image_path).stat().st_size / 1024, 1)

    logger.info("=" * 60)
    logger.info("LegalMove — Análisis de Contratos")
    logger.info(f"  trace_id : {trace.id}")
    logger.info(f"  Original : {original_image_path} ({original_size_kb} KB)")
    logger.info(f"  Enmienda : {amendment_image_path} ({amendment_size_kb} KB)")
    logger.info("=" * 60)

    # ── Paso 1: Parsear contrato original ─────────────────────────────────────
    logger.info("[1/4] Parsing contrato original (GPT-4o Vision)...")
    original_text = parse_contract_image(
        image_path=original_image_path,
        openai_client=openai_client,
        langfuse_client=langfuse_client,
        parent_trace=trace,
        span_name="parse_original_contract",
    )
    logger.info(f"      OK — {len(original_text)} caracteres extraídos")

    # ── Paso 2: Parsear enmienda ───────────────────────────────────────────────
    logger.info("[2/4] Parsing enmienda (GPT-4o Vision)...")
    amendment_text = parse_contract_image(
        image_path=amendment_image_path,
        openai_client=openai_client,
        langfuse_client=langfuse_client,
        parent_trace=trace,
        span_name="parse_amendment_contract",
    )
    logger.info(f"      OK — {len(amendment_text)} caracteres extraídos")

    # ── Paso 3: Agente de contextualización ───────────────────────────────────
    logger.info("[3/4] Agente 1 — Construyendo mapa contextual...")
    context_agent = ContextualizationAgent(model="gpt-4o")
    context_map = context_agent.run(
        original_text=original_text,
        amendment_text=amendment_text,
        parent_trace=trace,
    )
    logger.info(f"      OK — Tipo: {context_map.get('document_type', 'N/A')}")
    logger.info(f"           Partes: {', '.join(context_map.get('parties', [])) or 'N/A'}")
    logger.info(f"           Secciones mapeadas: {len(context_map.get('structure_summary', {}))}")

    # ── Paso 4: Agente de extracción ──────────────────────────────────────────
    logger.info("[4/4] Agente 2 — Extrayendo y validando cambios...")
    extraction_agent = ExtractionAgent(model="gpt-4o")
    try:
        result: ContractChangeOutput = extraction_agent.run(
            original_text=original_text,
            amendment_text=amendment_text,
            context_map=context_map,
            parent_trace=trace,
        )
        if not result.sections_changed and not result.topics_touched:
            logger.info("      OK — Sin cambios detectados entre los documentos")
        else:
            logger.info(f"      OK — {len(result.sections_changed)} secciones modificadas: {', '.join(result.sections_changed)}")
            logger.info(f"           {len(result.topics_touched)} temas afectados: {', '.join(result.topics_touched)}")
        trace.update(
            output=result.model_dump(),
            metadata={
                "sections_changed_count": len(result.sections_changed),
                "topics_touched_count": len(result.topics_touched),
                "document_type": context_map.get("document_type"),
            },
        )
        return result
    except Exception as e:
        logger.error(f"      ERROR — {e}")
        trace.update(metadata={"error": str(e)})
        raise
    finally:
        langfuse_client.flush()
        logger.info(f"Traza Langfuse enviada — https://cloud.langfuse.com/trace/{trace.id}")


def _print_result(result: ContractChangeOutput) -> None:
    """Imprime el resultado final formateado en stdout."""
    print("\n" + "=" * 60)
    print("RESULTADO DEL ANÁLISIS")
    print("=" * 60)

    print("\nSecciones modificadas:")
    for s in result.sections_changed:
        print(f"  • {s}")

    print("\nTemas afectados:")
    for t in result.topics_touched:
        print(f"  • {t}")

    print("\nResumen de cambios:")
    print("-" * 60)
    print(result.summary_of_the_change)
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

    # Validar paths de entrada
    for path in [original_path, amendment_path]:
        if not Path(path).exists():
            print(f"Error: archivo no encontrado: {path}")
            sys.exit(1)

    result = run_pipeline(original_path, amendment_path)
    _print_result(result)


if __name__ == "__main__":
    main()

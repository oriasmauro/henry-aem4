"""
main.py — LegalMove: Pipeline autónomo de comparación de contratos.

Uso:
    python -m src.main <ruta_imagen_original> <ruta_imagen_enmienda>
    python -m src.main <ruta_imagen_original> <ruta_imagen_enmienda> --pretty
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


def _validate_env() -> None:
    missing = [var for var in _REQUIRED_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(
            f"Faltan variables de entorno requeridas: {', '.join(missing)}\n"
            "Copia .env.example a .env y completa los valores."
        )

from langfuse import Langfuse
from openai import OpenAI

from src.agents import ContextualizationAgent, ExtractionAgent
from src.image_parser import parse_contract_image
from src.models import ContractChangeOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("legalmove")


def run_pipeline(original_image_path: str, amendment_image_path: str) -> ContractChangeOutput:
    """Ejecuta el pipeline completo y devuelve un output validado por Pydantic."""
    _validate_env()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    trace = langfuse_client.trace(
        name="contract-analysis",
        metadata={
            "original_image": original_image_path,
            "amendment_image": amendment_image_path,
            "pipeline_version": "1.1.0",
            "output_mode": "json",
        },
    )

    pipeline_start = time.time()
    original_size_kb = round(Path(original_image_path).stat().st_size / 1024, 1)
    amendment_size_kb = round(Path(amendment_image_path).stat().st_size / 1024, 1)

    logger.info("=" * 60)
    logger.info("LegalMove - Analisis de Contratos")
    logger.info("  trace_id : %s", trace.id)
    logger.info("  Original : %s (%s KB)", original_image_path, original_size_kb)
    logger.info("  Enmienda : %s (%s KB)", amendment_image_path, amendment_size_kb)
    logger.info("=" * 60)

    try:
        logger.info("[1/4] Parsing contrato original (GPT-4o Vision)...")
        original_text = parse_contract_image(
            image_path=original_image_path,
            openai_client=openai_client,
            langfuse_client=langfuse_client,
            parent_trace=trace,
            span_name="parse_original_contract",
        )
        logger.info("      OK - %s caracteres extraidos", len(original_text))

        logger.info("[2/4] Parsing enmienda (GPT-4o Vision)...")
        amendment_text = parse_contract_image(
            image_path=amendment_image_path,
            openai_client=openai_client,
            langfuse_client=langfuse_client,
            parent_trace=trace,
            span_name="parse_amendment_contract",
        )
        logger.info("      OK - %s caracteres extraidos", len(amendment_text))

        logger.info("[3/4] Agente 1 - Construyendo mapa contextual...")
        context_agent = ContextualizationAgent(model="gpt-4o")
        context_map = context_agent.run(
            original_text=original_text,
            amendment_text=amendment_text,
            parent_trace=trace,
        )
        logger.info("      OK - Tipo: %s", context_map.get("document_type", "N/A"))
        logger.info("           Fecha: %s", context_map.get("contract_date", "N/A"))
        logger.info(
            "           Partes: %s",
            ", ".join(context_map.get("parties", [])) or "N/A",
        )
        logger.info(
            "           Secciones mapeadas: %s",
            len(context_map.get("structure_summary", {})),
        )

        logger.info("[4/4] Agente 2 - Extrayendo y validando cambios...")
        extraction_agent = ExtractionAgent(model="gpt-4o")
        result = extraction_agent.run(
            original_text=original_text,
            amendment_text=amendment_text,
            context_map=context_map,
            parent_trace=trace,
        )

        trace.update(
            output=result.model_dump(),
            metadata={
                "status": "success",
                "sections_changed_count": len(result.sections_changed),
                "topics_touched_count": len(result.topics_touched),
                "document_type": context_map.get("document_type"),
            },
        )
        return result
    except Exception as exc:
        logger.error("Pipeline fallido: %s", exc)
        trace.update(
            output={"error": str(exc)},
            metadata={"status": "failed", "error": str(exc)},
        )
        raise
    finally:
        langfuse_client.flush() # Asegura que todos los eventos se envíen a Langfuse antes de cerrar la aplicación
        elapsed = round(time.time() - pipeline_start, 1)
        logger.info(
            "Pipeline completado en %ss - Traza: https://cloud.langfuse.com/trace/%s",
            elapsed,
            trace.id,
        )


def _render_pretty(result: ContractChangeOutput) -> str:
    lines = [
        "=" * 60,
        "RESULTADO DEL ANALISIS",
        "=" * 60,
        "",
        "Secciones modificadas:",
    ]
    if result.sections_changed:
        lines.extend(f"  - {section}" for section in result.sections_changed)
    else:
        lines.append("  - No se detectaron cambios")

    lines.extend(["", "Temas afectados:"])
    if result.topics_touched:
        lines.extend(f"  - {topic}" for topic in result.topics_touched)
    else:
        lines.append("  - No se detectaron temas afectados")

    lines.extend(
        [
            "",
            "Resumen de cambios:",
            "-" * 60,
            result.summary_of_the_change,
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara un contrato original con su enmienda y devuelve JSON validado."
    )
    parser.add_argument("original_image", help="Ruta a la imagen del contrato original.")
    parser.add_argument("amendment_image", help="Ruta a la imagen de la enmienda.")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime una vista legible para humanos en lugar del JSON estructurado.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    for path in [args.original_image, args.amendment_image]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

    result = run_pipeline(args.original_image, args.amendment_image)
    if args.pretty:
        print(_render_pretty(result))
    else:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

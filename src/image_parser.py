"""
image_parser.py — Multimodal parsing of contract images via GPT-4o Vision.

Each call extracts the complete structured text from one image and registers
a child Langfuse span with input/output/latency/token metadata.
"""

import base64
import time
import logging
from pathlib import Path

from openai import OpenAI, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

PARSING_SYSTEM_PROMPT = """Eres un parser de documentos legales de precisión quirúrgica especializado en contratos y enmiendas.

Tu tarea es extraer el TEXTO COMPLETO del documento que se te presenta en la imagen, preservando su estructura exacta.

Reglas obligatorias:
1. Preserva TODOS los números de sección, identificadores de cláusula y numeración jerárquica exactamente como aparecen (ej: "1.", "Cláusula 3", "Sección 2.1(a)").
2. Mantén la separación entre párrafos y secciones usando líneas en blanco.
3. Reproduce títulos y subtítulos con su jerarquía original.
4. Incluye TODO el texto — no resumas, no omitas, no parafrasees ningún contenido.
5. Si alguna parte del texto es ilegible, indícalo con [ILEGIBLE] en lugar de adivinar.
6. Preserva viñetas, listas numeradas y estructura de sangría tal como aparecen.
7. Para términos definidos (en mayúsculas o entre comillas), preserva el formato exacto.
8. Conserva las referencias cruzadas exactas (ej: "según lo definido en la Cláusula 2").
9. Incluye las partes del contrato, fechas y datos de firma si son visibles.
10. Produce ÚNICAMENTE el texto extraído. Sin comentarios, sin metadatos, sin envolturas.
"""


def _encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and return (b64_string, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Formato de imagen no soportado: {path.suffix}")

    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    media_type = mime_map[path.suffix.lower()]
    return b64, media_type


def parse_contract_image(
    image_path: str,
    openai_client: OpenAI,
    langfuse_client,
    parent_trace,
    span_name: str = "parse_contract_image",
    max_retries: int = 3,
) -> str:
    """
    Parse a contract image using GPT-4o Vision and return the extracted text.

    Args:
        image_path: Path to the JPEG/PNG contract image.
        openai_client: Initialized OpenAI client.
        langfuse_client: Initialized Langfuse client (unused directly; tracing via parent_trace).
        parent_trace: Langfuse trace object to create child spans under.
        span_name: Name for this Langfuse span.
        max_retries: Number of retry attempts on transient API errors.

    Returns:
        Extracted text string preserving document structure.
    """
    span = parent_trace.span(
        name=span_name,
        input={"image_path": image_path},
    )

    start_time = time.time()
    last_error = None

    for attempt in range(max_retries):
        try:
            b64, media_type = _encode_image(image_path)

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": PARSING_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extrae el texto completo de este documento contractual.",
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=4096,
                timeout=60,
            )

            extracted_text = response.choices[0].message.content
            latency_ms = int((time.time() - start_time) * 1000)
            usage = response.usage

            span.end(
                output={
                    "text_length": len(extracted_text),
                    "text_preview": extracted_text[:200],
                },
                metadata={
                    "latency_ms": latency_ms,
                    "model": "gpt-4o",
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                    "attempt": attempt + 1,
                },
            )

            logger.info(f"[{span_name}] Parsed {len(extracted_text)} chars in {latency_ms}ms")
            return extracted_text

        except (RateLimitError, APITimeoutError) as e:
            last_error = e
            delay = 2.0 * (2 ** attempt)
            logger.warning(
                f"[{span_name}] API error on attempt {attempt + 1}/{max_retries}: {e}. "
                f"Retrying in {delay}s..."
            )
            if attempt < max_retries - 1:
                time.sleep(delay)

        except FileNotFoundError as e:
            span.end(
                output={"error": str(e)},
                metadata={"latency_ms": int((time.time() - start_time) * 1000)},
            )
            raise

        except Exception as e:
            logger.error(f"[{span_name}] Unexpected error: {e}")
            span.end(
                output={"error": str(e)},
                metadata={"latency_ms": int((time.time() - start_time) * 1000)},
            )
            raise

    # All retries exhausted
    span.end(
        output={"error": str(last_error)},
        metadata={"latency_ms": int((time.time() - start_time) * 1000)},
    )
    raise RuntimeError(
        f"[{span_name}] Falló después de {max_retries} intentos: {last_error}"
    )

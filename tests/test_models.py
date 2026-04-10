import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from src.image_parser import parse_contract_image
from src.models import ContextMap, ContractChangeOutput


class DummySpan:
    def __init__(self):
        self.ended = False
        self.output = None
        self.metadata = None

    def end(self, output=None, metadata=None):
        self.ended = True
        self.output = output
        self.metadata = metadata


class DummyTrace:
    def __init__(self):
        self.last_span = None

    def span(self, name, input):
        self.last_span = DummySpan()
        return self.last_span


class ContractModelsTestCase(unittest.TestCase):
    def test_contract_change_output_accepts_valid_payload(self) -> None:
        payload = {
            "sections_changed": ["Cláusula 2", "Cláusula 7"],
            "topics_touched": ["Plazo", "Protección de datos"],
            "summary_of_the_change": (
                "La Cláusula 2 modifica el plazo contractual de 12 a 24 meses "
                "y la Cláusula 7 agrega obligaciones específicas de protección de datos."
            ),
        }

        result = ContractChangeOutput.model_validate(payload)

        self.assertEqual(result.sections_changed, ["Cláusula 2", "Cláusula 7"])
        self.assertIn("Plazo", result.topics_touched)

    def test_contract_change_output_rejects_short_summary(self) -> None:
        with self.assertRaises(ValidationError):
            ContractChangeOutput.model_validate(
                {
                    "sections_changed": ["Cláusula 2"],
                    "topics_touched": ["Plazo"],
                    "summary_of_the_change": "Cambio corto.",
                }
            )

    def test_context_map_rejects_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ContextMap.model_validate(
                {
                    "document_type": "Contrato de Servicios",
                    "parties": ["LegalMove", "Cliente"],
                    "contract_date": "2026-01-01",
                    "general_purpose": "Regular la prestación de servicios.",
                    "structure_summary": {"Cláusula 1": "presente en ambos"},
                    "unexpected": True,
                }
            )

    def test_context_map_marks_degraded_when_structure_summary_missing(self) -> None:
        result = ContextMap.model_validate(
            {
                "document_type": "Contrato de Servicios",
                "parties": ["LegalMove", "Cliente"],
                "contract_date": "2026-01-01",
                "general_purpose": "Regular la prestación de servicios.",
            }
        )

        self.assertEqual(result.structure_summary, {})
        self.assertTrue(result.is_degraded)


class ImageParserTestCase(unittest.TestCase):
    def test_parse_contract_image_detects_truncation(self) -> None:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(b"fake-image-content")
            image_path = tmp.name

        trace = DummyTrace()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="texto truncado"),
                    finish_reason="length",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: response
                )
            )
        )

        try:
            with patch("src.image_parser.time.sleep", return_value=None):
                with self.assertRaises(RuntimeError):
                    parse_contract_image(
                        image_path=image_path,
                        openai_client=client,
                        langfuse_client=None,
                        parent_trace=trace,
                        span_name="parse_original_contract",
                        max_retries=1,
                    )
        finally:
            Path(image_path).unlink(missing_ok=True)

        self.assertTrue(trace.last_span.ended)
        self.assertIn("truncada", trace.last_span.output["error"])


if __name__ == "__main__":
    unittest.main()

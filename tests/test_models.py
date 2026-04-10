import unittest

from pydantic import ValidationError

from src.models import ContextMap, ContractChangeOutput


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


if __name__ == "__main__":
    unittest.main()

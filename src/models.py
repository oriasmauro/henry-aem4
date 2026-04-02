from pydantic import BaseModel, Field, field_validator


class ContractChangeOutput(BaseModel):
    """Validated output schema for contract change analysis."""

    sections_changed: list[str] = Field(
        description=(
            "List of section/clause identifiers that were modified, added, or removed. "
            "Use the exact identifiers from the document (e.g., 'Cláusula 3', 'Sección 2.1')."
        )
    )
    topics_touched: list[str] = Field(
        description=(
            "Legal or commercial categories affected by the changes "
            "(e.g., 'Plazo', 'Honorarios', 'Confidencialidad', 'Terminación')."
        )
    )
    summary_of_the_change: str = Field(
        description=(
            "Detailed narrative summary of ALL changes between the original contract "
            "and the amendment. Must reference specific sections and state: what changed, "
            "from what value, to what value."
        )
    )

    @field_validator("sections_changed", "topics_touched")
    @classmethod
    def non_empty_list(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("La lista no puede estar vacía — se requiere al menos un elemento.")
        return v

    @field_validator("summary_of_the_change")
    @classmethod
    def summary_min_length(cls, v: str) -> str:
        if len(v) < 50:
            raise ValueError("El resumen es demasiado corto — debe incluir una descripción detallada.")
        return v

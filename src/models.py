from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContextMap(BaseModel):
    """Salida estructurada del agente de contextualización."""

    model_config = ConfigDict(extra="forbid")

    document_type: str = Field(
        description="Tipo de contrato o acuerdo detectado en los documentos."
    )
    parties: list[str] = Field(
        default_factory=list,
        description="Partes involucradas, idealmente con su rol contractual."
    )
    contract_date: str = Field(
        default="No identificada",
        description="Fecha del contrato original tal como aparece en el documento."
    )
    general_purpose: str = Field(
        default="Propósito general no identificado.",
        description="Resumen breve del propósito general del contrato."
    )
    structure_summary: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapa de secciones entre original y enmienda. "
            "Ejemplo: {'Cláusula 1': 'presente en ambos'}."
        )
    )
    is_degraded: bool = Field(
        default=False,
        description=(
            "Indica si el mapa contextual fue aceptado con informacion parcial "
            "y no cumple el nivel ideal de completitud."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_structure_summary(cls, data):
        if not isinstance(data, dict):
            return data

        if "structure_summary" not in data:
            for alias in ("section_mapping", "sections_map", "sections_summary", "clause_mapping"):
                if alias in data and isinstance(data[alias], dict):
                    data["structure_summary"] = data[alias]
                    break
            else:
                data["structure_summary"] = {}

        if not data.get("structure_summary"):
            data["is_degraded"] = True

        return data


class ContractChangeOutput(BaseModel):
    """Schema de salida validado para el análisis de cambios contractuales."""

    model_config = ConfigDict(extra="forbid")

    sections_changed: list[str] = Field(
        description=(
            "Lista de identificadores de sección/cláusula que fueron modificados, agregados o eliminados. "
            "Usar los identificadores exactos del documento (ej. 'Cláusula 3', 'Sección 2.1')."
        )
    )
    topics_touched: list[str] = Field(
        description=(
            "Categorías legales o comerciales afectadas por los cambios "
            "(ej. 'Plazo', 'Honorarios', 'Confidencialidad', 'Terminación')."
        )
    )
    summary_of_the_change: str = Field(
        description=(
            "Resumen narrativo detallado de TODOS los cambios entre el contrato original "
            "y la enmienda. Debe referenciar secciones específicas e indicar: qué cambió, "
            "de qué valor, a qué valor."
        )
    )

    @field_validator("summary_of_the_change")
    @classmethod
    def summary_min_length(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 50:
            raise ValueError("El resumen es demasiado corto — debe incluir una descripción detallada.")
        return cleaned

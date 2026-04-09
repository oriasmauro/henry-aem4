from pydantic import BaseModel, Field, field_validator


class ContractChangeOutput(BaseModel):
    """Schema de salida validado para el análisis de cambios contractuales."""

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
        if len(v) < 50:
            raise ValueError("El resumen es demasiado corto — debe incluir una descripción detallada.")
        return v

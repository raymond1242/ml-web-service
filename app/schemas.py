"""Schemas Pydantic para validación de inputs y respuestas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SummarizeRequest(BaseModel):
    """Input de POST /predict: texto a resumir y longitudes del resumen."""

    text: str = Field(
        ...,
        min_length=20,
        max_length=20_000,
        description="Texto a resumir (idealmente en inglés).",
        examples=[
            "The James Webb Space Telescope has captured its deepest image "
            "of the universe yet, revealing thousands of galaxies in a tiny "
            "sliver of sky. Astronomers say the observations will help them "
            "understand how the first galaxies formed after the Big Bang."
        ],
    )
    min_length: int = Field(30, ge=10, le=200, description="Mínimo de tokens del resumen.")
    max_length: int = Field(130, ge=20, le=400, description="Máximo de tokens del resumen.")

    @model_validator(mode="after")
    def _check_lengths(self) -> "SummarizeRequest":
        """Garantiza que min_length sea menor que max_length."""
        if self.min_length >= self.max_length:
            raise ValueError("min_length debe ser menor que max_length.")
        return self


class Metadata(BaseModel):
    """Metadatos de la predicción."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    device: str
    input_chars: int
    input_tokens: int
    output_tokens: int
    compression_ratio: float = Field(..., description="output_tokens / input_tokens.")


class SummarizeResponse(BaseModel):
    """Respuesta estructurada de POST /predict."""

    summary: str
    inference_time_ms: float
    metadata: Metadata


class HealthResponse(BaseModel):
    """Respuesta de GET /health."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    device: str

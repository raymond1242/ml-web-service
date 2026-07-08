"""API de resumen automático de texto (FastAPI + facebook/bart-large-cnn)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.model import SummarizationModel
from app.schemas import (
    HealthResponse,
    Metadata,
    SummarizeRequest,
    SummarizeResponse,
)

ml = SummarizationModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo en memoria al arrancar y lo libera al apagar."""
    print(f"[startup] Cargando modelo '{ml.model_name}' en '{ml.device}'...")
    ml.load()
    print("[startup] Modelo cargado y listo.")
    yield
    print("[shutdown] Cerrando servicio.")


app = FastAPI(
    title="ML Web Service — Resumen de texto",
    description=(
        "API que expone facebook/bart-large-cnn para resumen automático "
        "de texto (Hugging Face Transformers)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Confirma que el modelo está cargado en memoria."""
    return HealthResponse(
        status="ok" if ml.loaded else "loading",
        model_loaded=ml.loaded,
        model_name=ml.model_name,
        model_version=ml.model_version,
        device=ml.device,
    )


@app.post("/predict", response_model=SummarizeResponse, tags=["prediction"])
def predict(req: SummarizeRequest) -> SummarizeResponse:
    """Resume el texto recibido y devuelve resumen, tiempo de inferencia y metadatos."""
    if not ml.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo aún no está cargado. Reintenta en unos segundos.",
        )

    try:
        result = ml.summarize(req.text, req.min_length, req.max_length)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {exc}",
        ) from exc

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    compression = round(output_tokens / input_tokens, 3) if input_tokens else 0.0

    return SummarizeResponse(
        summary=result["summary"],
        inference_time_ms=result["inference_time_ms"],
        metadata=Metadata(
            model_name=ml.model_name,
            model_version=ml.model_version,
            device=ml.device,
            input_chars=len(req.text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compression_ratio=compression,
        ),
    )


@app.get("/", include_in_schema=False)
def root():
    """Enlaces útiles del servicio."""
    return {"message": "ML Web Service — Resumen de texto", "docs": "/docs", "health": "/health"}

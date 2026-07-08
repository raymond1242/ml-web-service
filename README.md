# ML Web Service — Resumen automático de texto

API HTTP construida con **FastAPI** que expone el modelo pre-entrenado
[`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)
(Hugging Face Transformers) para **resumen automático de texto**.

**Repositorio:** https://github.com/raymond1242/ml-web-service


## Qué predice el modelo

`bart-large-cnn` es un modelo *seq2seq* afinado sobre el dataset CNN/DailyMail.
Recibe un texto (en **inglés**) y genera un **resumen abstractivo**
más corto que conserva ideas principales.

## Requerimientos

| Requisito | Implementación |
|---|---|
| Health check | `GET /health` reporta estado y si el modelo está cargado |
| Endpoint de predicción | `POST /predict` recibe texto y devuelve el resumen |
| Validación de inputs | Schemas Pydantic con `min_length`/`max_length` y validador `min < max` |
| Modelo en memoria | Carga única al iniciar vía `lifespan` de FastAPI |
| Respuesta estructurada | JSON con `summary`, `inference_time_ms` y `metadata` |

## Estructura

```
ml-web-service/
├── app/
│   ├── __init__.py
│   ├── main.py       # FastAPI: lifespan (carga del modelo) + endpoints
│   ├── model.py      # Carga e inferencia de bart-large-cnn
│   └── schemas.py    # Schemas Pydantic (validación + respuesta)
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Cómo correr API localmente

Requiere **Python 3.11**.

```bash
# 1. Clonar el repositorio
git clone https://github.com/raymond1242/ml-web-service.git
cd ml-web-service

# 2. Crear entorno virtual e instalar dependencias
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Levantar el servidor (la primera vez descarga el modelo, ~1.6 GB)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Cuando veas `Application startup complete`, el modelo ya está en memoria.

- **Documentación interactiva (Swagger):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

## Ejemplos de request

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "facebook/bart-large-cnn",
  "model_version": "1.0.0",
  "device": "mps"
}
```

### `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The James Webb Space Telescope has captured its deepest image of the universe yet, revealing thousands of galaxies in a tiny sliver of sky. Astronomers say the observations, which took just over 12 hours, will help them understand how the first galaxies formed after the Big Bang. The telescope, launched in December 2021, is the most powerful space observatory ever built and is a joint project of NASA, the European Space Agency and the Canadian Space Agency."
  }'
```

```json
{
  "summary": "Astronomers say the observations will help them understand how the first galaxies formed after the Big Bang. The telescope, launched in December 2021, is the most powerful space observatory ever built.",
  "inference_time_ms": 5286.54,
  "metadata": {
    "model_name": "facebook/bart-large-cnn",
    "model_version": "1.0.0",
    "device": "mps",
    "input_chars": 461,
    "input_tokens": 91,
    "output_tokens": 41,
    "compression_ratio": 0.451
  }
}
```

> La primera inferencia es más lenta (~3 s) por el *warmup* de MPS; las
> siguientes bajan a ~0.5–1.5 s.

### Parámetros opcionales del resumen

| Campo | Tipo | Default | Rango | Descripción |
|---|---|---|---|---|
| `text` | str | — (requerido) | 20–20000 chars | Texto a resumir |
| `min_length` | int | 30 | 10–200 | Mínimo de tokens del resumen |
| `max_length` | int | 130 | 20–400 | Máximo de tokens del resumen |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "<tu texto>", "min_length": 20, "max_length": 60}'
```

### Ejemplo de validación (HTTP 422)

Texto demasiado corto:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" -d '{"text": "hola"}'
# -> 422: "String should have at least 20 characters"
```

## Cómo correr con Docker (opcional)

El modelo se pre-descarga durante el build, así el contenedor arranca rápido.

```bash
docker build -t ml-web-service .
docker run -p 8000:8000 ml-web-service
```

Luego prueba los mismos `curl` contra `http://127.0.0.1:8000`.

## Notas

- El modelo está entrenado en inglés; con textos en español la calidad del
  resumen baja notablemente.
- En hardware sin GPU corre en CPU (más lento pero funcional).

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator 
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"

try:
    from .predict import MODEL_PATH, predict_with_explanation
except ImportError:
    from predict import MODEL_PATH, predict_with_explanation

app = FastAPI(title="Credit Risk Decision Engine API")
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)


class PredictRequest(BaseModel):
    root: Dict[str, Any]


def apply_sklearn_compatibility() -> None:
    try:
        from sklearn.compose import _column_transformer as _ct

        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        return


def run_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    apply_sklearn_compatibility()
    return predict_with_explanation(payload)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    if INDEX_PATH.exists():
        return INDEX_PATH.read_text(encoding="utf-8")
    return """
    <html>
      <head><title>Credit Risk Decision Engine API</title></head>
      <body>
        <h2>Credit Risk Decision Engine API</h2>
        <p>Open <code>/docs</code> for interactive API docs.</p>
      </body>
    </html>
    """


@app.get("/status")
def status() -> Dict[str, Any]:
    model_path = Path(MODEL_PATH)
    if not model_path.is_absolute():
        model_path = (APP_DIR.parent / model_path).resolve()
    return {
        "status": "ok" if model_path.exists() else "error",
        "model_exists": model_path.exists(),
        "model_path": str(model_path),
    }


@app.post("/predict")
def predict(data: PredictRequest) -> JSONResponse:
    try:
        result = run_prediction(data.root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return JSONResponse(result)

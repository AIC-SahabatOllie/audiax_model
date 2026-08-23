import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ai import MachineBaseline, calibrate, inspect, warm_up

_model_ready = False


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _model_ready
    try:
        warm_up()
        _model_ready = True
    except Exception:
        _model_ready = False
    yield


app = FastAPI(title="audiax-ai", lifespan=_lifespan)


@app.get("/healthz")
def healthz() -> dict:
    if not _model_ready:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    return {"status": "ok"}


def _save_upload_to_tempfile(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name


@app.post("/v1/calibrate")
def v1_calibrate(audio: UploadFile = File(...), machine_label: str = Form(...)) -> dict:
    path = _save_upload_to_tempfile(audio)
    try:
        baseline = calibrate(path, machine_label=machine_label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(path).unlink(missing_ok=True)
    return baseline.to_json_dict()


@app.post("/v1/inspect")
def v1_inspect(audio: UploadFile = File(...), baseline_json: str = Form(...)) -> dict:
    try:
        baseline = MachineBaseline.from_json_dict(json.loads(baseline_json))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"baseline_json tidak valid: {e}")

    path = _save_upload_to_tempfile(audio)
    try:
        card = inspect(path, baseline)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(path).unlink(missing_ok=True)

    return card.to_dict()

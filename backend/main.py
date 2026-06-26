"""
main.py — FastAPI app for the document-KB assistant.

Routes:
  POST   /api/chat          -> SSE stream (thinking / tool / token / sources / done)
  GET    /api/files         -> list KB documents + status
  POST   /api/files         -> upload a document (convert + index)
  DELETE /api/files/{name}  -> remove a document (source + markdown + index)
  GET    /api/status        -> model / counts
Serves the built UI at / (single-process mode; see run.sh).
"""
from __future__ import annotations
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import models
import indexer
import agent
import kb
import converter


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.load_reasoning()
    models.get_embedder()
    if not config.BM25_INDEX_PATH.exists():
        n = indexer.rebuild_bm25()
        print(f"[startup] BM25 index built over {n} documents")
    yield


app = FastAPI(title="Personal Document KB", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"], allow_headers=["*"],
)


# ── chat ─────────────────────────────────────────────────────────────────────
class ChatBody(BaseModel):
    message: str
    history: list[dict] = []


@app.post("/api/chat")
def chat(body: ChatBody):
    def event_stream():
        try:
            for ev in agent.run_agent(body.message, body.history):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── files ────────────────────────────────────────────────────────────────────
@app.get("/api/files")
def list_files():
    return {"files": kb.list_files()}


@app.post("/api/files")
async def upload(file: UploadFile = File(...)):
    if not converter.is_supported(file.filename):
        raise HTTPException(400, f"Unsupported file type: {Path(file.filename).suffix}")
    data = await file.read()
    try:
        rec = kb.add_file(file.filename, data)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")
    return {"status": "ok", "file": rec}


@app.delete("/api/files/{name:path}")
def delete_file(name: str):
    if not kb.remove_file(name):
        raise HTTPException(404, f"No such document: {name}")
    return {"status": "ok", "removed": Path(name).name}


# ── status ───────────────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    return {
        "reasoning_model": models.reasoning_model_name(),
        "backend": config.LLM_BACKEND,
        "documents": kb.count(),
        "models_loaded": models.models_loaded(),
        "assistant_name": config.ASSISTANT_NAME,
        "user_name": config.USER_NAME,
    }


# ── serve built UI (single-process) ──────────────────────────────────────────
_UI_DIST = config.FRONTEND_DIR / "dist"
if _UI_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
else:
    print(f"[ui] {_UI_DIST} not found - run 'npm run build' in frontend/ (or use ./run.sh)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

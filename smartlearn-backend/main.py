import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .services.rag import prepare_rag_chat_record, answer_chat_turn

app = FastAPI(title="SmartLearn Lite API")

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
async def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/upload")
async def upload_pdf(
    chat_id: str = Query(..., description="Chat session identifier"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    # Reject non-PDF
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Read into bytes
    pdf_bytes = await file.read()

    # Reject empty
    if not pdf_bytes:
        raise HTTPException(400, "Uploaded file is empty")

    # Build the richer RAG-ready record in one call
    # (pages → chunks → embeddings → FAISS, plus PDF save to disk)
    try:
        document = prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename,
            pdf_bytes=pdf_bytes,
            upload_root="uploads",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Reject zero readable text
    pages = document["pages"]
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(
            422,
            "PDF contains no machine-readable text — OCR is not supported",
        )

    # Stamp with upload time for the periodic cleanup task
    document["uploaded_at"] = time.time()

    # Store the richer record — only on success so no half-written state
    documents[chat_id] = document

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.get("/documents/{chat_id}/file")
async def serve_uploaded_pdf(chat_id: str):
    """Serve the uploaded PDF file for a chat session."""
    document = documents.get(chat_id)
    if document is None:
        raise HTTPException(404, f"No document found for chat_id '{chat_id}'")

    saved_path = document.get("saved_pdf_path")
    if not saved_path or not Path(saved_path).exists():
        raise HTTPException(404, f"PDF file not found for chat_id '{chat_id}'")

    return FileResponse(saved_path, media_type="application/pdf")


@app.post("/chat")
async def chat(request: ChatRequest):
    stored = documents.get(request.chat_id)
    if stored is None:
        raise HTTPException(
            404,
            f"No document uploaded for chat_id '{request.chat_id}'. "
            f"Upload a PDF first via POST /upload?chat_id={request.chat_id}",
        )

    try:
        result = await asyncio.to_thread(
            answer_chat_turn,
            document=stored,
            message=request.message,
            top_k=3,
            candidate_pool=60,
            answer_model="poolside/laguna-s-2.1:free",
        )
    except FileNotFoundError as e:
        raise HTTPException(500, f"RAG artifact missing: {e}")
    except Exception:
        raise HTTPException(502, "Upstream AI service failed")

    return result


async def _cleanup_old_documents():
    """Remove documents older than 1 hour to prevent unbounded memory growth."""
    while True:
        await asyncio.sleep(3600)
        now = time.time()
        expired = [
            cid
            for cid, doc in documents.items()
            if now - doc.get("uploaded_at", 0) > 3600
        ]
        for cid in expired:
            del documents[cid]


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_cleanup_old_documents())

import asyncio
import os
import re
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.pdf import extract_pages
from .services.llm import answer_from_pages

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

documents: dict[str, tuple[list[dict], float]] = {}


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

    # Extract pages
    try:
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Reject zero readable text
    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(
            422,
            "PDF contains no machine-readable text — OCR is not supported",
        )

    # Store page records keyed by chat session, with timestamp for cleanup
    documents[chat_id] = (pages, time.time())

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


def extract_citations(answer: str, pages: list[dict]) -> list[int]:
    """Extract [Page X] numbers from the answer, keeping only those that exist in pages."""
    existing = {p["page"] for p in pages}
    found = {int(n) for n in re.findall(r"\[Page (\d+)\]", answer)}
    return sorted(found & existing)


@app.post("/chat")
async def chat(request: ChatRequest):
    stored = documents.get(request.chat_id)
    if stored is None:
        raise HTTPException(
            404,
            f"No document uploaded for chat_id '{request.chat_id}'. "
            f"Upload a PDF first via POST /upload?chat_id={request.chat_id}",
        )

    pages, _ = stored

    try:
        answer = await asyncio.to_thread(answer_from_pages, pages, request.message)
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except Exception:
        raise HTTPException(502, "Upstream AI service failed")

    citations = extract_citations(answer, pages)
    return {"answer": answer, "citations": citations}


async def _cleanup_old_documents():
    """Remove documents older than 1 hour to prevent unbounded memory growth."""
    while True:
        await asyncio.sleep(3600)
        now = time.time()
        expired = [cid for cid, (_, ts) in documents.items() if now - ts > 3600]
        for cid in expired:
            del documents[cid]


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_cleanup_old_documents())

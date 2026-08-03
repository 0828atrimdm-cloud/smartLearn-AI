"""
RAG (Retrieval-Augmented Generation) helpers for SmartLearn.

Day 3 preparation:
  - clean extracted PDF text
  - extract pages without a hard page cap
  - save / load JSON artifacts
  - preview records as a simple table
  - embedding pipeline: model loading, text encoding, artifact persistence
"""

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text.

    Handles the most common PDF-extraction noise:
    * null bytes (``\\x00``)           ― leftover encoding artifacts
    * soft hyphens (``\\u00ad``)       ― invisible hyphenation markers
    * repeated whitespace              ― tabs, multiple spaces, etc.
    * hard line breaks mid-sentence    ― PDFs often break lines at column
                                         width, not at sentence boundaries
    """
    if not text:
        return ""

    # Remove null bytes (encoding noise)
    text = text.replace("\x00", "")

    # Remove soft hyphens (invisible hyphenation markers)
    text = text.replace("­", "")

    # Collapse repeated whitespace (tabs, spaces, etc.) into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse three or more consecutive newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Heuristic: join lines that were broken mid-sentence.
    # A line ending with a lowercase letter or common mid-sentence punctuation
    # (comma, semicolon, colon) is probably NOT a real paragraph break.
    # We replace that single newline with a space to re-join the sentence.
    text = re.sub(r"(?<=[a-z,;:])\n(?=[a-z])", " ", text)

    # Strip leading/trailing whitespace from each line, then the whole text
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    return text


# ---------------------------------------------------------------------------
# 2. PDF extraction (RAG-friendly — no hard page cap)
# ---------------------------------------------------------------------------

def extract_pages_for_rag(
    file_path: bytes | str | Path,
    page_limit: int | None = None,
) -> list[dict]:
    """Read a PDF page by page, keeping original PDF page numbers.

    *file_path* may be raw PDF bytes or a file path (``str`` or ``Path``).
    When *page_limit* is set only the first *page_limit* non-empty pages
    are returned (``None`` means no limit).

    Returns only pages whose cleaned text is non-empty.
    """
    if isinstance(file_path, (str, Path)):
        pdf_bytes = Path(file_path).read_bytes()
    else:
        pdf_bytes = file_path
    reader = PdfReader(BytesIO(pdf_bytes))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:                         # drop empty pages
            records.append({"page": page_number, "text": cleaned})
            if page_limit is not None and len(records) >= page_limit:
                break

    return records


# ---------------------------------------------------------------------------
# 3. JSON persistence helpers
# ---------------------------------------------------------------------------

def save_json(obj, file_path: str | Path) -> None:
    """Save a Python object to a UTF-8 JSON file.

    Creates parent folders automatically when they don't exist yet.
    Accepts either a ``str`` or a ``pathlib.Path``.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(file_path: str | Path) -> object:
    """Read a JSON artifact back into Python.

    Accepts either a ``str`` or a ``pathlib.Path``.
    Returns the decoded Python object.
    """
    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 4. Notebook-style preview
# ---------------------------------------------------------------------------

def preview_records(
    records: list[dict],
    columns: Optional[list[str]] = None,
    max_rows: int = 12,
    max_cell_width: int = 80,
) -> None:
    """Print a small table for a slice of *records* so you can inspect
    page / chunk artifacts quickly.

    Parameters
    ----------
    records : list[dict]
        The list of ``{page, text, ...}`` records to display.
    columns : list[str] or None
        Which keys to show as columns.  Defaults to every key
        found in the first record.
    max_rows : int
        Maximum number of data rows to print (default 12).
    max_cell_width : int
        Characters beyond this width are truncated with ``…``.
    """
    if not records:
        print("(no records)")
        return

    # Determine columns
    if columns is None:
        columns = list(records[0].keys())

    # Compute column widths — never smaller than the header, capped
    col_widths: dict[str, int] = {}
    for col in columns:
        header_w = len(col)
        data_w = 0
        for r in records[:max_rows]:
            data_w = max(data_w, len(str(r.get(col, ""))))
        col_widths[col] = min(max(header_w, data_w), max_cell_width)

    # Helper: format one cell
    def _cell(value) -> str:
        s = str(value) if value is not None else ""
        if len(s) > max_cell_width:
            s = s[: max_cell_width - 1] + "…"
        return s

    # Build separator line
    sep = "+" + "+".join("-" * (col_widths[c] + 2) for c in columns) + "+"

    # Print header
    print(sep)
    header = "| " + " | ".join(
        c.ljust(col_widths[c]) for c in columns
    ) + " |"
    print(header)
    print(sep.replace("-", "="))

    # Print rows
    for i, r in enumerate(records):
        if i >= max_rows:
            print(f"  … ({len(records) - max_rows} more rows)")
            break
        row = "| " + " | ".join(
            _cell(r.get(c, "")).ljust(col_widths[c]) for c in columns
        ) + " |"
        print(row)
        if i < min(max_rows, len(records)) - 1:
            print(sep.replace("-", " ").replace("+", "|"))

    print(sep)


# ---------------------------------------------------------------------------
# 5. Chunking helpers
# ---------------------------------------------------------------------------

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into smaller pieces.

    Prefers natural boundaries in this order:

    1. paragraph breaks (double newlines)
    2. sentence breaks (``.`` ``!`` ``?`` followed by whitespace)
    3. word boundaries (spaces)
    4. character-level split — *only* when a single word exceeds
       ``chunk_size`` and cannot be broken any other way

    Never splits mid-word unless a solitary word is longer than
    ``chunk_size`` and character-level fallback is unavoidable.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []

    # ---- level 1: paragraph boundaries --------------------------------
    paragraphs = re.split(r"\n\n+", text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue

        # ---- level 2: sentence boundaries -------------------------
        sentences = re.split(r"(?<=[.!?])\s+", para)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= chunk_size:
                pieces.append(sent)
                continue

            # ---- level 3: word boundaries -------------------------
            words = sent.split()
            buf: list[str] = []
            buf_len = 0
            for w in words:
                # +1 accounts for the space before this word
                need = len(w) + (1 if buf else 0)
                if buf_len + need <= chunk_size:
                    buf.append(w)
                    buf_len += need
                else:
                    if buf:
                        pieces.append(" ".join(buf))
                        buf = []
                        buf_len = 0
                    # Single word longer than chunk_size → character split
                    if len(w) > chunk_size:
                        for i in range(0, len(w), chunk_size):
                            pieces.append(w[i : i + chunk_size])
                    else:
                        buf.append(w)
                        buf_len = len(w)
            if buf:
                pieces.append(" ".join(buf))

    return pieces


def chunk_by_paragraph(records: list[dict], chunk_size: int) -> list[dict]:
    """Convert page records into chunks while preserving paragraph boundaries.

    Each record's text is first split into paragraphs (on double newlines).
    Paragraphs that fit within *chunk_size* become a single chunk; oversized
    paragraphs are subdivided via :func:`slice_long_text` so no chunk exceeds
    the limit.
    """
    chunks: list[dict] = []
    chunk_id = 0

    for rec in records:
        page = rec["page"]
        text = rec["text"]

        paragraphs = re.split(r"\n\n+", text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            pieces = slice_long_text(para, chunk_size)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page,
                        "text": piece,
                        "chunk_mode": "paragraph",
                    }
                )
                chunk_id += 1

    return chunks


def chunk_by_characters(
    records: list[dict], chunk_size: int, overlap: int = 0
) -> list[dict]:
    """Create plain fixed-size sliding-window chunks with optional overlap.

    Each window is *chunk_size* characters wide.  When *overlap* is 0
    (``"character"`` mode) every character appears in exactly one chunk.
    When *overlap* > 0 (``"character_overlap"`` mode) consecutive chunks
    share *overlap* characters and the overlap value is recorded in each
    chunk's metadata.
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    chunks: list[dict] = []
    chunk_id = 0
    mode = "character_overlap" if overlap > 0 else "character"
    step = chunk_size - overlap

    for rec in records:
        page = rec["page"]
        text = rec["text"]

        if not text:
            continue

        pos = 0
        text_len = len(text)

        while pos < text_len:
            piece = text[pos : pos + chunk_size].strip()

            # Skip whitespace-only remainders at the very end
            if pos + chunk_size >= text_len and not piece:
                break

            if piece:
                chunk: dict = {
                    "chunk_id": chunk_id,
                    "page": page,
                    "text": piece,
                    "chunk_mode": mode,
                }
                if overlap > 0:
                    chunk["overlap"] = overlap
                chunks.append(chunk)
                chunk_id += 1

            pos += step

            # Guard against zero-step (shouldn't happen after the validation
            # above, but keeps the loop safe during maintenance)
            if step <= 0:  # pragma: no cover
                break

    return chunks


def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    separators: list[str] | None = None,
) -> list[dict]:
    """Split pages into chunks using LangChain's RecursiveCharacterTextSplitter.

    This splitter tries to break text at the most natural boundary first,
    then falls back to coarser splits when a chunk still exceeds *chunk_size*.
    The default separator priority is designed for prose / PDF text:

    1. ``"\\n\\n"``  — paragraph breaks
    2. ``"\\n"``     — line breaks
    3. ``" "``       — word boundaries
    4. ``""``        — character-level fallback (last resort)

    Parameters
    ----------
    pages : list[dict]
        Page records produced by :func:`extract_pages_for_rag`.
    chunk_size : int
        Target maximum characters per chunk (default 500).
    chunk_overlap : int
        Number of characters shared between consecutive chunks (default 0).
    separators : list[str] or None
        Override the default separator priority list.
        When *None* the default ``["\\n\\n", "\\n", " ", ""]`` is used.

    Returns
    -------
    list[dict]
        Every chunk contains ``chunk_id``, ``page``, ``text``, and
        ``chunk_mode`` set to ``"langchain_recursive"``.

    Raises
    ------
    ImportError
        If ``langchain-text-splitters`` is not installed.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "langchain-text-splitters is required for langchain_recursive mode. "
            "Install it with:  pip install langchain-text-splitters"
        )

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
        strip_whitespace=True,
    )

    chunks: list[dict] = []
    chunk_id = 0

    for rec in pages:
        page = rec["page"]
        text = rec.get("text", "")

        if not text or not text.strip():
            continue

        pieces = splitter.split_text(text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": page,
                    "text": piece,
                    "chunk_mode": "langchain_recursive",
                }
            )
            chunk_id += 1

    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str,
    chunk_size: int = 500,
    overlap: int = 0,
) -> list[dict]:
    """Select a chunking strategy and return uniform chunks.

    Parameters
    ----------
    records : list[dict]
        Page records produced by :func:`extract_pages_for_rag`.
    chunk_mode : str
        One of ``"paragraph"``, ``"character"``, or
        ``"character_overlap"``.
    chunk_size : int
        Target maximum characters per chunk (default 500).
    overlap : int
        Character overlap between consecutive windows.
        Only meaningful for ``"character_overlap"`` mode (default 0).

    Returns
    -------
    list[dict]
        Every chunk contains at least ``chunk_id``, ``page``, ``text``,
        and ``chunk_mode``.

    Raises
    ------
    ValueError
        If *chunk_mode* is unrecognised or ``"character_overlap"`` is requested
        with ``overlap <= 0``.
    """
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)

    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)

    if chunk_mode == "character_overlap":
        if overlap <= 0:
            raise ValueError(
                "character_overlap mode requires overlap > 0; "
                f"got overlap={overlap}"
            )
        return chunk_by_characters(records, chunk_size, overlap=overlap)

    if chunk_mode == "langchain_recursive":
        return chunk_with_langchain_recursive(records, chunk_size, overlap)

    raise ValueError(
        f"Unknown chunk_mode: {chunk_mode!r}. "
        "Expected 'paragraph', 'character', 'character_overlap', "
        "or 'langchain_recursive'."
    )


# ---------------------------------------------------------------------------
# 6. Embedding pipeline helpers
# ---------------------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Sanitise a string so it is safe to use as a path component."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))


def model_tag(model_name: str) -> str:
    """Turn a HuggingFace-style model name into a safe filename suffix.

    ``"sentence-transformers/all-MiniLM-L6-v2"`` → ``"all-MiniLM-L6-v2"``
    """
    # Strip common org prefix so we keep just the model id
    tag = model_name.rsplit("/", 1)[-1]
    tag = _safe_name(tag)
    return tag or "unknown_model"


def resolve_model_source(
    model_name: str, artifact_root: str | Path = "artifacts/rag"
) -> str:
    """Return a local cached model path when it already exists.

    Otherwise return *model_name* unchanged so that
    ``SentenceTransformer`` downloads from HuggingFace on first use.

    The local cache is expected under
    ``{artifact_root}/hf_models/{model_tag}``.
    """
    root = Path(artifact_root)
    local = root / "hf_models" / model_tag(model_name)
    if local.exists():
        return str(local)
    return model_name


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is available, ``"cpu"`` otherwise."""
    try:
        import torch  # type: ignore[import-untyped]

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_model(
    model_name: str,
    device: str | None = None,
    artifact_root: str | Path = "artifacts/rag",
):
    """Create (or reuse, from a local cache) one SentenceTransformer instance.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"all-MiniLM-L6-v2"``.
    device : str or None
        Device string (``"cpu"``, ``"cuda"``).  Auto-detected when *None*.
    artifact_root : str or Path
        Root directory for cached model folders.

    Returns
    -------
    sentence_transformers.SentenceTransformer
    """
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    if device is None:
        device = get_device()

    source = resolve_model_source(model_name, artifact_root)
    return SentenceTransformer(source, device=device)


def embed_texts(
    texts: list[str],
    model,
    batch_size: int = 32,
):
    """Encode a list of texts into L2-normalised ``float32`` vectors.

    Parameters
    ----------
    texts : list[str]
        Chunk texts to embed.
    model : SentenceTransformer
        An already-loaded model.
    batch_size : int
        Mini-batch size for encoding (default 32).

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)``, dtype ``float32``.
    """
    import numpy as np  # type: ignore[import-untyped]

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def artifact_paths_for(
    document_id: str,
    pdf_name: str,
    model_name: str,
    chunk_mode: str = "paragraph",
    chunk_size: int = 500,
    overlap: int = 0,
    artifact_root: str | Path = "artifacts/rag",
) -> dict:
    """Decide where artifacts for one PDF + config are stored.

    Returns a dictionary with keys:

    * ``raw_pages``       — path to saved page records
    * ``chunks``          — path to saved chunk records
    * ``embeddings``      — path to the ``.npy`` embedding matrix
    * ``manifest``        — path to the JSON manifest
    * ``faiss_index``     — path to the ``.faiss`` binary index
    * ``faiss_meta``      — path to the FAISS metadata JSON
    * ``chunk_root``      — directory shared by chunks / embeddings / manifest
    """
    root = Path(artifact_root) / _safe_name(document_id)
    tag = model_tag(model_name)

    prefix = f"{chunk_mode}_{chunk_size}_{overlap}_"

    return {
        "raw_pages": root / "raw_pages" / "pages.json",
        "chunks": root / "chunks" / f"{prefix}chunks.json",
        "embeddings": root / "embeddings" / f"{prefix}embeddings_{tag}.npy",
        "manifest": root / "embeddings" / f"{prefix}manifest_{tag}.json",
        "faiss_index": root / "faiss" / f"{prefix}index_{tag}.faiss",
        "faiss_meta": root / "faiss" / f"{prefix}faiss_meta_{tag}.json",
        "chunk_root": root,
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | bytes | str | Path,
    chunk_mode: str = "paragraph",
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    overlap: int = 0,
    batch_size: int = 32,
    artifact_root: str | Path = "artifacts/rag",
) -> dict:
    """Build or reuse the full pages → chunks → embeddings → manifest bundle.

    On the first call the function extracts pages, chunks, embeds, and
    persists everything to disk.  On subsequent calls with the same
    parameters it returns the cached manifest — as long as the three
    artifact files (pages, chunks, embeddings) still exist.

    Parameters
    ----------
    document_id : str
        Stable identifier for the document (e.g. ``"day3-demo"``).
    pdf_name : str
        Human-readable label for the PDF (used in path names).
    pages : list[dict] | bytes | str | Path
        Either page records (``[{"page":1, "text":"..."}, ...]``) or a
        PDF source (raw bytes, file path) to extract pages from.
    chunk_mode : str
        ``"paragraph"``, ``"character"``, or ``"character_overlap"``.
    model_name : str
        HuggingFace model id (default ``"all-MiniLM-L6-v2"``).
    chunk_size : int
        Max characters per chunk (default 500).
    overlap : int
        Character overlap for sliding-window modes (default 0).
    batch_size : int
        Mini-batch size for the embedding model (default 32).
    artifact_root : str or Path
        Root directory for all artifacts (default ``"artifacts/rag"``).

    Returns
    -------
    dict
        Manifest with keys: ``document_id``, ``pdf_name``, ``num_pages``,
        ``chunk_mode``, ``chunk_size``, ``overlap``, ``model_name``,
        ``num_chunks``, ``embedding_dim``, ``device``, ``chunk_path``,
        ``embedding_path``, ``raw_pages_path``.
    """
    import numpy as np  # type: ignore[import-untyped]

    paths = artifact_paths_for(
        document_id=document_id,
        pdf_name=pdf_name,
        model_name=model_name,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        artifact_root=artifact_root,
    )

    # ---- reuse cached bundle when it still matches --------------------------
    manifest_path: Path = paths["manifest"]
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        # Quick sanity check — every file the manifest points to must exist
        expected_files = [
            manifest.get("chunk_path"),
            manifest.get("embedding_path"),
            manifest.get("raw_pages_path"),
        ]
        if all(p and Path(p).exists() for p in expected_files):
            chunks = load_json(manifest["chunk_path"])
            embeddings = np.load(manifest["embedding_path"])
            return {"manifest": manifest, "chunks": chunks, "embeddings": embeddings}

    # ---- 1. pages -----------------------------------------------------------
    if isinstance(pages, (bytes, str, Path)):
        page_records = extract_pages_for_rag(pages)
    else:
        page_records = list(pages)

    if not page_records:
        raise ValueError("No non-empty pages found — cannot build artifacts")

    save_json(page_records, paths["raw_pages"])

    # ---- 2. chunks ----------------------------------------------------------
    chunk_records = build_chunks(page_records, chunk_mode, chunk_size, overlap)

    if not chunk_records:
        raise ValueError(
            "Chunking produced zero chunks — check chunk_size / page content"
        )

    save_json(chunk_records, paths["chunks"])

    # ---- 3. embeddings ------------------------------------------------------
    device = get_device()
    model = load_model(model_name, device=device, artifact_root=artifact_root)

    texts = [c["text"] for c in chunk_records]
    embeddings = embed_texts(texts, model, batch_size=batch_size)

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)

    # ---- 4. manifest --------------------------------------------------------
    manifest: dict = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(page_records),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunk_records),
        "embedding_dim": int(embeddings.shape[1]),
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, manifest_path)

    return {"manifest": manifest, "chunks": chunk_records, "embeddings": embeddings}


# ---------------------------------------------------------------------------
# 7. Path display helper
# ---------------------------------------------------------------------------


def relative_path_str(path: str | Path, base: str | Path) -> str:
    """Return *path* as a string relative to *base* when possible.

    When *path* does not live under *base* the original absolute (or
    relative) string is returned unchanged.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 8. Server-style document record
# ---------------------------------------------------------------------------


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build (or reuse) a complete RAG document record.

    This is the server-facing entry point: it runs the full
    pages → chunks → embeddings → FAISS pipeline and returns a dict
    that can be stored in ``documents[document_id]`` for later
    retrieval and answering.

    The returned dict carries *pages*, *chunks*, *chunk_size* (the
    actual number of chunks), *embedding_dim*, *model_name*,
    *model_source*, an empty *history* list, an *artifacts* block
    with relative paths to the index / chunks / embeddings files, and
    a *rag_index* block with the active settings.
    """
    if artifact_root is None:
        artifact_root = "artifacts/rag"

    # Run the full build-or-reuse pipeline --------------------------------
    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    model_source = resolve_model_source(model_name, artifact_root)
    root = Path(artifact_root)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": bundle["chunks"],
        "chunk_size": len(bundle["chunks"]),
        "embedding_dim": int(bundle["embeddings"].shape[1]),
        "model_name": model_name,
        "model_source": str(model_source),
        "history": [],
        "artifacts": {
            "index": relative_path_str(bundle["faiss_path"], root),
            "chunks": relative_path_str(bundle["chunk_path"], root),
            "embeddings": relative_path_str(bundle["embedding_path"], root),
            "raw_pages": relative_path_str(bundle["raw_pages_path"], root),
        },
        "rag_index": {
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model_name": model_name,
            "batch_size": batch_size,
            "artifact_root": str(artifact_root),
        },
    }


# ---------------------------------------------------------------------------
# 9. FAISS index helpers
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: "np.ndarray") -> "faiss.Index":
    """Build a FAISS inner-product index from **L2-normalised** embedding vectors.

    Because the input vectors are already normalised (‖v‖ = 1), inner
    product is equivalent to cosine similarity:

        cos(u, v) = u·v / (‖u‖·‖v‖) = u·v   (when ‖u‖ = ‖v‖ = 1)

    Using ``IndexFlatIP`` is therefore an **exact** cosine-similarity
    search — no approximation, no extra normalisation at query time.
    """
    import faiss  # type: ignore[import-untyped]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: "faiss.Index", index_path: str | Path) -> None:
    """Write one FAISS index to a binary ``.faiss`` file on disk.

    Parent directories are created automatically when they don't exist yet.
    """
    import faiss  # type: ignore[import-untyped]

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(index_path: str | Path) -> "faiss.Index":
    """Read a FAISS index from a ``.faiss`` file back into memory."""
    import faiss  # type: ignore[import-untyped]

    return faiss.read_index(str(Path(index_path)))


# ---------------------------------------------------------------------------
# 10. Configuration signature (cache-busting)
# ---------------------------------------------------------------------------


def _config_signature(
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
    num_chunks: int,
    embedding_dim: int,
) -> str:
    """Short deterministic hash of the parameters that affect the FAISS index.

    When any of these values change the index must be rebuilt; when they
    stay the same a cached ``.faiss`` file can be reused.
    """
    payload = (
        f"{chunk_mode}|{chunk_size}|{overlap}|"
        f"{model_name}|{num_chunks}|{embedding_dim}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 11. High-level ensure_index (pages → chunks → embeddings → FAISS)
# ---------------------------------------------------------------------------


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build or reuse the complete RAG pipeline including a FAISS index.

    This is the one-stop helper for a notebook cell: feed it a PDF (or
    pre-extracted pages) and get back chunks, embeddings, a FAISS index,
    and all the saved artifact paths.

    On the first call the function extracts pages, chunks, embeds, builds
    a FAISS ``IndexFlatIP``, and persists everything to disk.  On later
    calls with the same configuration it reloads the cached ``.faiss``
    file — as long as the configuration *signature* hasn't changed.

    Parameters
    ----------
    document_id : str
        Stable identifier for the document (e.g. ``"day3-demo"``).
    pdf_name : str
        Human-readable label for the PDF (used in path names).
    pages : list[dict] or None
        Pre-extracted page records ``[{"page":1, "text":"..."}, ...]``.
        When *None*, *pdf_path* must be provided.
    pdf_path : str or Path or None
        Path to a PDF file to extract pages from.  Only used when
        *pages* is *None*.
    chunk_mode : str
        One of ``"paragraph"``, ``"character"``, ``"character_overlap"``,
        or ``"langchain_recursive"`` (default ``"character_overlap"``).
    model_name : str
        HuggingFace model id (default ``"sentence-transformers/all-MiniLM-L6-v2"``).
    chunk_size : int
        Max characters per chunk (default 700).
    overlap : int
        Character overlap for sliding-window modes (default 120).
    batch_size : int
        Mini-batch size for the embedding model (default 32).
    artifact_root : str or Path or None
        Root directory for all artifacts.  Defaults to ``"artifacts/rag"``.

    Returns
    -------
    dict
        Bundle with keys: ``chunks``, ``embeddings``, ``manifest``,
        ``index`` (the FAISS ``IndexFlatIP``), ``faiss_path``,
        ``faiss_meta_path``, ``chunk_path``, ``embedding_path``,
        ``raw_pages_path``.
    """
    import numpy as np  # type: ignore[import-untyped]

    if artifact_root is None:
        artifact_root = "artifacts/rag"

    # ---- resolve pages --------------------------------------------------------
    if pages is not None:
        page_records = list(pages)
    elif pdf_path is not None:
        page_records = extract_pages_for_rag(pdf_path)
    else:
        raise ValueError("One of 'pages' or 'pdf_path' must be provided")

    if not page_records:
        raise ValueError("No non-empty pages found — cannot build index")

    # ---- resolve artifact paths -----------------------------------------------
    paths = artifact_paths_for(
        document_id=document_id,
        pdf_name=pdf_name,
        model_name=model_name,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        artifact_root=artifact_root,
    )

    # ---- 1. raw pages ---------------------------------------------------------
    save_json(page_records, paths["raw_pages"])

    # ---- 2. chunks ------------------------------------------------------------
    chunk_records = build_chunks(page_records, chunk_mode, chunk_size, overlap)
    if not chunk_records:
        raise ValueError(
            "Chunking produced zero chunks — check chunk_size / page content"
        )
    save_json(chunk_records, paths["chunks"])

    # ---- 3. embeddings --------------------------------------------------------
    device = get_device()
    model = load_model(model_name, device=device, artifact_root=artifact_root)
    texts = [c["text"] for c in chunk_records]
    embeddings = embed_texts(texts, model, batch_size=batch_size)
    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)

    # ---- 4. configuration signature -------------------------------------------
    sig = _config_signature(
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        num_chunks=len(chunk_records),
        embedding_dim=int(embeddings.shape[1]),
    )

    # ---- 5. reuse cached FAISS index when signature still matches -------------
    faiss_meta_path: Path = paths["faiss_meta"]
    if faiss_meta_path.exists() and paths["faiss_index"].exists():
        try:
            cached_meta = load_json(faiss_meta_path)
            if cached_meta.get("signature") == sig:
                index = load_faiss_index(paths["faiss_index"])

                manifest = _build_manifest(
                    document_id=document_id,
                    pdf_name=pdf_name,
                    num_pages=len(page_records),
                    num_chunks=len(chunk_records),
                    chunk_mode=chunk_mode,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    model_name=model_name,
                    device=device,
                    embedding_dim=int(embeddings.shape[1]),
                    paths=paths,
                )
                save_json(manifest, paths["manifest"])

                return {
                    "chunks": chunk_records,
                    "embeddings": embeddings,
                    "manifest": manifest,
                    "index": index,
                    "faiss_path": str(paths["faiss_index"]),
                    "faiss_meta_path": str(faiss_meta_path),
                    "chunk_path": str(paths["chunks"]),
                    "embedding_path": str(paths["embeddings"]),
                    "raw_pages_path": str(paths["raw_pages"]),
                }
        except Exception:
            # Any hiccup (corrupt index, missing file, …) → rebuild
            pass

    # ---- 6. build & persist fresh FAISS index ---------------------------------
    index = build_faiss_index(embeddings)
    save_faiss_index(index, paths["faiss_index"])

    faiss_meta = {
        "signature": sig,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunk_records),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_index_path": str(paths["faiss_index"]),
    }
    save_json(faiss_meta, faiss_meta_path)

    # ---- 7. manifest ----------------------------------------------------------
    manifest = _build_manifest(
        document_id=document_id,
        pdf_name=pdf_name,
        num_pages=len(page_records),
        num_chunks=len(chunk_records),
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        device=device,
        embedding_dim=int(embeddings.shape[1]),
        paths=paths,
    )
    save_json(manifest, paths["manifest"])

    return {
        "chunks": chunk_records,
        "embeddings": embeddings,
        "manifest": manifest,
        "index": index,
        "faiss_path": str(paths["faiss_index"]),
        "faiss_meta_path": str(faiss_meta_path),
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }


# ---------------------------------------------------------------------------
# 12. Internal helpers
# ---------------------------------------------------------------------------


def _build_manifest(
    document_id: str,
    pdf_name: str,
    num_pages: int,
    num_chunks: int,
    chunk_mode: str,
    chunk_size: int,
    overlap: int,
    model_name: str,
    device: str,
    embedding_dim: int,
    paths: dict,
) -> dict:
    """Assemble a manifest dict describing one fully-built RAG document."""
    return {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": num_pages,
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": num_chunks,
        "embedding_dim": embedding_dim,
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
        "faiss_index_path": str(paths["faiss_index"]),
    }


# ---------------------------------------------------------------------------
# 13. Retrieval helpers
# ---------------------------------------------------------------------------

# Lightweight English stop-word list for lexical token filtering.
# Kept small and inline so the reranker stays easy to explain.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "we", "you", "he", "she", "they", "me", "us", "him", "her",
    "them", "my", "your", "his", "our", "their", "what", "which", "who",
    "whom", "when", "where", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "only", "own", "same", "into",
    "over", "also", "very", "just", "about", "above", "after", "again",
    "any", "as", "up", "out", "down", "off", "here", "there",
}


def keyword_set(text: str) -> set[str]:
    """Extract lightweight lexical tokens from *text* for simple reranking.

    Lowercases the input, splits on word boundaries, then drops common
    English stop words and tokens shorter than 3 characters.  Returns a
    set of lowercase keyword strings suitable for Jaccard-style overlap
    scoring in :func:`_lexical_score`.
    """
    if not text:
        return set()
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _lexical_score(question_kw: set[str], chunk_kw: set[str]) -> float:
    """Fraction of question keywords that appear in the chunk text.

    This is a recall-oriented measure: a score of 1.0 means every
    question keyword was found in the chunk.  Returns 0.0 when the
    question yields no keywords (all stop words or too short).
    """
    if not question_kw:
        return 0.0
    return len(question_kw & chunk_kw) / len(question_kw)


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search an in-memory FAISS index bundle for chunks relevant to *question*.

    Embeds the question with the same model used to build the index,
    retrieves *candidate_pool* nearest neighbours via cosine similarity
    (inner product on L2-normalised vectors), applies a lightweight
    lexical rerank (90 % semantic + 10 % keyword overlap), and returns
    the top *k* hits.

    Parameters
    ----------
    question : str
        The user's question.
    bundle : dict
        An in-memory bundle as returned by :func:`ensure_index`.  Must
        contain at least ``index`` (a FAISS index), ``chunks`` (list of
        chunk dicts), and ``manifest`` (with ``model_name``).
    top_k : int
        Number of hits to return after reranking (default 3).
    candidate_pool : int
        How many candidates to fetch from FAISS for the rerank step
        (default 60).  Clamped to the actual index size.
    batch_size : int
        Mini-batch size passed to the embedding model (default 1).
    history : list[dict] or None
        Optional conversation history.  Accepted for future
        context-aware retrieval; not yet used in scoring.

    Returns
    -------
    list[dict]
        Up to *top_k* hits, each with keys ``page``, ``chunk_id``,
        ``text``, and ``score`` (higher is better, range ~ [0, 1]).
    """
    import numpy as np

    # ---- resolve embedding model -----------------------------------------
    manifest = bundle.get("manifest", {})
    model_name = manifest.get("model_name", "all-MiniLM-L6-v2")
    model = load_model(model_name)

    # ---- embed the question ----------------------------------------------
    q_vec = embed_texts([question], model, batch_size=max(batch_size, 1))
    q_vec = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)

    # ---- semantic search (cosine similarity via inner product) ------------
    index = bundle["index"]
    chunks = bundle["chunks"]
    pool_size = min(candidate_pool, index.ntotal)
    scores, indices = index.search(q_vec, pool_size)

    # ---- build candidate hits with combined scores -----------------------
    q_kw = keyword_set(question)
    candidates: list[dict] = []

    for i in range(pool_size):
        idx = int(indices[0][i])
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        sem_score = float(scores[0][i])
        lex_score = _lexical_score(q_kw, keyword_set(chunk.get("text", "")))
        # 90 % semantic + 10 % lexical — simple, explainable blend
        combined = 0.9 * sem_score + 0.1 * lex_score
        candidates.append(
            {
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get("text", ""),
                "score": round(combined, 6),
            }
        )

    # ---- rerank by combined score, keep top-k ----------------------------
    candidates.sort(key=lambda h: h["score"], reverse=True)
    return candidates[:top_k]


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search a prepared document record for chunks relevant to *question*.

    Loads the saved FAISS index and chunk file from disk (using paths
    stored in *document*), runs semantic retrieval with light lexical
    reranking, and returns the top *k* hits.

    Parameters
    ----------
    question : str
        The user's question.
    document : dict
        A document record as returned by :func:`prepare_rag_document`.
        Must contain ``rag_index.artifact_root``, ``model_name``, and
        ``artifacts.index`` / ``artifacts.chunks`` relative paths.
    top_k : int
        Number of hits to return after reranking (default 3).
    candidate_pool : int
        How many candidates to fetch from FAISS for the rerank step
        (default 60).
    history : list[dict] or None
        Optional conversation history.  Accepted for future
        context-aware retrieval; not yet used in scoring.

    Returns
    -------
    list[dict]
        Up to *top_k* hits, each with keys ``page``, ``chunk_id``,
        ``text``, and ``score`` (higher is better).
    """
    artifact_root = Path(
        document.get("rag_index", {}).get("artifact_root", "artifacts/rag")
    )
    model_name = document.get("model_name", "all-MiniLM-L6-v2")

    # Resolve absolute artifact paths from the document record
    index_rel = document.get("artifacts", {}).get("index", "")
    chunks_rel = document.get("artifacts", {}).get("chunks", "")

    index_path = artifact_root / index_rel
    chunks_path = artifact_root / chunks_rel

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}")

    # Load from disk
    index = load_faiss_index(index_path)
    chunks_list = load_json(chunks_path)

    # Delegate to the in-memory search engine
    bundle = {
        "index": index,
        "chunks": chunks_list,
        "manifest": {"model_name": model_name},
    }
    return search_bundle(
        question=question,
        bundle=bundle,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=history,
    )


def split_sentences(text: str) -> list[str]:
    """Split *text* into candidate answer sentences.

    Breaks on sentence-ending punctuation (``.``, ``!``, ``?``) followed
    by whitespace or end-of-string.  Returns only non-empty, stripped
    sentences.
    """
    if not text or not text.strip():
        return []
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Pick the single best answer sentence from the retrieved *hits*.

    Each hit's text is split into sentences via :func:`split_sentences`.
    Every sentence is scored by **keyword density**: the square of
    overlapping question keywords divided by the sentence length in
    keywords.  This rewards short, targeted sentences that pack many
    question-relevant words and penalises long sentences that rephrase
    the question without delivering the answer.  The highest-scoring
    sentence across all hits is returned.  When the winning sentence
    comes from a hit that carries a ``page`` field, a page tag like
    ``" (page 5)"`` is appended.

    Returns an empty string when *hits* is empty or no sentence shares
    any keywords with the question.
    """
    if not hits:
        return ""

    q_kw = keyword_set(question)
    best_sent = ""
    best_score = -1.0
    best_page = None

    for hit in hits:
        for sent in split_sentences(hit.get("text", "")):
            sent_kw = keyword_set(sent)
            if not sent_kw:
                continue

            # Only sentences that share at least one keyword with the
            # question are relevant — skip unrelated ones entirely.
            overlap = len(q_kw & sent_kw)
            if overlap == 0:
                continue

            # Keyword density: overlap² / sentence length.
            #   - Squaring the overlap amplifies multi-keyword matches
            #     (3 hits → 9, 1 hit → 1 — 9× difference, not 3×).
            #   - Dividing by sentence length penalises long-winded
            #     sentences that mention the topic but never answer it.
            score = (overlap * overlap) / len(sent_kw)
            if score > best_score:
                best_score = score
                best_sent = sent
                best_page = hit.get("page")

    if not best_sent:
        return ""

    if best_page is not None:
        return f"{best_sent} (page {best_page})"
    return best_sent


# ---------------------------------------------------------------------------
# 14. Project-facing helpers
# ---------------------------------------------------------------------------


def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
    """Extract numeric PDF page citations from an answer string and optional hits.

    Parses page references like ``(page 5)``, ``[Page 5]``, ``(p. 5)``,
    or ``[p5]`` from the answer text.  When *hits* are provided their
    page numbers are also collected as additional citations.

    Returns a sorted list of unique page numbers.
    """
    pages: set[int] = set()

    # Extract page tags from the answer text
    if answer:
        for m in re.finditer(r"(?:page|p\.?)\s*(\d+)", answer, re.IGNORECASE):
            pages.add(int(m.group(1)))

    # Collect page numbers from hits
    if hits:
        for hit in hits:
            p = hit.get("page")
            if p is not None:
                pages.add(int(p))

    return sorted(pages)


def build_sources(hits: list[dict]) -> list[dict]:
    """Convert retrieval hits into frontend-friendly source objects.

    Each source carries *page*, *chunk_id*, *score*, and a *preview*
    (first ~150 characters of the chunk text, truncated at the nearest
    word boundary).  The *page* field lets the frontend render
    clickable page links.
    """
    sources: list[dict] = []
    for hit in hits:
        text = hit.get("text", "")
        if len(text) > 150:
            preview = text[:150].rsplit(" ", 1)[0] + "…"
        else:
            preview = text
        sources.append(
            {
                "page": hit.get("page"),
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("score"),
                "preview": preview,
            }
        )
    return sources


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """Answer one question against a prepared document record.

    Runs retrieval via :func:`search_document`, then either calls the
    LLM (when ``OPENROUTER_API_KEY`` is set in the environment) or
    falls back to local answer extraction via
    :func:`best_sentence_answer`.  LLM call failures also fall back
    gracefully.

    Parameters
    ----------
    document : dict
        A document record as returned by :func:`prepare_rag_document`.
    question : str
        The user's question.
    top_k : int
        Number of retrieval hits to use (default 3).
    candidate_pool : int
        How many candidates to fetch from FAISS for reranking (default 60).
    answer_model : str
        OpenRouter model id used for LLM answering (default
        ``"openrouter/free"``).

    Returns
    -------
    dict
        Three keys:

        * ``answer`` — the answer string (LLM or locally extracted)
        * ``citations`` — sorted list of unique page numbers
        * ``sources`` — frontend-ready source objects
    """
    import os

    # ---- retrieval -------------------------------------------------------
    hits = search_document(
        question=question,
        document=document,
        top_k=top_k,
        candidate_pool=candidate_pool,
        history=document.get("history"),
    )

    # ---- answer generation -----------------------------------------------
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        try:
            answer = _llm_answer_from_hits(
                question=question,
                hits=hits,
                api_key=api_key,
                model=answer_model,
            )
        except Exception:
            # LLM call failed (network, quota, …) → fall back to local
            answer = best_sentence_answer(question, hits)
    else:
        answer = best_sentence_answer(question, hits)

    # ---- citations & sources --------------------------------------------
    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {"answer": answer, "citations": citations, "sources": sources}


def _llm_answer_from_hits(
    question: str,
    hits: list[dict],
    api_key: str,
    model: str = "openrouter/free",
) -> str:
    """Call the OpenRouter LLM with retrieved chunks as context.

    Formats each hit as a ``### [Page X]`` block and sends them
    alongside *question* to the chat-completions endpoint.
    """
    import httpx
    from openai import OpenAI

    # Build PDF-like context from retrieved hits
    parts: list[str] = []
    for hit in hits:
        page = hit.get("page", "?")
        text = hit.get("text", "")
        parts.append(f"### [Page {page}]\n{text}")
    context = "\n\n".join(parts)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=httpx.Timeout(30.0, connect=10.0),
    )

    system_prompt = (
        "You answer messages only from the supplied PDF text. "
        "Cite factual claims with [Page X]. "
        "If the answer is not in the PDF, say that the document does not "
        "provide enough information. "
        "Never invent a page number. "
        "Format all math with $...$ for inline and $$...$$ for display LaTeX. "
        "Never use \\( ... \\) or \\[ ... \\] delimiters."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"PDF text:\n{context}\n\nmessage: {question}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def append_history(
    document: dict,
    question: str,
    result: dict,
) -> list[dict]:
    """Append a Q&A turn to the document's in-memory history.

    Adds an entry with *question*, *answer*, and *citations* to
    ``document["history"]`` and returns the updated history list.
    The *document* dict is mutated in place.

    Parameters
    ----------
    document : dict
        A document record as returned by :func:`prepare_rag_document`.
    question : str
        The user's question.
    result : dict
        The result dict returned by :func:`answer_document` (must
        contain at least ``answer`` and ``citations``).

    Returns
    -------
    list[dict]
        The updated history list (same object as ``document["history"]``).
    """
    entry = {
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
    document["history"].append(entry)
    return document["history"]


# ---------------------------------------------------------------------------
# 15. Retrieval evaluation helpers
# ---------------------------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Normalize text so that simple substring matching is more robust.

    Lowercases, strips surrounding whitespace, replaces common
    punctuation with spaces, and collapses repeated whitespace.
    The result is a clean, lowercased string where only words and
    single spaces remain — ideal for checking whether one piece of
    text contains another regardless of formatting quirks.
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Replace punctuation and special characters with a space so word
    # boundaries are preserved and concatenation artefacts are avoided.
    text = re.sub(r"[.,;:!?\"'()\[\]{}…–—/]", " ", text)
    # Collapse any run of whitespace into a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Return ``True`` when at least one gold answer appears in *text*.

    Both *text* and every candidate answer are first normalised with
    :func:`normalize_for_match`.  Then a simple substring check decides
    whether the answer is present.  This is deliberately straightforward:
    it rewards exact word-level overlap and its behaviour is easy to
    explain to someone learning about retrieval evaluation.
    """
    norm_text = normalize_for_match(text)
    if not norm_text:
        return False
    for answer in answers:
        norm_answer = normalize_for_match(answer)
        if norm_answer and norm_answer in norm_text:
            return True
    return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
) -> "pandas.DataFrame":
    """Run a simple end-to-end retrieval evaluation and return a table.

    For every question in *eval_set* the function:

    1. looks up the matching document by ``pdf_name``,
    2. retrieves the top *k* chunks via :func:`search_document`,
    3. extracts a local answer with :func:`best_sentence_answer`,
    4. checks whether the correct page was retrieved
       (``retrieval_hit``), and
    5. checks whether the extracted answer contains any of the
       gold answers (``answer_hit``).

    Parameters
    ----------
    eval_set : list[dict]
        Each record describes one test question and must provide:

        - ``question`` (str) — the question to ask
        - ``pdf_name`` (str) — which document to search (must be a
          key in *documents_by_name*)
        - ``answer_page`` (int | list[int], optional) — the PDF page (or pages)
          that contain the answer.  When omitted, ``retrieval_hit``
          will be ``None`` ("not evaluated").
        - ``answers`` (list[str]) — one or more acceptable gold
          answer strings

    documents_by_name : dict[str, dict]
        Mapping from ``pdf_name`` to prepared document records, as
        returned by :func:`prepare_rag_document`.
    top_k : int
        Number of retrieval hits to use (default 3).
    candidate_pool : int
        How many FAISS candidates to fetch before reranking (default 60).

    Returns
    -------
    pandas.DataFrame
        One row per question.  Columns:

        - ``question`` — the original question text
        - ``pdf_name`` — which document was searched
        - ``answer_page`` — the expected page(s) from the eval record
        - ``retrieved_pages`` — list of page numbers found by retrieval
        - ``local_answer`` — the best sentence extracted from hits
        - ``retrieval_hit`` — ``True`` when at least one expected page
          appears in the retrieved pages
        - ``answer_hit`` — ``True`` when the local answer contains any
          of the gold answers (checked via :func:`contains_any_answer`)
    """
    import pandas as pd  # type: ignore[import-untyped]

    rows: list[dict] = []

    for item in eval_set:
        question = item["question"]
        pdf_name = item["pdf_name"]
        answer_page = item.get("answer_page")
        gold_answers = item.get("answers", [])

        # Normalise answer_page to a set of ints so we can handle both
        # a single page number and a list of pages the same way.
        # When answer_page is missing we leave target_pages empty and
        # mark retrieval_hit as None ("not evaluated").
        if answer_page is None:
            target_pages: set[int] = set()
        elif isinstance(answer_page, (int, float)):
            target_pages = {int(answer_page)}
        else:
            target_pages = {int(p) for p in answer_page}

        # --- look up the document ----------------------------------------
        document = documents_by_name.get(pdf_name)
        if document is None:
            # PDF not found in the provided mapping — record a miss
            rows.append(
                {
                    "question": question,
                    "pdf_name": pdf_name,
                    "answer_page": answer_page,
                    "retrieved_pages": [],
                    "local_answer": "",
                    "retrieval_hit": False,
                    "answer_hit": False,
                }
            )
            continue

        # --- retrieval ---------------------------------------------------
        hits = search_document(
            question=question,
            document=document,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

        retrieved_pages = sorted(
            {h["page"] for h in hits if h.get("page") is not None}
        )
        retrieval_hit = (
            bool(target_pages & set(retrieved_pages))
            if target_pages
            else None  # answer_page was not provided — skip this metric
        )

        # --- local answer extraction -------------------------------------
        local_answer = best_sentence_answer(question, hits)
        answer_hit = (
            contains_any_answer(local_answer, gold_answers)
            if gold_answers
            else False
        )

        rows.append(
            {
                "question": question,
                "pdf_name": pdf_name,
                "answer_page": answer_page,
                "retrieved_pages": retrieved_pages,
                "local_answer": local_answer,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
            }
        )

    return pd.DataFrame(rows)

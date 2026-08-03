"""
RAG (Retrieval-Augmented Generation) helpers for SmartLearn.

Day 3 preparation:
  - clean extracted PDF text
  - extract pages without a hard page cap
  - save / load JSON artifacts
  - preview records as a simple table
  - embedding pipeline: model loading, text encoding, artifact persistence
"""

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

def extract_pages_for_rag(pdf_input: bytes | str | Path) -> list[dict]:
    """Read a PDF page by page, keeping original PDF page numbers.

    *pdf_input* may be raw PDF bytes or a file path (``str`` or ``Path``).
    Returns only pages whose cleaned text is non-empty.  No hard page
    limit — the caller decides how many pages to process.
    """
    if isinstance(pdf_input, (str, Path)):
        pdf_bytes = Path(pdf_input).read_bytes()
    else:
        pdf_bytes = pdf_input
    reader = PdfReader(BytesIO(pdf_bytes))

    records: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if cleaned:                         # drop empty pages
            records.append({"page": page_number, "text": cleaned})

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

    raise ValueError(
        f"Unknown chunk_mode: {chunk_mode!r}. "
        "Expected 'paragraph', 'character', or 'character_overlap'."
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
    * ``chunk_root``      — directory shared by chunks / embeddings / manifest
    """
    root = Path(artifact_root) / _safe_name(document_id)
    tag = model_tag(model_name)

    prefix = f"{chunk_mode}_{chunk_size}_{overlap}_"

    return {
        "raw_pages": root / "pages.json",
        "chunks": root / f"{prefix}chunks.json",
        "embeddings": root / f"{prefix}embeddings_{tag}.npy",
        "manifest": root / f"{prefix}manifest_{tag}.json",
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

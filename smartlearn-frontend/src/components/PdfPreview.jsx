import { API, CHAT_ID } from "../api";

export function getDocumentFileURL(page = 1) {
  return `${API}/documents/${encodeURIComponent(CHAT_ID)}/file#page=${page}`;
}

export default function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <div className="card pdf-placeholder">
        <p>Upload a PDF to preview it here.</p>
      </div>
    );
  }

  return (
    <div className="card pdf-preview">
      <iframe
        key={`${previewKey}-${activePage}`}
        src={getDocumentFileURL(activePage)}
        title="PDF Preview"
        className="pdf-frame"
      />
    </div>
  );
}

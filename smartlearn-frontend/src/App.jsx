import { useState } from "react";
import { uploadPDF } from "./api";
import PdfUploader from "./components/PdfUploader";
import PdfPreview from "./components/PdfPreview";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [uploadStatus, setUploadStatus] = useState("");
  const [error, setError] = useState("");
  const [previewKey, setPreviewKey] = useState(0);

  const busy = uploadStatus !== "";

  function handleJumpToPage(page) {
    setCurrentPage(page);
  }

  async function handleUpload() {
    if (!file || busy) return;
    setError("");
    setUploadStatus("Uploading…");
    try {
      const result = await uploadPDF(file);
      setUpload(result);
      setCurrentPage(1);
      setPreviewKey((k) => k + 1);
    } catch (e) {
      setError(e.message);
      setUpload(null);
    } finally {
      setUploadStatus("");
    }
  }

  return (
    <main>
      <h1>SmartLearn</h1>
      <p>Your AI-powered learning assistant.</p>

      <div className="workspace">
        <div className="pdf-column">
          <PdfPreview upload={upload} activePage={currentPage} previewKey={previewKey} />
        </div>

        <div className="chat-column">
          <PdfUploader
            file={file}
            upload={upload}
            status={uploadStatus}
            error={error}
            onFileChange={setFile}
            onUpload={handleUpload}
          />

          <ChatPanel
            key={previewKey}
            enabled={!!upload}
            disabled={busy}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      </div>
    </main>
  );
}

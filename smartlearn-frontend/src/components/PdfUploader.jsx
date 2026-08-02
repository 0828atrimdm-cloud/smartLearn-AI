export default function PdfUploader({ file, upload, status, error, onFileChange, onUpload }) {
  return (
    <div className="card">
      <form onSubmit={(e) => e.preventDefault()}>
        <div>
          <label htmlFor="pdf">PDF file</label>
          <input
            id="pdf"
            type="file"
            accept=".pdf"
            onChange={(e) => onFileChange(e.target.files[0] || null)}
          />
        </div>
        <button type="button" disabled={!file || status !== ""} onClick={onUpload}>
          Upload
        </button>
      </form>

      {status && <p className="status">{status}</p>}

      {error && <p role="alert">{error}</p>}

      {upload && (
        <div className="upload-meta">
          <p>Uploaded: {upload.filename}</p>
          <p>
            {upload.pages} pages, {upload.characters} characters
          </p>
        </div>
      )}
    </div>
  );
}

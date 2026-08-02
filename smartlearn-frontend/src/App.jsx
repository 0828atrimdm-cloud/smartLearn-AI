import { useState } from "react";
import { uploadPDF, askQuestion } from "./api";
import PdfUploader from "./components/PdfUploader";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  const busy = status !== "";

  async function handleUpload() {
    if (!file || busy) return;
    setError("");
    setStatus("Uploading…");
    try {
      const result = await uploadPDF(file);
      setUpload(result);
      setAnswer(null);
    } catch (e) {
      setError(e.message);
      setUpload(null);
    } finally {
      setStatus("");
    }
  }

  async function handleAsk() {
    if (!upload || !message.trim() || busy) return;
    setError("");
    setAnswer(null);
    setStatus("Asking…");
    try {
      const result = await askQuestion(message.trim());
      setAnswer(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setStatus("");
    }
  }

  return (
    <main>
      <h1>SmartLearn</h1>
      <p>Your AI-powered learning assistant.</p>

      <PdfUploader
        file={file}
        upload={upload}
        status={status}
        error={error}
        onFileChange={setFile}
        onUpload={handleUpload}
      />

      <ChatPanel
        message={message}
        answer={answer}
        status={status}
        error={error}
        onChangeMessage={setMessage}
        onAsk={handleAsk}
      />
    </main>
  );
}

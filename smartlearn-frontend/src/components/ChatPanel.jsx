import { useState, useRef, useEffect } from "react";
import { askQuestion } from "../api";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

/** Convert \(…\) / \[…\] LaTeX delimiters to $…$ / $$…$$ for remark-math v6 compatibility. */
function normalizeLatex(text) {
  if (!text) return text;
  return text
    .replace(/\\\[/g, "$$")
    .replace(/\\\]/g, "$$")
    .replace(/\\\(/g, "$")
    .replace(/\\\)/g, "$");
}

export default function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const listEndRef = useRef(null);

  useEffect(() => {
    listEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    if (!enabled || disabled || !message.trim() || loading) return;
    const userMsg = message.trim();
    setMessage("");
    setError("");
    setLoading(true);
    onBusy?.(true);

    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);

    try {
      const result = await askQuestion(userMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: result.answer,
          citations: result.citations || [],
          sources: result.sources || [],
        },
      ]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      onBusy?.(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const inputDisabled = !enabled || disabled;

  return (
    <div className="card chat-card">
      <div className="message-list">
        {messages.length === 0 && (
          <p className="chat-hint">Ask a question about the uploaded PDF.</p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.role === "assistant" ? (
              <>
                <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                  {normalizeLatex(msg.content)}
                </ReactMarkdown>
                {msg.citations && msg.citations.length > 0 && (
                  <div className="citations">
                    {msg.citations.map((page) => (
                      <button
                        key={page}
                        type="button"
                        className="citation-btn"
                        onClick={() => onJumpToPage(page)}
                      >
                        Page {page}
                      </button>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p>{msg.content}</p>
            )}
          </div>
        ))}
        {loading && <p className="status">Asking…</p>}
        {error && <p role="alert">{error}</p>}
        <div ref={listEndRef} />
      </div>

      <form onSubmit={(e) => e.preventDefault()}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={enabled ? "Ask a question…" : "Upload a PDF first"}
          disabled={inputDisabled}
        />
        <button
          type="button"
          disabled={inputDisabled || !message.trim() || loading}
          onClick={handleSend}
        >
          Send
        </button>
      </form>
    </div>
  );
}

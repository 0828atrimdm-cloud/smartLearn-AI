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

export default function ChatPanel({ message, answer, status, error, onChangeMessage, onAsk }) {
  return (
    <div className="card">
      <form onSubmit={(e) => e.preventDefault()}>
        <div>
          <label htmlFor="message">Question</label>
          <textarea
            id="message"
            value={message}
            onChange={(e) => onChangeMessage(e.target.value)}
          />
        </div>
        <button
          type="button"
          disabled={!message.trim() || status !== ""}
          onClick={onAsk}
        >
          Ask
        </button>
      </form>

      {status && <p className="status">{status}</p>}

      {error && <p role="alert">{error}</p>}

      {answer && (
        <div className="answer">
          <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
            {normalizeLatex(answer.answer)}
          </ReactMarkdown>
          {answer.citations && answer.citations.length > 0 && (
            <div className="citations">
              {answer.citations.map((page) => (
                <span key={page}>Page {page}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

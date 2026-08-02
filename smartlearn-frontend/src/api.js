const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const CHAT_ID = "day2-demo";

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  const url = `${API}/upload?chat_id=${encodeURIComponent(CHAT_ID)}`;

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${response.status})`);
  }

  return response.json();
}

export async function askQuestion(message) {
  const response = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, chat_id: CHAT_ID }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Chat failed (${response.status})`);
  }

  return response.json();
}

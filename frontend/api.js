// Backend base URL + stream parsing.
// Served by FastAPI itself (run.sh) -> same-origin; Vite dev -> configured API.
export const API_BASE =
  window.location.port === "5173" || window.location.port === "3000"
    ? (import.meta.env.VITE_API_BASE || "http://localhost:8787")
    : "";

export async function getStatus() {
  try {
    return await (await fetch(`${API_BASE}/api/status`)).json();
  } catch {
    return { models_loaded: false, reasoning_model: "loading", documents: 0, assistant_name: "Assistant" };
  }
}

export async function listFiles() {
  try {
    return (await (await fetch(`${API_BASE}/api/files`)).json()).files || [];
  } catch {
    return [];
  }
}

export async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API_BASE}/api/files`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || "upload failed");
  return await r.json();
}

export async function deleteFile(name) {
  const r = await fetch(`${API_BASE}/api/files/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!r.ok) throw new Error("delete failed");
  return await r.json();
}

// Streams the agent run. Handlers: onToken, onThinkingDelta, onThinkingEnd,
// onToolCall, onToolResult, onSources, onDone, onError.
export async function streamChat(message, history, handlers) {
  const resp = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!resp.ok || !resp.body) {
    handlers.onError?.(`server error (${resp.status})`);
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      let evt;
      try { evt = JSON.parse(line.slice(5).trim()); } catch { continue; }
      const h = handlers;
      if (evt.type === "token") h.onToken?.(evt.content);
      else if (evt.type === "thinking_delta") h.onThinkingDelta?.(evt.content);
      else if (evt.type === "thinking_end") h.onThinkingEnd?.();
      else if (evt.type === "tool_call") h.onToolCall?.(evt);
      else if (evt.type === "tool_result") h.onToolResult?.(evt);
      else if (evt.type === "sources") h.onSources?.(evt.data);
      else if (evt.type === "done") h.onDone?.();
      else if (evt.type === "error") h.onError?.(evt.content);
    }
  }
  handlers.onDone?.();
}

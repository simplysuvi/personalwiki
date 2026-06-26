import React, { useState, useEffect, useRef, useCallback } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import ToolCallTrace from "./components/ToolCallTrace.jsx";
import FilesPanel from "./components/FilesPanel.jsx";
import ThemeToggle from "./components/ThemeToggle.jsx";
import { getStatus, listFiles, streamChat } from "./api.js";

function initialTheme() {
  const saved = localStorage.getItem("wiki-theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [trace, setTrace] = useState([]);
  const [sources, setSources] = useState([]);
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState({ models_loaded: false, reasoning_model: "loading", documents: 0, assistant_name: "Assistant" });
  const [input, setInput] = useState("");
  const [theme, setTheme] = useState(initialTheme);
  const historyRef = useRef([]);
  const inputRef = useRef(null);
  const assistantName = status.assistant_name || "Assistant";

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("wiki-theme", theme);
  }, [theme]);

  const refresh = useCallback(async () => {
    setStatus(await getStatus());
    setFiles(await listFiles());
  }, []);
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const autosize = (el) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 160) + "px";
  };

  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";
    setBusy(true);
    setTrace([]);
    setSources([]);
    const history = [...historyRef.current];
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "" }]);

    let acc = "";
    const setLast = (content) =>
      setMessages((m) => { const c = m.slice(); c[c.length - 1] = { role: "assistant", content }; return c; });

    await streamChat(q, history, {
      onThinkingDelta: (text) => setTrace((t) => {
        const c = t.slice(); const last = c[c.length - 1];
        if (last && last.kind === "thinking" && !last.done) c[c.length - 1] = { ...last, text: last.text + text };
        else c.push({ kind: "thinking", text, done: false });
        return c;
      }),
      onThinkingEnd: () => setTrace((t) => {
        const c = t.slice(); const last = c[c.length - 1];
        if (last && last.kind === "thinking" && !last.done) c[c.length - 1] = { ...last, done: true };
        return c;
      }),
      onToolCall: (ev) => setTrace((t) => [...t, { kind: "tool", tool: ev.tool, args: ev.args, done: false, preview: "" }]),
      onToolResult: (ev) => setTrace((t) => {
        const c = t.slice();
        for (let i = c.length - 1; i >= 0; i--) {
          if (c[i].kind === "tool" && c[i].tool === ev.tool && !c[i].done) { c[i] = { ...c[i], done: true, preview: ev.preview }; break; }
        }
        return c;
      }),
      onToken: (tok) => { acc += tok; setLast(acc); },
      onSources: (s) => setSources(s || []),
      onError: (msg) => { acc += `\n\n_[error: ${msg}]_`; setLast(acc); },
      onDone: () => {},
    });

    historyRef.current = [...history, { role: "user", content: q }, { role: "assistant", content: acc }];
    setBusy(false);
  }, [input, busy]);

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo-dot" />
          <span className="wordmark">{assistantName.toLowerCase()}</span>
        </div>
        <div className="header-right">
          <span className={`model-indicator ${status.models_loaded ? "ok" : "down"}`}>
            <span className="dot" />
            {status.reasoning_model} - {status.documents} docs
          </span>
          <ThemeToggle theme={theme} onToggle={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} />
        </div>
      </header>

      <main className="main">
        <section className="chat-col">
          <ChatWindow messages={messages} busy={busy} assistantName={assistantName} />
          <div className="input-bar">
            <div className="input-wrap">
              <textarea ref={inputRef} className="input" rows={1} placeholder={`Ask ${assistantName} anything...`}
                value={input} disabled={busy}
                onChange={(e) => { setInput(e.target.value); autosize(e.target); }}
                onKeyDown={onKey} />
              <button className="send" onClick={send} disabled={busy || !input.trim()}>
                {busy ? <span className="send-dots"><i>.</i><i>.</i><i>.</i></span> : "Ask"}
              </button>
            </div>
          </div>
        </section>
        <aside className="side-col">
          <FilesPanel files={files} onChange={refresh} />
          <ToolCallTrace trace={trace} sources={sources} running={busy} />
        </aside>
      </main>
    </div>
  );
}

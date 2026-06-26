import React, { useRef, useState } from "react";
import { uploadFile, deleteFile } from "../api.js";

function fmtSize(b) {
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return Math.round(b / 1e3) + " KB";
  return b + " B";
}

export default function FilesPanel({ files, onChange }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [drag, setDrag] = useState(false);

  const upload = async (fileList) => {
    const arr = Array.from(fileList || []);
    if (!arr.length) return;
    setBusy(true);
    for (const f of arr) {
      setNote(`converting ${f.name}...`);
      try {
        await uploadFile(f);
      } catch (e) {
        setNote(`Error: ${f.name}: ${e.message}`);
        await new Promise((r) => setTimeout(r, 2500));
      }
    }
    setNote("");
    setBusy(false);
    onChange?.();
  };

  const remove = async (name) => {
    if (!confirm(`Remove "${name}" from the knowledge base?`)) return;
    await deleteFile(name);
    onChange?.();
  };

  return (
    <div className="files-panel"
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files); }}
    >
      <div className="files-head">
        <span>Knowledge base</span>
        <button className="add-btn" onClick={() => inputRef.current?.click()} disabled={busy}>
          {busy ? "..." : "+ Add"}
        </button>
        <input ref={inputRef} type="file" multiple accept=".pdf,.csv,.md,.markdown,.txt,.json"
          style={{ display: "none" }} onChange={(e) => { upload(e.target.files); e.target.value = ""; }} />
      </div>

      {note && <div className="files-note">{note}</div>}

      <div className={`files-drop ${drag ? "over" : ""}`}>
        {files.length === 0 && !busy && (
          <div className="files-empty">Drop documents here, or click "+ Add".<br />PDF - CSV - MD - TXT - JSON</div>
        )}
        {files.map((f) => (
          <div className="file-row" key={f.name}>
            <div className="file-main">
              <div className="file-name" title={f.name}>{f.name}</div>
              <div className="file-meta">
                {fmtSize(f.bytes)} - {f.method}
                {f.indexed ? <span className="ok"> - indexed</span> : <span className="warn"> - no text</span>}
              </div>
            </div>
            <button className="file-del" onClick={() => remove(f.name)} title="Remove">Remove</button>
          </div>
        ))}
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef } from "react";

function argSummary(tool, args) {
  if (!args) return "";
  if (args.path) return args.path;
  if (args.query) return `"${args.query}"`;
  if (args.contains || args.date_from)
    return [args.contains, args.date_from && `${args.date_from}…`].filter(Boolean).join(" ");
  return "";
}

function ThinkingRow({ text, live }) {
  const [open, setOpen] = useState(false);
  // while streaming: always show the full growing text with a caret;
  // once done: collapse to a preview, click to expand
  const expanded = live || open;
  const short = text.length > 140 ? text.slice(0, 140) + "…" : text;
  return (
    <div className={`trace-row thinking ${live ? "live" : ""}`}>
      <button className="trace-think" onClick={() => !live && setOpen(!open)} title="Model reasoning">
        <span className="think-icon">✦</span>
        <span className="think-text">
          {expanded ? text : short}
          {live && <span className="caret">▋</span>}
        </span>
      </button>
    </div>
  );
}

export default function ToolCallTrace({ trace, sources, running }) {
  const [openIdx, setOpenIdx] = useState(null);
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [trace]);

  return (
    <div className="trace-panel">
      <div className="trace-head">Agent trace</div>
      {trace.length === 0 && !running && (
        <div className="trace-empty">
          The model's reasoning and file reads appear here in real time.
        </div>
      )}
      {trace.map((t, i) =>
        t.kind === "thinking" ? (
          <ThinkingRow key={i} text={t.text} live={!t.done} />
        ) : (
          <div key={i} className={`trace-row ${t.done ? "done" : "pending"}`}>
            <button
              className="trace-line"
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
              title={t.preview ? "Click to view result preview" : ""}
            >
              <span className="trace-arrow">→</span>
              <span className="trace-tool">{t.tool}</span>
              <span className="trace-arg">{argSummary(t.tool, t.args)}</span>
              <span className="trace-check">{t.done ? "✓" : "…"}</span>
            </button>
            {openIdx === i && t.preview && (
              <pre className="trace-preview">{t.preview}</pre>
            )}
          </div>
        )
      )}
      {running && trace.length > 0 && <div className="trace-live">working<span className="dots"><i>.</i><i>.</i><i>.</i></span></div>}
      {sources.length > 0 && (
        <div className="trace-sources">
          <div className="trace-head">Sources</div>
          {sources.map((s, i) => (
            <div key={i} className="trace-source">{s}</div>
          ))}
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}

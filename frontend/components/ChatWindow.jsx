import React, { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble.jsx";

export default function ChatWindow({ messages, busy, assistantName = "Assistant" }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <p>{assistantName} is ready.</p>
          <p className="hint">
            Ask about uploaded documents, compare details across files, or talk through an idea.
            <br />
            "Summarize this lease" - "Find renewal dates" - "What changed between these files?"
          </p>
        </div>
      )}
      {messages.map((m, i) => (
        <MessageBubble
          key={i}
          role={m.role}
          content={m.content}
          pending={busy && i === messages.length - 1 && m.role === "assistant"}
        />
      ))}
      <div ref={endRef} />
    </div>
  );
}

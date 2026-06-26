import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function MessageBubble({ role, content, pending }) {
  const isUser = role === "user";
  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"}`}>
        {isUser ? (
          content
        ) : (
          <div className="markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // open any links the model emits in a new tab
                a: ({ node, ...props }) => (
                  <a target="_blank" rel="noreferrer" {...props} />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        )}
        {pending && <span className="caret">▋</span>}
      </div>
    </div>
  );
}

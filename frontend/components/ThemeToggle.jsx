import React from "react";

export default function ThemeToggle({ theme, onToggle }) {
  const dark = theme === "dark";
  return (
    <button
      className="theme-toggle"
      onClick={onToggle}
      title={dark ? "Switch to light" : "Switch to dark"}
      aria-label="Toggle color theme"
    >
      {dark ? "☀" : "☾"}
    </button>
  );
}

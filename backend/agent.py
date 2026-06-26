"""
agent.py — the core agentic loop.

Given a question, the agent repeatedly calls the LLM with the five tool schemas.
The model decides which files to read; tool results are fed back as observations.
The loop ends when the model produces a plain answer (or MAX_AGENT_ITERATIONS).

run_agent() is a *generator of event dicts*, matching the SSE protocol:
  {"type": "tool_call",   "tool": str, "args": dict}
  {"type": "tool_result", "tool": str, "preview": str}
  {"type": "token",       "content": str}
  {"type": "sources",     "data": [paths]}
  {"type": "done"}
"""
from __future__ import annotations
import datetime
import json
import queue
import re
import threading

import config
import models
from tools import TOOLS, TOOL_SCHEMAS

_litert_lock = threading.Lock()  # the LiteRT engine is not concurrency-safe


def system_prompt() -> str:
    now = datetime.datetime.now().astimezone()
    stamp = now.strftime("%A, %Y-%m-%d, %H:%M %Z").strip()
    # The document list is injected up-front so the model never has to *decide*
    # to look at the map — source selection was the most failure-prone step.
    from tools import list_documents
    doc_map = list_documents()
    return (
        f"KNOWLEDGE BASE — {config.USER_NAME}'s documents (each converted to markdown). "
        "Consult this list FIRST to pick the right document for any personal question:\n"
        f"{doc_map}\n\n" +
        f"You are {config.ASSISTANT_NAME} — {config.USER_NAME}'s personal AI assistant: warm, "
        "direct, and genuinely helpful, like a sharp friend who knows his life inside out.\n\n"
        "Tools over the knowledge base:\n"
        "- list_documents — see everything in the KB (shown above).\n"
        "- read_document(name[, section, grep]) — read a document's text; use section/grep "
        "to pull just the relevant part of a long one.\n"
        "- search(query) — keyword search across all documents when unsure which holds a fact.\n"
        "- query_csv(name, …) — filter rows of a CSV document (dates, amounts, merchants).\n\n"
        "How to answer questions about his data:\n"
        "- Pick the most relevant document from the list and read it. If unsure which, search first.\n"
        "- For a specific line-item (one paycheck's tax, one transaction), open that exact "
        "document (names often contain dates) and quote the number.\n"
        "- Act — never ask permission to use a tool, never end by naming a document you "
        "haven't read. If a result is empty or errors, adjust and try the next-best document "
        "before concluding 'not found'.\n"
        "- Never guess personal facts; cite the document(s) your answer came from.\n"
        "- Be direct, clean Markdown. General questions (not about his data) need no tools "
        "— just answer naturally.\n\n"
        f"The current date and time is {stamp}. Treat this as 'now' when reasoning about "
        "deadlines, expiries, ages, or how far away dated events are."
    )


def build_messages(question: str, history: list[dict] | None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt()}]
    for t in (history or [])[-8:]:
        if t.get("role") in ("user", "assistant") and t.get("content"):
            msgs.append({"role": t["role"], "content": t["content"]})
    msgs.append({"role": "user", "content": question})
    return msgs


_CITING_TOOLS = ("read_document", "query_csv")


def extract_cited_sources(messages: list[dict]) -> list[str]:
    """Unique document names from read_document / query_csv calls, in order."""
    seen, out = set(), []
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if fn.get("name") in _CITING_TOOLS:
                try:
                    nm = json.loads(fn.get("arguments") or "{}").get("name")
                except json.JSONDecodeError:
                    nm = None
                if nm and nm not in seen:
                    seen.add(nm)
                    out.append(nm)
    return out


def _balanced_json_objects(text: str):
    """Yield each balanced {...} substring in text (string-aware brace matching)."""
    i = 0
    while True:
        start = text.find("{", i)
        if start == -1:
            return
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:j + 1]
                    break
        else:
            return
        i = start + 1


_DSL_CALL_RE = re.compile(r"(?:<\|tool_call>)?call:([\w\-]+)\{(.*?)(?:\}\s*(?:<tool_call\|>)?\s*$|\}<tool_call\|>)", re.S)


def _parse_dsl_args(body: str) -> dict:
    """Parse gemma-4 DSL argument bodies: key:<|"|>str<|"|>, n:42, flag:true, x:null."""
    args, i, n = {}, 0, len(body)
    while i < n:
        # key
        m = re.match(r"\s*([\w\-]+)\s*:", body[i:])
        if not m:
            break
        key = m.group(1)
        i += m.end()
        rest = body[i:]
        if rest.startswith('<|"|>'):                      # quoted string value
            end = rest.find('<|"|>', 5)
            if end == -1:
                args[key] = rest[5:]
                break
            args[key] = rest[5:end]
            i += end + 5
        else:                                             # bare scalar until top-level comma
            m2 = re.match(r"\s*([^,]*)", rest)
            raw = (m2.group(1) if m2 else "").strip()
            try:
                args[key] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                args[key] = raw
            i += m2.end() if m2 else len(rest)
        m3 = re.match(r"\s*,", body[i:])
        i += m3.end() if m3 else 0
        if not m3:
            break
    return args


def _parse_textual_tool_call(content: str):
    """
    Fallback parser for tool calls the runtime didn't surface natively.
    Recognizes two formats:
      1. gemma-4 DSL:  <|tool_call>call:read_document{name:<|"|>x.pdf<|"|>}<tool_call|>
      2. JSON object:  {"tool": "read_document", "args": {"name": "..."}}
    Returns (name, args) or None.
    """
    if not content:
        return None
    m = _DSL_CALL_RE.search(content)
    if m and m.group(1) in TOOLS:
        return m.group(1), _parse_dsl_args(m.group(2))
    for candidate in _balanced_json_objects(content):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        args = obj.get("args") or obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if name in TOOLS and isinstance(args, dict):
            return name, args
    return None


_THOUGHT_RE = re.compile(r"<\|channel>(?:thought|analysis)\b\n?.*?(?:<channel\|>|$)", re.S)


def _strip_channels(text: str) -> str:
    """Drop gemma-4 reasoning blocks (<|channel>thought ... <channel|>) but keep
    any other channel CONTENT (only the markers are removed), so a channel-wrapped
    final answer survives."""
    text = _THOUGHT_RE.sub("", text)
    return re.sub(r"<\|?(?:turn|channel)[^>]*\|?>?|<tool_call\|>", "", text).strip()


def _stream_text(text: str):
    """Yield token events from already-generated text (word-chunked)."""
    for piece in re.findall(r"\S+\s*", text):
        yield {"type": "token", "content": piece}


# ── context budgeting for the llamacpp loop ─────────────────────────────────
def _evict_old_results(messages: list[dict]):
    """
    Replace all but the most recent KEEP_FULL_TOOL_RESULTS tool results with a
    one-line stub. The model's own reasoning (fed back on assistant turns)
    carries its conclusions forward, and any file can be re-read on demand —
    so old raw dumps are pure context waste.
    """
    keep = max(0, config.KEEP_FULL_TOOL_RESULTS - 1)  # the result about to be
    # appended counts as the most recent one
    tool_idxs = [i for i, m in enumerate(messages)
                 if m.get("role") == "tool" and not m.get("_evicted")]
    for i in tool_idxs[:-keep] if keep else tool_idxs:
        m = messages[i]
        target = ""
        prev = messages[i - 1] if i > 0 else {}
        for tc in prev.get("tool_calls") or []:
            a = tc.get("function", {}).get("arguments")
            if isinstance(a, dict):
                target = a.get("name") or a.get("query") or ""
        m["content"] = (f"[{m.get('name', 'tool')}('{target}') was read earlier — "
                        "result evicted to save context; call the tool again if needed]")
        m["_evicted"] = True
def _llm_ntok(llm, text: str) -> int:
    try:
        return len(llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
    except Exception:
        return max(1, len(text) // 4)


def _msg_tokens(llm, m: dict) -> int:
    n = 16  # per-message template overhead
    if m.get("content"):
        n += _llm_ntok(llm, str(m["content"]))
    if m.get("reasoning"):
        n += _llm_ntok(llm, str(m["reasoning"]))
    for tc in m.get("tool_calls") or []:
        n += _llm_ntok(llm, json.dumps(tc.get("function", {})))
    return n


def _fit_messages(llm, messages: list[dict], budget: int) -> list[dict]:
    """Drop the oldest non-system messages until the conversation fits the budget.
    The system prompt (index 0) and the most recent 3 messages are always kept."""
    total = sum(_msg_tokens(llm, m) for m in messages)
    while total > budget and len(messages) > 4:
        removed = messages.pop(1)
        total -= _msg_tokens(llm, removed)
    return messages


def _truncate_to_tokens(llm, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return "[result omitted — context budget exhausted]"
    try:
        ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        if len(ids) <= max_tokens:
            return text
        return (llm.detokenize(ids[:max_tokens]).decode("utf-8", "ignore")
                + "\n[... truncated to fit context]")
    except Exception:
        keep = max_tokens * 4
        return text if len(text) <= keep else text[:keep] + "\n[... truncated to fit context]"


# ── live streaming for the native gemma-4 template ──────────────────────────
_MARKERS = ("<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>")
_HOLDBACK = max(len(m) for m in _MARKERS)


def _stream_native_turn(llm, kwargs):
    """
    Run ONE generation turn with stream=True, classifying tokens live:
      <|channel>thought ...   -> {"type": "thinking_delta"} events as they arrive
      plain text              -> {"type": "token"} events (the final answer)
      <|tool_call>call:...    -> collected silently for the tool dispatcher
    Markers can split across chunks, so a small holdback tail is kept unemitted
    until it can't be a marker prefix. Returns (via generator return value):
      {"answer": str, "thought": str, "tool_body": str | None}
    """
    state = "text"            # text | channel_name | think | tool | tool_done
    buf = ""
    answer_parts: list[str] = []
    thoughts: list[str] = []
    cur_thought = ""
    tool_body: str | None = None

    def classify(seg):
        """Route a confirmed content segment to its destination; yield UI events."""
        nonlocal cur_thought, tool_body
        if not seg:
            return
        if state == "think":
            cur_thought += seg
            yield {"type": "thinking_delta", "content": seg}
        elif state == "text":
            answer_parts.append(seg)
            yield {"type": "token", "content": seg}
        elif state == "tool":
            tool_body = (tool_body or "") + seg

    def close_thought():
        nonlocal cur_thought
        if cur_thought.strip():
            thoughts.append(cur_thought.strip())
        cur_thought = ""
        yield {"type": "thinking_end"}

    def process():
        """Consume buf as far as possible; yield events."""
        nonlocal buf, state, tool_body
        while True:
            if state == "channel_name":
                nl = buf.find("\n")
                if nl == -1:
                    return
                name = buf[:nl].strip()
                buf = buf[nl + 1:]
                state = "think" if name in ("thought", "analysis") else "text"
                continue
            # earliest marker in buf?
            idx, mark = None, None
            for m in _MARKERS:
                i = buf.find(m)
                if i != -1 and (idx is None or i < idx):
                    idx, mark = i, m
            if mark is None:
                # no full marker — emit all but a holdback tail (a marker prefix
                # might be split across chunks)
                if state in ("text", "think") and len(buf) > _HOLDBACK:
                    seg, buf = buf[:-_HOLDBACK], buf[-_HOLDBACK:]
                    yield from classify(seg)
                return
            yield from classify(buf[:idx])
            buf = buf[idx + len(mark):]
            if mark == "<|channel>":
                if state == "think":
                    yield from close_thought()
                state = "channel_name"
            elif mark == "<channel|>":
                if state == "think":
                    yield from close_thought()
                state = "text"
            elif mark == "<|tool_call>":
                if state == "think":
                    yield from close_thought()
                state = "tool"
                tool_body = tool_body or ""
            elif mark == "<tool_call|>":
                if state == "tool":
                    state = "tool_done"

    stream = llm.create_chat_completion(stream=True, **kwargs)
    for chunk in stream:
        delta = (chunk["choices"][0].get("delta") or {}).get("content") or ""
        if not delta:
            continue
        buf += delta
        yield from process()

    # end of stream — flush everything that's left
    if state == "channel_name":
        state = "text"
    if state in ("text", "think"):
        yield from classify(buf)
    elif state == "tool":
        tool_body = (tool_body or "") + buf
    if state == "think" or cur_thought.strip():
        yield from close_thought()

    return {"answer": "".join(answer_parts).strip(),
            "thought": "\n\n".join(thoughts),
            "tool_body": tool_body}


# ════════════════════════════════════════════════════════════════════════════
# LiteRT backend — native tool calling (automatic_tool_calling + event handler)
# ════════════════════════════════════════════════════════════════════════════
def _litert_history(history):
    msgs = []
    for t in (history or [])[-8:]:
        role = t.get("role")
        if role in ("user", "assistant") and t.get("content"):
            role = "model" if role == "assistant" else role
            msgs.append({"role": role, "content": [{"type": "text", "text": t["content"]}]})
    return msgs


def run_agent_litert(question: str, history: list[dict] | None = None):
    """Agent loop on the LiteRT-LM engine. The runtime executes tools itself;
    our ToolEventHandler streams trace events and enforces the iteration cap."""
    import litert_lm

    engine = models.get_litert_engine()
    q: queue.Queue = queue.Queue()
    DONE = object()
    sources: list[str] = []

    def ntok(text: str) -> int:
        try:
            return len(engine.tokenize(text))
        except Exception:
            return max(1, len(text) // 4)

    # token budget — the E2B engine hard-aborts (SIGABRT) past its window, so
    # tool results are truncated to fit and tool calls denied when budget is gone.
    limit = config.LITERT_CONTEXT_TOKENS - config.LITERT_OUTPUT_RESERVE
    used = ntok(system_prompt()) + ntok(question) + 64
    for t in (history or [])[-8:]:
        used += ntok(t.get("content", "")) + 8
    state = {"calls": 0, "last_tool": "?", "used": used}

    class TraceHandler(litert_lm.interfaces.ToolEventHandler):
        def approve_tool_call(self, tool_call):
            fn = tool_call.get("function", {}) or {}
            name = fn.get("name", "?")
            args = fn.get("args") or fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if state["calls"] >= config.MAX_AGENT_ITERATIONS + 2:
                return False  # absolute backstop
            if state["calls"] >= config.MAX_AGENT_ITERATIONS or state["used"] >= limit - 350:
                state["starve"] = True  # approve, but replace result with a stop order
            state["calls"] += 1
            state["last_tool"] = name
            q.put({"type": "tool_call", "tool": name, "args": args})
            doc = args.get("name") if isinstance(args, dict) else None
            if name in _CITING_TOOLS and doc and doc not in sources:
                sources.append(doc)
            return True

        def process_tool_response(self, tool_response):
            text = str(tool_response)
            # if the tool fuzzy-resolved the path, record the real one in sources
            m = re.match(r"\[resolved '([^']+)' -> '([^']+)'\]", text)
            if m and m.group(1) in sources:
                sources[sources.index(m.group(1))] = m.group(2)
            q.put({"type": "tool_result", "tool": state["last_tool"],
                   "preview": text[:120]})
            if state.get("starve"):
                return ("[Context budget exhausted. STOP calling tools. Give your final "
                        "answer NOW using what you have already read.]")
            # cap every result so multi-step flows fit the small window
            cap = config.LITERT_TOOL_RESULT_MAX_CHARS
            if len(text) > cap:
                text = text[:cap] + "\n[... truncated — full file is longer]"
            # and never exceed the remaining budget
            remaining = limit - state["used"] - 64
            if ntok(text) > remaining:
                keep_chars = max(200, remaining * 4)
                text = text[:keep_chars] + "\n[... truncated to fit context]"
            state["used"] += ntok(text) + 16
            return text

    def work():
        try:
            with _litert_lock:
                with engine.create_conversation(
                    system_message=system_prompt(),
                    messages=_litert_history(history),
                    tools=list(TOOLS.values()),
                    tool_event_handler=TraceHandler(),
                    automatic_tool_calling=True,
                ) as conv:
                    got_text = False
                    for chunk in conv.send_message_async(question):
                        for item in chunk.get("content", []):
                            if item.get("type") == "text" and item.get("text"):
                                got_text = True
                                q.put({"type": "token", "content": item["text"]})
                    if not got_text:
                        # loop ended on a denied/silent tool turn — force the answer
                        for chunk in conv.send_message_async(
                            "Give your final answer now from what you already read. "
                            "Do not call any tools."
                        ):
                            for item in chunk.get("content", []):
                                if item.get("type") == "text" and item.get("text"):
                                    q.put({"type": "token", "content": item["text"]})
        except Exception as e:
            q.put({"type": "error", "content": str(e)})
        finally:
            q.put(DONE)

    threading.Thread(target=work, daemon=True).start()
    while True:
        ev = q.get()
        if ev is DONE:
            break
        yield ev
    yield {"type": "sources", "data": sources}
    yield {"type": "done"}


def run_agent(question: str, history: list[dict] | None = None):
    """Dispatch to the configured backend."""
    if config.LLM_BACKEND == "litert":
        yield from run_agent_litert(question, history)
        return
    yield from run_agent_llamacpp(question, history)


def _tool_protocol_block() -> str:
    """Textual tool-calling protocol for chat templates without native tool support."""
    lines = ["\n\nTOOLS — to use one, reply with ONLY this JSON (no other text):",
             '{"tool": "<name>", "args": {...}}',
             "After the result comes back, either call another tool or write your final answer.",
             "Available tools:"]
    for s in TOOL_SCHEMAS:
        fn = s["function"]
        props = fn["parameters"].get("properties", {})
        args = ", ".join(props.keys()) or "none"
        lines.append(f"- {fn['name']}({args}): {fn['description']}")
    return "\n".join(lines)


def run_agent_llamacpp(question: str, history: list[dict] | None = None):
    """Synchronous generator of SSE event dicts (llama-cpp is blocking)."""
    llm = models.get_llm()
    native_template = models.using_custom_template()
    messages = build_messages(question, history)
    if not native_template:
        # no tool-aware template — teach the model our JSON protocol instead
        messages[0]["content"] += _tool_protocol_block()
    iterations = 0
    last_content = ""
    cited: list[str] = []

    while iterations < config.MAX_AGENT_ITERATIONS:
        # ── native template: TRUE streaming (thinking + answer live) ─────────
        if native_template:
            budget = config.N_CTX - config.CONTEXT_RESERVE
            messages = _fit_messages(llm, messages, budget)
            used = sum(_msg_tokens(llm, m) for m in messages)
            if used > budget - 600 and iterations > 0:
                # almost full — stop tooling, force the answer from what we have
                messages.append({"role": "user", "content":
                                 "Context is nearly full. STOP calling tools and give "
                                 "your final answer now from what you already read."})
            kwargs = dict(messages=messages, temperature=0.3, tools=TOOL_SCHEMAS,
                          stop=["<tool_call|>"])
            try:
                turn = yield from _stream_native_turn(llm, kwargs)
            except Exception as e:
                if "context window" not in str(e) and "exceed" not in str(e).lower():
                    raise
                # backstop: prune hard and force a final answer
                messages = [messages[0]] + messages[-3:]
                messages.append({"role": "user", "content":
                                 "Answer the question now from what you already read. "
                                 "Do not call tools."})
                turn = yield from _stream_native_turn(
                    llm, dict(messages=messages, temperature=0.3))

            if turn["tool_body"] is not None:
                parsed = _parse_textual_tool_call("call:" + turn["tool_body"]
                                                  if not turn["tool_body"].lstrip().startswith("call:")
                                                  else turn["tool_body"])
                if not parsed:
                    messages.append({"role": "user", "content":
                                     "Your tool call was malformed. Use the exact DSL "
                                     "format and retry, or answer from what you have."})
                    iterations += 1
                    continue
                name, args = parsed
                yield {"type": "tool_call", "tool": name, "args": args}
                fn = TOOLS.get(name)
                try:
                    result = fn(**args) if fn else f"Unknown tool: {name}. Available: {', '.join(TOOLS)}"
                except TypeError as e:
                    result = f"Bad arguments for {name}: {e}"
                except Exception as e:
                    result = f"Tool {name} failed: {e}"
                doc = args.get("name") if isinstance(args, dict) else None
                if name in _CITING_TOOLS and doc:
                    m = re.match(r"\[resolved '[^']+' -> '([^']+)'\]", result)
                    real = m.group(1) if m else doc
                    if real not in cited:
                        cited.append(real)
                yield {"type": "tool_result", "tool": name, "preview": result[:120]}
                amsg = {"role": "assistant", "content": None,
                        "tool_calls": [{"id": f"call_{iterations}", "type": "function",
                                        "function": {"name": name, "arguments": args}}]}
                if turn["thought"]:
                    # cap fed-back reasoning (it compounds fast); keep the tail —
                    # conclusions live at the end of a thought
                    amsg["reasoning"] = turn["thought"][-config.REASONING_FEEDBACK_CHARS:]
                messages.append(amsg)
                # evict stale tool dumps first, then truncate the new result to
                # the REMAINING budget
                _evict_old_results(messages)
                used_now = sum(_msg_tokens(llm, m) for m in messages)
                remaining = (config.N_CTX - config.CONTEXT_RESERVE) - used_now - 64
                result_fed = _truncate_to_tokens(llm, result, remaining)
                messages.append({"role": "tool", "name": name,
                                 "tool_call_id": f"call_{iterations}", "content": result_fed})
                iterations += 1
                continue

            # no tool call — the answer already streamed live as token events
            last_content = turn["answer"]
            yield {"type": "sources", "data": cited}
            yield {"type": "done"}
            return

        # ── plain template path (JSON protocol, non-streaming) ───────────────
        kwargs = dict(messages=messages, stream=False, temperature=0.3,
                      tools=TOOL_SCHEMAS, tool_choice="auto")
        try:
            response = llm.create_chat_completion(**kwargs)
        except Exception:
            # template/tool rendering failed — retry plain
            response = llm.create_chat_completion(
                messages=messages, stream=False, temperature=0.3,
            )
        choice = response["choices"][0]
        msg = choice["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        raw_content = content   # keep reasoning text even after a call is parsed out

        # surface the model's reasoning to the UI (gemma-4 thought channels)
        for tm in _THOUGHT_RE.finditer(raw_content):
            thought_text = re.sub(r"<\|channel>\w+\n?|\n?<channel\|>", "", tm.group(0)).strip()
            if thought_text:
                yield {"type": "thinking", "content": thought_text}

        # fallback: model wrote a tool call as JSON text instead of native call
        if not tool_calls:
            parsed = _parse_textual_tool_call(content)
            if parsed:
                name, args = parsed
                tool_calls = [{
                    "id": f"call_{iterations}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }]
                content = ""

        if tool_calls:
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield {"type": "tool_call", "tool": name, "args": args}
                fn = TOOLS.get(name)
                if fn is None:
                    result = f"Unknown tool: {name}. Available: {', '.join(TOOLS)}"
                else:
                    try:
                        result = fn(**args)
                    except TypeError as e:
                        result = f"Bad arguments for {name}: {e}"
                    except Exception as e:
                        result = f"Tool {name} failed: {e}"
                doc = args.get("name") if isinstance(args, dict) else None
                if name in _CITING_TOOLS and doc:
                    m = re.match(r"\[resolved '[^']+' -> '([^']+)'\]", result)
                    real = m.group(1) if m else doc
                    if real not in cited:
                        cited.append(real)
                if native_template:
                    # template understands tool turns natively (arguments MUST be a dict);
                    # preserve the model's reasoning across turns (template P4 — helps
                    # multi-step tool accuracy)
                    tmatch = _THOUGHT_RE.search(raw_content or "")
                    thought = re.sub(r"<\|channel>\w+\n?|\n?<channel\|>", "",
                                     tmatch.group(0)).strip() if tmatch else None
                    amsg = {"role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": tc.get("id", f"call_{iterations}"),
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            }]}
                    if thought:
                        amsg["reasoning"] = thought
                    messages.append(amsg)
                    messages.append({"role": "tool", "name": name,
                                     "tool_call_id": tc.get("id", f"call_{iterations}"),
                                     "content": result})
                else:
                    # template-agnostic feedback (plain templates reject the 'tool' role)
                    messages.append({"role": "assistant",
                                     "content": json.dumps({"tool": name, "args": args})})
                    messages.append({"role": "user",
                                     "content": f"TOOL RESULT ({name}):\n{result}"})
                yield {"type": "tool_result", "tool": name, "preview": result[:120]}
            iterations += 1
            continue

        # ── final answer ─────────────────────────────────────────────────────
        last_content = _strip_channels(content) if native_template else content
        yield from _stream_text(last_content)
        yield {"type": "sources", "data": cited}
        yield {"type": "done"}
        return

    # max iterations hit — force a final answer from what was gathered
    yield {"type": "token", "content": "I reached my search limit. Based on what I found: "}
    messages.append({"role": "user",
                     "content": "Stop using tools. Give your best final answer now from "
                                "what you have already read."})
    try:
        response = llm.create_chat_completion(messages=messages, stream=False,
                                              temperature=0.3)
        final = response["choices"][0]["message"].get("content") or last_content
    except Exception:
        final = last_content
    yield from _stream_text(final)
    yield {"type": "sources", "data": extract_cited_sources(messages)}
    yield {"type": "done"}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Summarize the knowledge base."
    print(f"Q: {q}\n" + "─" * 60)
    for ev in run_agent(q):
        if ev["type"] == "token":
            print(ev["content"], end="", flush=True)
        elif ev["type"] == "tool_call":
            print(f"\n→ {ev['tool']}  {ev['args']}")
        elif ev["type"] == "tool_result":
            print(f"  ✓ {ev['preview'][:80]!r}")
        elif ev["type"] == "sources":
            print(f"\n─ sources: {ev['data']}")

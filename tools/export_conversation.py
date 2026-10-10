"""Export a Claude Code session transcript (JSONL) to a readable Markdown log.

    python tools/export_conversation.py OUT.md transcript1.jsonl [transcript2.jsonl ...]

Keeps user messages (including mid-turn ones), assistant text, a one-line
summary per tool call and subagent reports; drops system context, injected
reminders and raw tool output.  Entries from several files are merged by
uuid and sorted by timestamp.
"""

import json
import re
import sys

REMINDER = re.compile(r"<system-reminder[^>]*>.*?</system-reminder[^>]*>", re.S)


def text_of(content):
    if isinstance(content, str):
        return [content]
    out = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return out


def tool_line(block):
    name = block.get("name", "tool")
    inp = block.get("input", {}) or {}
    what = (inp.get("description") or inp.get("summary") or inp.get("file_path")
            or inp.get("query") or inp.get("method") or inp.get("pattern") or "")
    if not what and "message" in inp:
        what = str(inp["message"])[:100]
    what = " ".join(str(what).split())[:160]
    return f"- `{name}`" + (f": {what}" if what else "")


def main(out, paths):
    entries = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid, ts, kind = o.get("uuid"), o.get("timestamp"), o.get("type")
            if not uid or not ts:
                continue
            if kind == "user":
                content = o.get("message", {}).get("content")
                if isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
                    continue
                raw = "\n".join(text_of(content))
                reports = re.findall(r"<agent-message[^>]*>(.*?)</agent-message>", raw, re.S)
                if reports:
                    body = "\n\n".join(r.strip() for r in reports)
                    entries[uid] = (ts, "agent", body)
                    continue
                if o.get("isMeta"):
                    continue
                txt = "\n".join(REMINDER.sub("", t).strip() for t in text_of(content)).strip()
                txt = re.sub(r"\[SYSTEM NOTIFICATION.*", "", txt, flags=re.S).strip()
                if not txt or txt.startswith("<task-notification"):
                    continue
                if txt.startswith("Another Claude session sent a message"):
                    body = re.sub(r"</?agent-message[^>]*>", "", txt)
                    body = re.sub(r"\nThat \"other Claude session\".*", "", body, flags=re.S)
                    entries[uid] = (ts, "agent", body.strip())
                else:
                    entries[uid] = (ts, "user", txt)
            elif kind == "attachment":
                a = o.get("attachment", {})
                if a.get("type") == "queued_command" and isinstance(a.get("prompt"), str):
                    p = a["prompt"].strip()
                    if p.startswith("<agent-message") or "Subagent hand-back" in p[:300]:
                        body = re.sub(r"</?agent-message[^>]*>", "", p)
                        entries[uid] = (ts, "agent", body.strip())
                    elif p.startswith("<task-notification") or p.startswith("<"):
                        continue
                    else:
                        entries[uid] = (ts, "user-midturn", p)
            elif kind == "assistant":
                content = o.get("message", {}).get("content", [])
                parts = []
                for b in content if isinstance(content, list) else []:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        parts.append(b["text"].strip())
                    elif b.get("type") == "tool_use":
                        parts.append(tool_line(b))
                if parts:
                    entries[uid] = (ts, "assistant", "\n".join(parts))
    rows = sorted(entries.values(), key=lambda e: e[0])
    label = {"user": "User", "user-midturn": "User (mid-turn)",
             "assistant": "Claude", "agent": "Subagent report"}
    with open(out, "w", encoding="utf-8") as f:
        f.write("# CB-ELE development session: conversation log\n\n")
        f.write("Exported from the Claude Code session transcript. System context, injected "
                "reminders and raw tool output are omitted; each tool call is summarised in "
                "one line.\n\n")
        prev = None
        for ts, who, txt in rows:
            if who == "assistant" and prev == "assistant":
                f.write(txt + "\n\n")
            else:
                f.write(f"---\n\n**{label[who]}** · {ts[:16].replace('T', ' ')} UTC\n\n{txt}\n\n")
            prev = who
    print(f"{len(rows)} entries -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

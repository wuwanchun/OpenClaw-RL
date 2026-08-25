"""Build structured sessions from WildClawBench run outputs.

Each session carries:
  session_id, task_id, score (official overall_score when present),
  trajectory (compact, lossy-clipped step trace),
  skills_referenced (skill names the agent actually read),
  has_tool_errors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CLIP = 400


def _clip(text: Any, limit: int = _CLIP) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."


def _coerce_content(content: Any) -> str:
    """content 可能是 str、parts 列表（[{type:text,text:...}]）或空。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("toolCall", "tool_call", "tool_use"):
                    continue  # 工具调用由 _render_message 单独渲染
                if "text" in part:
                    parts.append(str(part["text"]))
                else:
                    parts.append(json.dumps(part, ensure_ascii=False)[:200])
            else:
                parts.append(str(part))
        return " ".join(parts)
    return str(content or "")


def _unwrap_message(obj: dict) -> dict | None:
    """兼容两种 transcript schema：
    - openclaw 实际落盘: {"type": "message", "message": {"role", "content", ...}}
    - 扁平记录: {"role", "content", ...}
    """
    if obj.get("type") == "message" and isinstance(obj.get("message"), dict):
        return obj["message"]
    if "role" in obj:
        return obj
    return None


def _render_message(msg: dict) -> str:
    role = msg.get("role", "?")
    content = msg.get("content", "")
    text = _coerce_content(content)
    parts = [f"[{role}] {_clip(text)}"]
    # OpenAI 风格 tool_calls
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name", call.get("name", "tool"))
        args = _clip(fn.get("arguments", call.get("arguments", "")), 200)
        parts.append(f"  -> {name}({args})")
    # parts 列表里内嵌的工具调用（openclaw/pi 风格）
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("toolCall", "tool_call", "tool_use"):
                name = part.get("name", "tool")
                args = _clip(part.get("arguments", part.get("input", "")), 200)
                parts.append(f"  -> {name}({args})")
    if role in ("tool", "toolResult", "tool_result"):
        parts = [f"[tool_result] {_clip(text)}"]
    return "\n".join(parts)


def _detect_skills(text: str, known_skills: list[str]) -> set[str]:
    found = set()
    for name in known_skills:
        if name and name in text:
            found.add(name)
    # 只认 managed 路径的引用（.openclaw/skills/<name>/SKILL.md）。
    # 宽松正则会把 agent 在任务里自建的路径（如 /root/skills/...，
    # 06_task_10 这类注入测试任务）误判为"被引用的技能"，污染技能库。
    for match in re.findall(r"\.openclaw/skills/([A-Za-z0-9_.-]+)/SKILL\.md", text):
        found.add(match)
    return found


def build_session(run_dir: Path, known_skills: list[str]) -> dict[str, Any] | None:
    chat_path = run_dir / "chat.jsonl"
    if not chat_path.is_file():
        return None

    messages = []
    for line in chat_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict):
            unwrapped = _unwrap_message(msg)
            if unwrapped is not None:
                messages.append(unwrapped)

    trajectory = "\n".join(_render_message(m) for m in messages[:80])

    score = None
    score_path = run_dir / "score.json"
    if score_path.is_file():
        try:
            score = json.loads(score_path.read_text(encoding="utf-8")).get("overall_score")
        except json.JSONDecodeError:
            score = None

    task_id = run_dir.parent.name
    category = run_dir.parent.parent.name
    full_text = trajectory + "\n" + chat_path.read_text(encoding="utf-8", errors="replace")

    return {
        "session_id": run_dir.name,
        "task_id": f"{category}/{task_id}",
        "score": score,
        "num_turns": len(messages),
        "trajectory": trajectory,
        "skills_referenced": _detect_skills(full_text, known_skills),
        "has_tool_errors": "error" in trajectory.lower() or "✗" in trajectory,
        "run_dir": str(run_dir),
    }


def list_known_skills(skills_root: Path) -> list[str]:
    if not skills_root.is_dir():
        return []
    return sorted(p.name for p in skills_root.iterdir() if (p / "SKILL.md").is_file())


def build_sessions(raw_dir: str | Path, skills_root: str | Path) -> list[dict[str, Any]]:
    raw = Path(raw_dir)
    known = list_known_skills(Path(skills_root))
    sessions = []
    for chat_path in sorted(raw.rglob("chat.jsonl")):
        session = build_session(chat_path.parent, known)
        if session is not None:
            sessions.append(session)
    return sessions

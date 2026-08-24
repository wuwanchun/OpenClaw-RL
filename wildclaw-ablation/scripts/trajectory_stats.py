#!/usr/bin/env python3
"""统计各任务轨迹：轮数、工具调用、时长、分数、疑似循环。

用法:
  python3 trajectory_stats.py --results-dir <ABLATION_ROOT>/results [--variant collect_base]

输出按任务聚合的表格；--per-run 可打印每个 run 的明细。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# 复用进化器的 openclaw transcript 解析
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slime-coding-agent"))
from skill_evolve.sessions import _coerce_content, _unwrap_message  # noqa: E402


def _ts(entry: dict) -> datetime | None:
    raw = entry.get("timestamp")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def run_stats(run_dir: Path) -> dict | None:
    chat = run_dir / "chat.jsonl"
    if not chat.is_file():
        return None

    n_user = n_asst = n_tool_result = n_tool_calls = 0
    asst_chars = 0
    call_signatures: list[str] = []
    timestamps: list[datetime] = []

    for line in chat.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        ts = _ts(entry)
        if ts:
            timestamps.append(ts)
        msg = _unwrap_message(entry)
        if msg is None:
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            n_user += 1
        elif role == "assistant":
            n_asst += 1
            asst_chars += len(_coerce_content(content))
            for call in msg.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = fn.get("name", call.get("name", "tool"))
                args = fn.get("arguments", call.get("arguments", ""))
                n_tool_calls += 1
                call_signatures.append(f"{name}:{str(args)[:80]}")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("toolCall", "tool_call", "tool_use"):
                        n_tool_calls += 1
                        args = part.get("arguments", part.get("input", ""))
                        call_signatures.append(f"{part.get('name', 'tool')}:{str(args)[:80]}")
        elif role in ("tool", "toolResult", "tool_result"):
            n_tool_result += 1

    score = None
    score_path = run_dir / "score.json"
    if score_path.is_file():
        try:
            score = json.loads(score_path.read_text(encoding="utf-8")).get("overall_score")
        except json.JSONDecodeError:
            pass

    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None

    # 疑似循环：工具调用 >=10 且最频繁的调用签名占比 >50%（调用太少时不判定）
    loop_ratio = 0.0
    if len(call_signatures) >= 10:
        top = Counter(call_signatures).most_common(1)[0][1]
        ratio = top / len(call_signatures)
        loop_ratio = ratio if ratio > 0.5 else 0.0

    return {
        "run": run_dir.name,
        "turns": n_user + n_asst + n_tool_result,
        "assistant_msgs": n_asst,
        "tool_calls": n_tool_calls,
        "asst_chars": asst_chars,
        "duration_s": duration,
        "score": score,
        "loop_ratio": round(loop_ratio, 2),
    }


def fmt(v, digits=1):
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--variant", default=None, help="只看某个变体（默认全部）")
    ap.add_argument("--per-run", action="store_true", help="打印每个 run 明细")
    args = ap.parse_args()

    root = Path(args.results_dir)
    variants = [args.variant] if args.variant else [
        p.name for p in sorted(root.iterdir()) if p.is_dir() and (p / "raw").is_dir()
    ]

    for variant in variants:
        raw = root / variant / "raw"
        if not raw.is_dir():
            continue
        # task -> list[run_stats]
        per_task: dict[str, list[dict]] = {}
        for chat in raw.rglob("chat.jsonl"):
            run_dir = chat.parent
            st = run_stats(run_dir)
            if st is None:
                continue
            task = run_dir.parent.name  # .../<category>/<task>/<run>
            per_task.setdefault(task, []).append(st)

        print(f"\n===== variant: {variant} =====")
        header = f"{'task':52} {'runs':>4} {'score':>6} {'turns':>6} {'tools':>6} {'dur(s)':>7} {'loop':>5} {'zero%':>6}"
        print(header)
        print("-" * len(header))
        all_scores: list[float] = []
        for task in sorted(per_task):
            runs = per_task[task]
            scores = [r["score"] for r in runs if r["score"] is not None]
            all_scores.extend(scores)
            zero = sum(1 for s in scores if s == 0)
            mean = lambda key: statistics.mean(r[key] for r in runs if r[key] is not None) if any(r[key] is not None for r in runs) else None
            score_s = fmt(statistics.mean(scores), 3) if scores else "-"
            turns_s = fmt(mean("turns")) if mean("turns") else "-"
            tools_s = fmt(mean("tool_calls")) if mean("tool_calls") is not None else "-"
            dur_s = fmt(mean("duration_s"), 0) if mean("duration_s") else "-"
            loop_s = fmt(max(r["loop_ratio"] for r in runs), 2)
            zero_s = f"{zero}/{len(runs)}"
            print(f"{task[:52]:52} {len(runs):>4} {score_s:>6} {turns_s:>6} {tools_s:>6} {dur_s:>7} {loop_s:>5} {zero_s:>6}")

        if all_scores:
            print(f"\nvariant overall (macro over {len(all_scores)} runs): "
                  f"{statistics.mean(all_scores):.4f}")

        if args.per_run:
            for task in sorted(per_task):
                for r in per_task[task]:
                    print(f"  {task} | {r['run']} | score={r['score']} turns={r['turns']} "
                          f"tools={r['tool_calls']} dur={r['duration_s']} loop={r['loop_ratio']}")


if __name__ == "__main__":
    main()

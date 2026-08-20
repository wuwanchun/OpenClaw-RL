#!/usr/bin/env python3
"""Deterministically split WildClawBench text tasks into train / eval sets.

Writes configs/split.json:
  {"train": [task md paths...], "eval": [...]}

Split is stratified per category and stable for a given seed, so all variants
evaluate on the same held-out set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

# 联网依赖判定（2026-08-20 按 26 个真实任务校准）:
# 规则 A: ## Skills 段列出联网工具（agent-browser 等）—— 01_task_1/2/4/5/7/9
# 规则 B: Prompt/Expected Behavior 段显式要求联网 —— 01_task_3 ("use ... web
#         search")、06_task_1 ("search for ... download the PDF")
# 只扫 Prompt+Expected，不扫 Grading/Automated Checks：后者里的 URL 是数据
# （03 聊天记录、06 注入文本里的 reddit/scmp/localhost 链接都不算联网依赖）。
WEB_SKILLS = {"agent-browser", "browser-use", "web-browser", "web-search", "playwright"}
WEB_PROMPT_PATTERNS = [
    r"\bweb[ -]?search\b",
    r"\bsearch\s+(the\s+)?(web|internet|online)\b",
    r"\bsearch\s+for\b[\s\S]{0,80}?\bdownload\b",
    r"\bdownload\s+(the|this|a|that)\b[\s\S]{0,40}?\b(pdf|file|paper|dataset|repo)\b",
    r"\b(crawl|scrape|scraping)\b",
    r"\bvisit\b\s*https?://",
    r"请访问\s*https?://",
]


def web_dependency_reason(path: Path) -> str | None:
    """Return why the task needs internet access, or None if it is offline."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"## Skills\s*```(.*?)```", text, re.S)
    if m:
        used = {s.strip() for s in m.group(1).splitlines() if s.strip()}
        hit = sorted(used & WEB_SKILLS)
        if hit:
            return f"skills: {', '.join(hit)}"
    # 只看 Prompt + Expected Behavior（Grading/Checks 段里的 URL 是数据）
    scope = re.split(r"^## (?:Grading Criteria|Automated Checks|Workspace Path)", text, flags=re.M)[0]
    for pat in WEB_PROMPT_PATTERNS:
        hit = re.search(pat, scope, re.I)
        if hit:
            return f"prompt: /{pat}/ -> ...{hit.group(0)[:60]}..."
    return None


def split_tasks(tasks: list[Path], train_ratio: float, seed: str) -> tuple[list[str], list[str]]:
    ranked = sorted(
        tasks,
        key=lambda p: hashlib.sha256(f"{seed}:{p}".encode("utf-8")).hexdigest(),
    )
    n_train = max(1, round(len(ranked) * train_ratio)) if len(ranked) > 1 else len(ranked)
    train = [str(p) for p in ranked[:n_train]]
    eval_ = [str(p) for p in ranked[n_train:]]
    return train, eval_


def task_modality(path: Path) -> str:
    """Read the YAML frontmatter modality of a task markdown file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:3000]
    except OSError:
        return "unknown"
    m = re.search(r"^modality:\s*(\S+)", text, re.M)
    return m.group(1) if m else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wcb-root", required=True, help="WildClawBench clone root")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "01_Productivity_Flow",
            "03_Social_Interaction",
            "06_Safety_Alignment",
        ],
        help="默认排除 04_Search_Retrieval（需要 Brave API key）；要用就显式传入",
    )
    parser.add_argument(
        "--include-web",
        action="store_true",
        help="不排除联网依赖任务（默认排除：无搜索 key 时它们只能拿 0 分）",
    )
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--seed", default="wcb-ablation-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wcb_root = Path(args.wcb_root)
    train_all: list[str] = []
    eval_all: list[str] = []
    per_category = {}
    excluded_web: dict[str, str] = {}

    for cat in args.categories:
        task_dir = wcb_root / "tasks" / cat
        tasks = sorted(task_dir.glob("*.md")) if task_dir.is_dir() else []
        # exclude the annotated template
        tasks = [p for p in tasks if "template" not in p.name]
        # pure-text only: multimodal tasks need vision models
        tasks = [p for p in tasks if task_modality(p) == "pure-text"]
        # offline only: web-dependent tasks score 0 without search keys
        if not args.include_web:
            kept = []
            for p in tasks:
                reason = web_dependency_reason(p)
                if reason:
                    excluded_web[str(p)] = reason
                else:
                    kept.append(p)
            tasks = kept
        train, eval_ = split_tasks(tasks, args.train_ratio, args.seed + cat)
        per_category[cat] = {"train": len(train), "eval": len(eval_)}
        train_all.extend(train)
        eval_all.extend(eval_)

    payload = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "per_category": per_category,
        "excluded_web": excluded_web,
        "train": train_all,
        "eval": eval_all,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(per_category, ensure_ascii=False))
    for p, reason in excluded_web.items():
        print(f"excluded(web): {Path(p).name}  [{reason}]")
    print(f"train={len(train_all)} eval={len(eval_all)} -> {out}")


if __name__ == "__main__":
    main()

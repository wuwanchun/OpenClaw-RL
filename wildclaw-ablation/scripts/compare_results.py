#!/usr/bin/env python3
"""Aggregate WildClawBench ablation results across variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_scores(summary: dict) -> dict:
    """Normalize whatever summary_all.json exposes into {category: score, overall: score}."""
    out = {}
    if not summary:
        return out

    # tolerate a few known shapes
    if isinstance(summary.get("categories"), dict):
        for cat, payload in summary["categories"].items():
            if isinstance(payload, dict):
                score = payload.get("overall_score", payload.get("score"))
                if score is not None:
                    out[cat] = float(score)
    for key in ("overall_score", "overall", "total_score"):
        if key in summary:
            out["overall"] = float(summary[key])
            break
    if "overall" not in out and out:
        out["overall"] = round(sum(out.values()) / len(out), 4)
    return out


def aggregate_raw(variant_dir: Path) -> dict:
    """Fallback: aggregate per-run score.json files under raw/**/score.json.

    Layout: raw/openclaw/<category>/<task>/<run_id>/score.json.
    Per-task mean across rollouts first, then macro-average across tasks
    (so tasks with more rollouts don't dominate).
    """
    per_task: dict[tuple[str, str], list[float]] = {}
    for score_path in variant_dir.rglob("score.json"):
        try:
            data = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        score = data.get("overall_score")
        if score is None:
            continue
        parts = score_path.parts
        category = parts[-4] if len(parts) >= 4 else "unknown"
        task = parts[-3] if len(parts) >= 3 else score_path.parent.name
        per_task.setdefault((category, task), []).append(float(score))

    per_cat: dict[str, list[float]] = {}
    for (cat, _task), scores in per_task.items():
        per_cat.setdefault(cat, []).append(round(sum(scores) / len(scores), 4))

    out = {cat: round(sum(v) / len(v), 4) for cat, v in per_cat.items()}
    if out:
        out["overall"] = round(sum(out.values()) / len(out), 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.results_dir)
    table = {}
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    # 只有真正的评测变体进表；collect_* 是训练集采集（进化原料），logs 等是杂物
    VARIANT_PREFIXES = ("base", "skill_only", "rl_only", "rl_skill", "rl_opd")
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not variant_dir.name.startswith(VARIANT_PREFIXES):
            continue
        summary = load_summary(variant_dir / "summary_all.json")
        scores = extract_scores(summary)
        if not scores:
            # 单任务模式不产生 summary_all.json，从 raw/**/score.json 聚合
            scores = aggregate_raw(variant_dir)
        if not scores:
            continue  # 跳过 logs/ 等无分数目录
        table[variant_dir.name] = scores

    categories = sorted({cat for scores in table.values() for cat in scores if cat != "overall"})
    rows = []
    for variant, scores in table.items():
        row = {"variant": variant, "overall": scores.get("overall")}
        for cat in categories:
            row[cat] = scores.get(cat)
        rows.append(row)

    result = {"categories": categories, "rows": rows}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # also print a compact table
    header = ["variant", "overall", *categories]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in header))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate WildClawBench ablation results across variants.

默认输出: variant × (overall, per-category)。
带 --split configs/split.json 时额外输出 overall_train / overall_eval 两列,
把"train 集(参与过技能学习, 仅供参考)"和"eval 集(held-out, 可信口径)"分开展示。
"""

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


def aggregate_raw(variant_dir: Path) -> dict[str, list[float]]:
    """Aggregate per-run score.json files under raw/**/score.json.

    Layout: raw/openclaw/<category>/<task_dir>/<run_id>/score.json，
    其中 task_dir 已是全名（如 01_Productivity_Flow_task_6_calendar_scheduling），
    与 split.json 里任务 md 的文件名 stem 一致，直接作 key。
    以 "openclaw" 目录为锚点取层级，对前缀深度鲁棒。
    """
    per_task: dict[str, list[float]] = {}
    for score_path in variant_dir.rglob("score.json"):
        try:
            data = json.loads(score_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        score = data.get("overall_score")
        if score is None:
            continue
        parts = score_path.parts
        task_key = score_path.parent.parent.name  # <task_dir>
        if "openclaw" in parts:
            i = parts.index("openclaw")
            if i + 2 < len(parts):
                task_key = parts[i + 2]
        per_task.setdefault(task_key, []).append(float(score))
    return per_task


def macro_by_category(task_means: dict[str, float]) -> dict:
    """task key（含类别前缀的全名）-> {category: mean, overall: macro mean}。"""
    per_cat: dict[str, list[float]] = {}
    for stem, mean in task_means.items():
        cat = stem.split("_task_")[0]
        per_cat.setdefault(cat, []).append(mean)
    out = {cat: round(sum(v) / len(v), 4) for cat, v in per_cat.items()}
    if out:
        out["overall"] = round(sum(out.values()) / len(out), 4)
    return out


def load_split_tasks(split_path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    to_stems = lambda lst: {Path(p).stem for p in lst}
    return to_stems(data.get("train", [])), to_stems(data.get("eval", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=None,
                        help="split.json 路径；提供后按 train/eval 分开展示 overall")
    args = parser.parse_args()

    train_stems: set[str] = set()
    eval_stems: set[str] = set()
    if args.split:
        train_stems, eval_stems = load_split_tasks(Path(args.split))

    root = Path(args.results_dir)
    table: dict[str, dict] = {}
    if not root.is_dir():
        root.mkdir(parents=True, exist_ok=True)
    # 只有真正的评测变体进表；collect_* 是训练集采集（进化原料），logs 等是杂物
    VARIANT_PREFIXES = ("base", "skill_only", "rl_only", "rl_skill", "rl_opd")
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not variant_dir.name.startswith(VARIANT_PREFIXES):
            continue
        # raw 优先：summary_all.json 可能是陈旧批次留下的，raw/**/score.json 是实时真相
        per_task = aggregate_raw(variant_dir)
        scores: dict = {}
        row_extra: dict = {}
        if per_task:
            # per-task mean across rollouts first，再 macro over tasks
            task_means = {k: sum(v) / len(v) for k, v in per_task.items()}
            scores = macro_by_category(task_means)
            if args.split:
                for label, stems in (("train", train_stems), ("eval", eval_stems)):
                    subset = {k: v for k, v in task_means.items() if k in stems}
                    if subset:
                        row_extra[f"overall_{label}"] = round(
                            sum(subset.values()) / len(subset), 4)
        if not scores:
            scores = extract_scores(load_summary(variant_dir / "summary_all.json"))
        if not scores:
            continue  # 跳过 logs/ 等无分数目录
        scores.update(row_extra)
        table[variant_dir.name] = scores

    split_cols = ["overall_train", "overall_eval"] if args.split else []
    categories = sorted({cat for scores in table.values() for cat in scores
                         if cat not in ("overall", *split_cols)})
    rows = []
    for variant, scores in table.items():
        row = {"variant": variant, "overall": scores.get("overall")}
        for col in split_cols:
            row[col] = scores.get(col)
        for cat in categories:
            row[cat] = scores.get(cat)
        rows.append(row)

    result = {"categories": categories, "rows": rows}
    if args.split:
        result["split_note"] = (
            "overall_train = 参与过技能学习的任务（仅供参考）；"
            "overall_eval = held-out 任务（可信口径）"
        )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # also print a compact table
    header = ["variant", "overall", *split_cols, *categories]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in header))


if __name__ == "__main__":
    main()

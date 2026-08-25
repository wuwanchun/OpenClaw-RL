#!/usr/bin/env python3
"""Run one skill-evolution round over WCB train-split outputs.

Usage:
  python -m skill_evolve.run_round \
    --raw-dir wildclaw-ablation/results/collect/raw \
    --skills-dir wildclaw-ablation/skills \
    --report wildclaw-ablation/results/evolve_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolve import grouper, judge, llm, sessions as session_mod
from skill_evolve.evolver import LAST_LLM_STATUS, evolve_group, set_debug_dir
from skill_evolve.store import EvolvingSkillStore
from skill_evolve.verifier import verify


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:48] or "skill"


def _normalize_skill_md(skill_md: str, name: str) -> str:
    """强制 frontmatter 存在且 name 与存储目录名一致。

    grouper 按目录名归组、openclaw 按 frontmatter name 展示，两者不一致时
    用过该技能的轨迹归不到它名下，refine 回路断掉、每轮重复造重复技能。
    """
    text = skill_md.strip()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if m:
        fm = m.group(1)
        if re.search(r"^name:", fm, re.M):
            fm = re.sub(r"^name:.*$", f"name: {name}", fm, flags=re.M)
        else:
            fm = f"name: {name}\n{fm}"
        if not re.search(r"^description:", fm, re.M):
            fm = f"{fm}\ndescription: {name}"
        return f"---\n{fm}\n---\n{text[m.end():]}"
    return f"---\nname: {name}\ndescription: {name}\n---\n\n{text}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    skills_root = Path(args.skills_dir)
    store = EvolvingSkillStore(skills_root)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # LLM 原始输出落盘，便于诊断解析失败（必须在进化循环之前设置，否则永远不生效）
    set_debug_dir(str(report_path.parent / "evolve_debug"))

    sessions = session_mod.build_sessions(args.raw_dir, skills_root)
    judged = judge.backfill_scores(sessions)
    groups = grouper.group_sessions(sessions)

    report = {
        "sessions": len(sessions),
        "llm_judged": judged,
        "llm_configured": llm.is_configured(),
        "groups": {k: len(v) for k, v in groups.items()},
        "decisions": [],
    }

    for name, group in sorted(groups.items()):
        is_no_skill = name == grouper.NO_SKILL
        current = None if is_no_skill else store.current(name)
        history = [] if is_no_skill else store.history(name)

        candidate = evolve_group(name, group, current, history)
        if candidate is None:
            report["decisions"].append(
                {"group": name, "action": "skip", "llm": LAST_LLM_STATUS.get(name)}
            )
            continue

        gate = verify(candidate, group)
        if not gate["accepted"]:
            report["decisions"].append(
                {"group": name, "action": candidate["action"], "verifier": "reject",
                 "reason": gate["reason"]}
            )
            continue

        skill_name = name if not is_no_skill else _slug(f"wcb-recovery-{len(store.list_skills())+1}")
        skill_md = _normalize_skill_md(candidate["skill_md"], skill_name)
        version = store.publish(skill_name, skill_md, candidate["evidence"])
        report["decisions"].append(
            {"group": name, "action": candidate["action"], "verifier": "accept",
             "skill": skill_name, "version": version, "reason": gate["reason"]}
        )

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

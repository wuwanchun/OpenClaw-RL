#!/usr/bin/env python3
"""打印进化器/验证器实际收到的完整 prompt（不调 LLM），供人工审查。

用法:
  python3 show_evolve_prompt.py --raw-dir results/collect_e3/raw --skills-dir skills
  python3 show_evolve_prompt.py --raw-dir ... --skills-dir ... --group __no_skill__
  python3 show_evolve_prompt.py --raw-dir ... --skills-dir ... --verifier  # 看验证器 prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slime-coding-agent"))

from skill_evolve import grouper, sessions as session_mod
from skill_evolve.evolver import _SYSTEM as EVOLVER_SYSTEM, _format_evidence
from skill_evolve.store import EvolvingSkillStore
from skill_evolve.verifier import _SYSTEM as VERIFIER_SYSTEM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--skills-dir", required=True)
    ap.add_argument("--group", default=None, help="只看某组（默认全部）")
    ap.add_argument("--verifier", action="store_true", help="额外打印 verifier 的 system prompt 和证据结构")
    args = ap.parse_args()

    store = EvolvingSkillStore(Path(args.skills_dir))
    sessions = session_mod.build_sessions(args.raw_dir, args.skills_dir)
    groups = grouper.group_sessions(sessions)

    print(f"# sessions={len(sessions)} groups={ {k: len(v) for k, v in groups.items()} }")
    print()
    print("=" * 72)
    print("# EVOLVER SYSTEM PROMPT（所有组共用）")
    print("=" * 72)
    print(EVOLVER_SYSTEM)

    for name, group in sorted(groups.items()):
        if args.group and name != args.group:
            continue
        is_no_skill = name == grouper.NO_SKILL
        current = None if is_no_skill else store.current(name)
        history = [] if is_no_skill else store.history(name)
        history_text = "\n\n".join(
            f"### {h['version']}\n{h['content'][:1500]}\nEvidence: {h['evidence'][:500]}"
            for h in history[-4:]
        ) or "(no history)"
        current_text = current[:4000] if current else "(no current skill - decide create vs skip)"
        user = (
            f"Skill under review: {name}\n\n"
            f"## Current skill\n{current_text}\n\n"
            f"## History\n{history_text}\n\n"
            f"## Sessions\n{_format_evidence(group)}"
        )
        print()
        print("=" * 72)
        print(f"# GROUP: {name}  ({len(group)} sessions)")
        print("=" * 72)
        print(user)

    if args.verifier:
        print()
        print("=" * 72)
        print("# VERIFIER SYSTEM PROMPT（发布门禁，候选技能产生后才调用）")
        print("=" * 72)
        print(VERIFIER_SYSTEM)
        print()
        print("# verifier 的 user 消息结构：")
        print("## Candidate skill\\n<skill_md 前6000字符>\\n\\n"
              "## Motivating evidence\\n<evolver 给的 evidence 前1000字符>\\n\\n"
              "## Session excerpts\\n<该组前6条轨迹各1500字符>")


if __name__ == "__main__":
    main()

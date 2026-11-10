#!/usr/bin/env python3
"""Rebuild grounded supervision after an independent generated-view review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task_evidence_nvs.annotation import rebuild_grounded_text, review_generated_view
from task_evidence_nvs.task import parse_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task description or a JSON task file")
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--confirmations", type=Path, required=True)
    parser.add_argument("--view-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_path = Path(args.task)
    if task_path.exists():
        task_data = json.loads(task_path.read_text(encoding="utf-8"))
        task = parse_task(str(task_data["raw_text"]))
    else:
        task = parse_task(args.task)
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    confirmations = json.loads(args.confirmations.read_text(encoding="utf-8"))
    if isinstance(proposals, dict):
        proposals = proposals.get("objects", proposals.get("proposals", []))
    if isinstance(confirmations, dict):
        confirmations = confirmations.get("objects", confirmations.get("confirmations", []))
    annotation = review_generated_view(args.view_id, proposals, confirmations)
    result = {
        "annotation": annotation.to_dict(),
        "grounded_text": rebuild_grounded_text(annotation, task),
        "review_policy": {
            "source_boxes_are_not_inherited": True,
            "identity_unknown_is_not_a_positive_label": True,
            "generated_new_is_distinct_from_preserved": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"view_id": args.view_id, "usable_objects": len(annotation.usable_objects), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

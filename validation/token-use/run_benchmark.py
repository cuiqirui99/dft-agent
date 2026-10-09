"""Check the paired fixtures offline, or explicitly run six model requests."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from vasp_slurm_agent import agent
from vasp_slurm_agent.model_usage import response_usage
from vasp_slurm_agent.prompt_context import compact_history, compact_scientific_context, compact_structure


CASES = ("long-revision", "large-geometry", "retrieved-advisory")


def read(case, filename):
    return json.loads((ROOT / case / filename).read_text())


def geometry(value):
    result = deepcopy(value)
    columns = result.pop("site_columns", None)
    if columns is not None:
        result["sites"] = [dict(zip(columns, row)) for row in result["sites"]]
    return result


def check_fixture(case):
    baseline, compact = read(case, "baseline.payload.json"), read(case, "optimized.payload.json")
    spec = read(case, "pair-spec.json")
    assert baseline["goal"] == compact["goal"]
    assert geometry(baseline["structure"]) == geometry(compact["structure"])
    assert baseline["numeric_defaults"] == compact["numeric_defaults"]
    assert compact_history(baseline["history"], goal=baseline["goal"]) == compact["history"]
    assert compact_structure(baseline["structure"]) == compact["structure"]
    assert compact_scientific_context(baseline["scientific_context"]) == compact["scientific_context"]
    assert hashlib.sha256((ROOT / case / "POSCAR").read_bytes()).hexdigest() == spec["source_sha256"]
    sizes = {variant: (ROOT / case / (variant + ".payload.json")).stat().st_size
             for variant in ("baseline", "optimized")}
    assert all(size <= 96000 for size in sizes.values())
    return {"case": case, "fixture_checks": "passed", "payload_bytes": sizes}


def check_response(raw, case, settings):
    baseline = read(case, "baseline.payload.json")
    species = [site["element"] for site in geometry(baseline["structure"])["sites"]]
    plan = agent._validate_plan(json.loads(raw), species, settings)
    expected = read(case, "expected.json")
    mismatches = []
    for key in ("status", "tasks", "magnetic_states"):
        if plan[key] != expected[key]:
            mismatches.append(key)
    for key, value in expected["parameters"].items():
        if plan["parameters"].get(key) != value:
            mismatches.append("parameters." + key)
    for key, value in expected["intent"].items():
        if key == "forbidden_tasks_contains":
            matches = set(value) <= set(plan["intent"]["forbidden_tasks"])
        else:
            matches = plan["intent"].get(key) == value
        if not matches:
            mismatches.append("intent." + key)
    return {"semantic_checks": "passed" if not mismatches else "failed", "mismatches": mismatches,
            "plan": {key: plan[key] for key in ("status", "tasks", "parameters", "magnetic_states", "intent")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Call the selected provider; consumes model usage.")
    parser.add_argument("--provider", choices=("codex", "responses", "chat_completions"))
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", help="Optional API endpoint; authentication uses the usual environment.")
    parser.add_argument("--case", choices=CASES, action="append", help="Run selected cases; default: all three.")
    parser.add_argument("--output", type=Path, help="Save a JSON receipt.")
    args = parser.parse_args()
    selected = args.case or list(CASES)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": "live" if args.run else "offline",
              "limits": ["One paired call per context does not establish average savings or latency.",
                         "Cached input and reasoning tokens are subsets, not extra tokens.",
                         "Offline payload bytes are not token counts."],
              "fixtures": [check_fixture(case) for case in selected], "calls": []}
    if args.run:
        if not args.provider:
            parser.error("--run requires --provider")
        settings = agent._settings(agent.ModelSettings(args.provider, args.model, args.base_url))
        for case in selected:
            instructions = (ROOT / case / "instructions.txt").read_text()
            schema = read(case, "schema.json")
            for variant in ("baseline", "optimized"):
                payload = (ROOT / case / (variant + ".payload.json")).read_text()
                row = {"case": case, "variant": variant,
                       "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                       "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
                       "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()}
                raw = None
                try:
                    raw = agent._request_structured(payload, schema, settings,
                                                    instructions=instructions, name="dft_plan")
                    row["model_usage"] = response_usage(raw, settings)
                    row.update(check_response(raw, case, settings))
                except agent.AgentError as exc:
                    row.update(semantic_checks="failed", error=str(exc), model_usage=exc.model_usage)
                except (ValueError, KeyError, TypeError):
                    row.update(semantic_checks="failed", error="Invalid model response.",
                               model_usage=response_usage(raw, settings) if raw is not None else None)
                report["calls"].append(row)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({key: value for key, value in row.items() if key != "plan"}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not args.run:
        print(json.dumps(report, indent=2))
    return int(any(row["semantic_checks"] != "passed" for row in report["calls"]))


if __name__ == "__main__":
    raise SystemExit(main())

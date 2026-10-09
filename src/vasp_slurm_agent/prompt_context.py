"""Compact model inputs without shortening user instructions or current geometry."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json


_PLAN_METADATA = {"scientific_report", "preview", "provenance", "model_usage", "usage",
                  "source_sha256", "output_sha256", "context_sha256", "schema_version"}


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def compact_history(history: list[dict], *, goal: str | None = None) -> list[dict]:
    """Keep every turn and all plan choices; omit only derived plan metadata."""
    result = deepcopy(history)
    for item in result:
        if item.get("role") != "assistant" or not isinstance(item.get("content"), str):
            continue
        try:
            plan = json.loads(item["content"])
        except (TypeError, ValueError):
            continue
        if not isinstance(plan, dict) or plan.get("status") not in {"ready", "needs_input", "unsupported"}:
            continue
        if not (("tasks" in plan and "parameters" in plan) or "operations" in plan):
            continue
        plan = {key: value for key, value in plan.items() if key not in _PLAN_METADATA}
        if goal is not None and plan.get("goal") == goal:
            del plan["goal"]
        item["content"] = _json(plan)
    return result


def compact_structure(structure: dict | None) -> dict | None:
    """Use a complete site table for larger cells, with no rounding or sorting."""
    if structure is None:
        return None
    result = deepcopy(structure)
    sites = result.get("sites")
    if not isinstance(sites, list) or len(sites) <= 8 or not all(isinstance(site, dict) for site in sites):
        return result
    columns = list(sites[0])
    if not columns or any(set(site) != set(columns) for site in sites):
        return result
    rows = [[site[key] for key in columns] for site in sites]
    if len(_json({"site_columns": columns, "sites": rows})) < len(_json({"sites": sites})):
        result["site_columns"] = columns
        result["sites"] = rows
    return result


def _vector_reference(value: list) -> dict:
    return {"items": len(value), "sha256": hashlib.sha256(_json(value).encode()).hexdigest(),
            "location": "scientific_report.experience", "available_to_model": False}


def compact_scientific_context(science: dict) -> dict:
    """Keep guidance and evidence IDs; bulky advisory vectors stay in the report."""
    result = deepcopy(science)
    for evidence in result.get("evidence", []):
        evidence.pop("title", None)
        for source in evidence.get("sources", []):
            if isinstance(source, dict):
                source.pop("checked", None)
    omitted_vectors = False
    for case in result.get("experience", []):
        case.pop("context_sha256", None)
        for stage in case.get("stages", []):
            for field in ("method", "settings"):
                parameters = stage.get(field, {})
                moments = parameters.get("magmom")
                if isinstance(moments, list) and len(moments) > 8:
                    parameters["magmom_ref"] = _vector_reference(parameters.pop("magmom"))
                    omitted_vectors = True
            for evidence in stage.get("evidence", []):
                value = evidence.get("value")
                if isinstance(value, list) and len(value) > 8:
                    evidence["value_ref"] = _vector_reference(evidence.pop("value"))
                    omitted_vectors = True
    if omitted_vectors:
        result["experience_note"] = (
            "Large site vectors are retained in the saved report and referenced by hash. "
            "Their values are unavailable here. Do not infer or copy them into this plan."
        )
    return result


def compact_explanation_context(context: dict) -> dict:
    """Retain all verified fact values and IDs; source hashes remain checked locally."""
    result = deepcopy(context)
    result.pop("context_sha256", None)
    result["dialogue"] = compact_history(result.get("dialogue", []), goal=result.get("goal"))
    for fact in result.get("facts", {}).values():
        fact.pop("source", None)
    for artifact in result.get("artifacts", []):
        artifact.pop("path", None)
        artifact.pop("sha256", None)
    return result

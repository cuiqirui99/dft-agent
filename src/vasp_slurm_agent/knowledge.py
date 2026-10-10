"""Bundled lessons and selected local records. Retrieval never changes inputs."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
import tempfile
from urllib.parse import urlsplit

from pymatgen.core import Composition, Element

from .guidance import _signals


STATUSES = {"verified", "candidate", "shadow_verified", "rejected", "deprecated"}
KINDS = {"guidance", "repair", "case"}
TASKS = {"relax", "scf", "bands", "dos"}
MAX_RECORDS = 2000
MAX_FILE_BYTES = 8_000_000
MAX_SOURCE_BYTES = 32_000_000
MAX_RESULTS = 8
_LOCAL = ".dft-agent-memory"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE = re.compile(
    r"(?i)\b(?:password|passwd|token|api[_ -]?key|secret|authorization)\b\s*[:=]\s*[^\s,;]+"
    r"|\b(?:sk|ghp|gho|github_pat)-?[A-Za-z0-9_]{12,}\b"
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    r"|\b(?:ssh|https?)://[^\s]+"
    r"|(?<!\w)(?:~?/|[A-Za-z]:\\)[^\s,;]+"
)
_METHODS = {
    "pbe": "PBE", "hse": "HSE06", "hse06": "HSE06", "pbe0": "PBE0",
    "soc": "SOC", "spin-orbit": "SOC", "dft+u": "DFT+U", "hubbard_u": "DFT+U",
    "collinear": "collinear", "noncollinear": "noncollinear",
    "nonmagnetic": "nonmagnetic", "none": "nonmagnetic",
}
_TASK_ALIASES = {"relaxation": "relax", "static": "scf", "band": "bands", "band_structure": "bands"}


def _text(value, limit=600):
    if not isinstance(value, str):
        return ""
    value = _PRIVATE.sub("[private]", value)
    return " ".join(value.split())[:limit]


def _strings(value, *, limit=12, length=200):
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(text for item in value[:limit] if (text := _text(item, length))))


def _formula(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9(). ]{1,100}", value):
        return ""
    try:
        composition = Composition(value)
        if not composition or any(not isinstance(element, Element) for element in composition.elements):
            return ""
        return composition.reduced_formula
    except (ValueError, TypeError, AttributeError):
        return ""


def _query_formula(structure):
    if not isinstance(structure, dict):
        return ""
    sites = structure.get("sites")
    if isinstance(sites, list) and sites:
        elements = [site.get("element") for site in sites if isinstance(site, dict)]
        if len(elements) == len(sites) and all(isinstance(el, str) and re.fullmatch(r"[A-Z][a-z]?", el) for el in elements):
            try:
                return Composition(Counter(elements)).reduced_formula
            except ValueError:
                return ""
    return _formula(structure.get("formula"))


def _url(value):
    if not isinstance(value, str) or len(value) > 1000:
        return ""
    try:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query:
            return ""
        return value
    except ValueError:
        return ""


def _relative(value):
    if not isinstance(value, str) or "\\" in value or len(value) > 250:
        return False
    path = PurePosixPath(value)
    return bool(value and not path.is_absolute() and not PureWindowsPath(value).drive
                and all(part not in {".", ".."} for part in value.split("/")))


def validate_record(value):
    """Return a bounded record or raise ValueError for an invalid schema."""
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not _ID.fullmatch(value["id"]):
        raise ValueError("Invalid knowledge record ID.")
    if value.get("kind") not in KINDS or value.get("status") not in STATUSES:
        raise ValueError("Invalid knowledge kind or status.")
    if value.get("outcome") not in {"guidance", "succeeded", "failed", "not_evaluated"}:
        raise ValueError("Invalid knowledge outcome.")
    record = {key: value[key] for key in ("id", "kind", "status", "outcome")}
    for key, length in (("title", 140), ("summary", 1200)):
        record[key] = _text(value.get(key), length)
        if not record[key]:
            raise ValueError("Knowledge records need a title and summary.")
    for key in ("conditions", "limitations"):
        record[key] = _strings(value.get(key), length=500)
    for key in ("topics", "tasks", "formulas", "methods", "sources", "evidence"):
        if not isinstance(value.get(key, []), list):
            raise ValueError("Knowledge fields must be lists.")
    record["topics"] = [item for item in _strings(value.get("topics"), length=60)
                        if re.fullmatch(r"[a-z][a-z0-9_+-]*", item)]
    record["tasks"] = [item for item in _strings(value.get("tasks")) if item in TASKS]
    if value.get("tasks") and not record["tasks"]:
        raise ValueError("Invalid knowledge tasks.")
    record["formulas"] = list(dict.fromkeys(formula for item in _strings(value.get("formulas")) if (formula := _formula(item))))
    if value.get("formulas") and not record["formulas"]:
        raise ValueError("Invalid knowledge formulas.")
    record["methods"] = list(dict.fromkeys(_METHODS[item.lower()] for item in _strings(value.get("methods")) if item.lower() in _METHODS))
    if value.get("methods") and len(record["methods"]) != len(set(value["methods"])):
        raise ValueError("Invalid knowledge methods.")
    record["sources"] = [{"title": _text(item.get("title"), 160), "url": url}
                         for item in (value.get("sources") or [])[:12]
                         if isinstance(item, dict) and (url := _url(item.get("url")))]
    record["evidence"] = []
    for item in (value.get("evidence") or [])[:24]:
        if not isinstance(item, dict) or not _relative(item.get("path")) or not isinstance(item.get("sha256"), str) or not _SHA.fullmatch(item["sha256"]):
            raise ValueError("Invalid knowledge evidence reference.")
        record["evidence"].append({"path": item["path"], "sha256": item["sha256"],
                                   "description": _text(item.get("description"), 250)})
    if isinstance(value.get("legacy_id"), str):
        legacy_id = value["legacy_id"]
        record["legacy_id"] = legacy_id if _ID.fullmatch(legacy_id) else hashlib.sha256(legacy_id.encode()).hexdigest()[:16]
    if value.get("import_type") in {"experience", "lesson", "run"}:
        record["import_type"] = value["import_type"]
    if value.get("legacy_status") in STATUSES | {"production"}:
        record["legacy_status"] = value["legacy_status"]
    return record


def _read_json(path, maximum=MAX_FILE_BYTES):
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("The record is missing or too large.")
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog(path):
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("records"), list) or len(value["records"]) > MAX_RECORDS:
        raise ValueError("Unsupported knowledge catalogue.")
    return value["records"]


def _evidence_valid(record, root):
    try:
        for item in record["evidence"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                return False
        return bool(record["evidence"] or (record["kind"] == "guidance" and record["sources"]))
    except OSError:
        return False


def load_catalog(runs_root=None):
    """Load public and selected local records, including inactive lessons."""
    roots = [(Path(str(files("vasp_slurm_agent").joinpath("knowledge"))), "builtin")]
    if runs_root is not None:
        roots.append((Path(runs_root).expanduser().resolve() / _LOCAL, "local"))
    records = []
    for root, origin in roots:
        try:
            raw = _catalog(root / "catalog.json")
        except (OSError, ValueError, TypeError):
            continue
        seen = set()
        for value in raw:
            try:
                record = validate_record(value)
                if record["id"] in seen:
                    continue
                seen.add(record["id"])
                if origin == "local" and record["status"] == "verified":
                    record["status"] = "candidate"
                record.update(origin=origin, builtin=origin == "builtin",
                              evidence_valid=_evidence_valid(record, root))
                records.append(record)
            except (ValueError, TypeError, KeyError):
                continue
    return records


def _query(goal, parameters, tasks, history):
    signals = _signals(goal, history)
    requested = set(task for task in (tasks or []) if task in TASKS)
    if not requested:
        requested.update(task for task in TASKS if signals.get(task))
        if re.search(r"\bscf\b|self.consistent|自洽", str(goal), re.I):
            requested.add("scf")
    methods = {"PBE"}
    if parameters is not None:
        functional = parameters.get("functional", "PBE")
        methods = {functional} if functional in {"PBE", "HSE06", "PBE0"} else set()
        if parameters.get("soc"):
            methods.add("SOC")
        if parameters.get("hubbard_u"):
            methods.add("DFT+U")
        spin = parameters.get("spin", "none")
        methods.add("nonmagnetic" if spin == "none" else spin)
    else:
        if signals.get("hybrid"):
            text = " ".join([str(goal), *[str(item.get("content", "")) for item in (history or []) if isinstance(item, dict) and item.get("role") == "user"]])
            matches = list(re.finditer(r"\b(hse(?:06)?|pbe0)\b", text, re.I))
            methods = {_METHODS[matches[-1].group().lower()]} if matches else set()
        if signals.get("soc"):
            methods.add("SOC")
        if signals.get("hubbard_u"):
            methods.add("DFT+U")
        if signals.get("magnetism") is False:
            methods.add("nonmagnetic")
    topics = {"convergence"} | {key for key, enabled in signals.items() if enabled}
    if requested & {"bands", "dos"}:
        topics.add("smearing")
        if "PBE" in methods:
            topics.add("pbe_spectra")
    if "SOC" in methods:
        topics.add("soc")
    if "DFT+U" in methods:
        topics.add("hubbard_u")
    if methods & {"HSE06", "PBE0"}:
        topics.add("hybrid")
    if methods & {"collinear", "noncollinear"}:
        topics.add("magnetism")
    for topic, pattern in {
        "recovery": r"fail|error|repair|recover|not converg|unconverg|timeout|wall.?time|失败|报错|修复|不收敛|超时",
        "electronic": r"electronic|nelm|scf.{0,20}converg|电子.*收敛|自洽.*收敛",
        "ionic": r"ionic|nsw|relax.{0,20}converg|离子|弛豫.*收敛",
        "walltime": r"timeout|time.?limit|wall.?time|超时|时限",
        "missing_output": r"(?:missing|incomplete|unavailable).{0,25}output|output.{0,25}(?:missing|incomplete|unavailable)|missing_output|输出.*(?:缺失|不完整)|缺.*输出",
    }.items():
        if re.search(pattern, str(goal), re.I):
            topics.add(topic)
    excluded = {key for key, enabled in signals.items() if not enabled}
    if "soc" in excluded:
        methods.discard("SOC")
    if "hubbard_u" in excluded:
        methods.discard("DFT+U")
    return requested, methods, topics - excluded, excluded


def retrieve_knowledge(goal, structure, *, parameters=None, tasks=None, runs_root=None, history=None, limit=4):
    """Return relevant verified references, never executable parameter advice."""
    if type(limit) is not int or limit <= 0:
        return []
    formula = _query_formula(structure)
    requested, methods, topics, excluded = _query(goal, parameters, tasks, history)
    if tasks and not requested:
        return []
    matches = []
    for record in load_catalog(runs_root):
        if record["status"] != "verified" or not record["evidence_valid"]:
            continue
        record_topics = set(record["topics"])
        if record_topics & excluded or (record["formulas"] and formula not in record["formulas"]):
            continue
        if record["tasks"] and not requested.intersection(record["tasks"]):
            continue
        recorded_methods = set(record["methods"])
        functionals = recorded_methods & {"PBE", "HSE06", "PBE0"}
        if functionals and not functionals.intersection(methods):
            continue
        features = recorded_methods - functionals
        if not features.issubset(methods):
            continue
        if record["kind"] == "case" and recorded_methods:
            if any((feature in methods) != (feature in recorded_methods) for feature in ("SOC", "DFT+U")):
                continue
            if methods & {"collinear", "noncollinear"} and not methods.intersection(features & {"collinear", "noncollinear"}):
                continue
        if record["kind"] == "repair" and "recovery" not in topics:
            continue
        causes = {"electronic", "ionic", "walltime", "missing_output"}
        if record["kind"] == "repair" and topics & causes and record_topics & causes and not topics.intersection(record_topics & causes):
            continue
        if "missing_output" in record_topics and "missing_output" not in topics:
            continue
        overlap = record_topics & topics
        if not overlap and not record["formulas"]:
            continue
        score = len(overlap) + (3 if record["formulas"] else 0) + len(requested.intersection(record["tasks"]))
        reference = {key: record[key] for key in (
            "id", "kind", "title", "status", "summary", "conditions", "limitations", "outcome", "sources",
            "topics", "tasks", "formulas", "methods", "origin")}
        reference["evidence"] = [{"sha256": item["sha256"], "description": item["description"]} for item in record["evidence"]]
        reference["use"] = "Reference only. Recheck applicability; do not copy settings or infer a ground state."
        reference["numerical_convergence_verified"] = False
        matches.append((score, record["id"], reference))
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in matches[:min(limit, MAX_RESULTS)]]


def _status(value):
    return value if isinstance(value, str) and value in {"candidate", "shadow_verified", "rejected", "deprecated"} else "candidate"


def _json(value, fallback):
    if not isinstance(value, str):
        return value if isinstance(value, type(fallback)) else fallback
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, type(fallback)) else fallback
    except ValueError:
        return fallback


def _normal_import(raw, source_type):
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    identifier = "local-" + digest[:24]
    legacy_id = str(raw.get("record_id") or raw.get("id") or digest[:16])
    status = raw.get("current_status", raw.get("status", "candidate"))
    scope = _json(raw.get("scope"), {}) if source_type == "lesson" else raw.get("material_context", {})
    scope = scope if isinstance(scope, dict) else {}
    formula = _formula(scope.get("reduced_formula", scope.get("formula")))
    task = scope.get("task", raw.get("step_kind", raw.get("target_task", "")))
    task = _TASK_ALIASES.get(task, task) if isinstance(task, str) else ""
    tasks = [task] if task in TASKS else []
    kind = "repair" if raw.get("lesson_type") == "failure_repair" else "guidance" if source_type == "lesson" else "case"
    if source_type == "lesson":
        title = _text(raw.get("lesson_key"), 140) or "Imported lesson"
        summary = "Selected legacy lesson. Its evidence has not been checked by this version."
    else:
        title = " ".join([formula or "Material", task.upper() if tasks else "calculation", "record"])
        summary = "Selected legacy calculation. Its reported outcome has not been checked by this version."
    quality = raw.get("result_quality") if isinstance(raw.get("result_quality"), dict) else {}
    outcome = "failed" if "failure_signature" in raw or quality.get("converged") is False else "not_evaluated"
    record = {"id": identifier, "kind": kind, "title": title, "status": _status(status),
              "topics": ["recovery"] if kind == "repair" else [], "tasks": tasks,
              "formulas": [formula] if formula else [], "methods": [], "summary": summary,
              "conditions": ["Review the selected source record and its original evidence locally."],
              "limitations": ["Not used for automatic suggestions. Legacy status does not establish current validity."],
              "outcome": outcome, "sources": [], "evidence": [], "legacy_id": legacy_id,
              "import_type": source_type}
    if isinstance(status, str) and status in STATUSES | {"production"}:
        record["legacy_status"] = status
    method = raw.get("method_context")
    if isinstance(method, dict):
        functional = str(method.get("functional", "")).lower()
        if functional in _METHODS:
            record["methods"].append(_METHODS[functional])
        if method.get("has_soc") is True:
            record["methods"].append("SOC")
        if method.get("has_u") is True:
            record["methods"].append("DFT+U")
    return validate_record(record)


def _read_import(source):
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        if (path / "run.json").is_file() and (path / "plan.json").is_file():
            return _run_import(path)
        candidates = [path / "records.jsonl", path / "failures.jsonl"]
        if not any(item.is_file() for item in candidates):
            raise ValueError("Choose a JSONL file, a lessons database, or one saved run folder.")
        result = []
        for candidate in candidates:
            if candidate.is_file():
                result.extend(_read_import(candidate))
        if len(result) > MAX_RECORDS:
            raise ValueError("The import source has too many records.")
        return result
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("The import source is missing or too large.")
    result = []
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= MAX_RECORDS or len(line.encode()) > MAX_FILE_BYTES:
                    raise ValueError("The import source contains too many or oversized records.")
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict) or not raw.get("record_id"):
                    raise ValueError("Expected legacy experience records with record_id.")
                result.append((_normal_import(raw, "experience"), raw))
    elif path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only = ON")
                query = """SELECT lv.*, l.lesson_key, l.current_status FROM lessons AS l
                           JOIN lesson_versions AS lv ON l.current_version_id = lv.id
                           ORDER BY l.id LIMIT ?"""
                rows = connection.execute(query, (MAX_RECORDS + 1,)).fetchall()
                if len(rows) > MAX_RECORDS:
                    raise ValueError("The lessons database has too many records.")
                for row in rows:
                    raw = dict(row)
                    result.append((_normal_import(raw, "lesson"), raw))
        except sqlite3.Error as error:
            raise ValueError("Could not read the legacy lessons tables.") from error
    else:
        raise ValueError("Choose a JSONL file, a lessons database, or one saved run folder.")
    unique = {record["id"]: (record, raw) for record, raw in result}
    return list(unique.values())


def preview_import(source):
    """Read a source without writing. Returned summaries contain no raw payloads."""
    return [record for record, _ in _read_import(source)]


def _write_json(path, value):
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            os.chmod(temporary, 0o600)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def import_selected(source, ids, runs_root):
    """Save only explicitly selected records locally; source files stay untouched."""
    if not isinstance(ids, (list, tuple, set)) or any(not isinstance(item, str) for item in ids):
        raise ValueError("Choose record IDs from the import preview.")
    requested = set(ids)
    available = {record["id"]: (record, raw) for record, raw in _read_import(source)}
    if not requested.issubset(available):
        raise ValueError("The selection changed. Preview the source again.")
    root = Path(runs_root).expanduser().resolve() / _LOCAL
    catalog_path = root / "catalog.json"
    if not requested:
        return {"imported": 0, "skipped": 0, "ids": [], "catalog_path": str(catalog_path)}
    evidence_dir = root / "evidence"
    if root.is_symlink() or evidence_dir.is_symlink() or catalog_path.is_symlink():
        raise ValueError("The memory folder must not use symbolic links.")
    existing = _catalog(catalog_path) if catalog_path.exists() else []
    records = [validate_record(item) for item in existing]
    seen = {item["id"] for item in records}
    selected = sorted(requested - seen)
    if len(records) + len(selected) > MAX_RECORDS:
        raise ValueError("The local catalogue is full.")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    evidence_dir.mkdir(exist_ok=True, mode=0o700)
    for identifier in selected:
        record, raw = available[identifier]
        relative = f"evidence/{identifier}.json"
        digest = _write_json(root / relative, raw)
        record["evidence"] = [{"path": relative, "sha256": digest,
                               "description": "Selected local source record; not public or sent to a model."}]
        records.append(record)
    _write_json(catalog_path, {"schema_version": 1, "records": records})
    return {"imported": len(selected), "skipped": len(requested & seen), "ids": selected,
            "catalog_path": str(catalog_path)}


def _run_import(path):
    from . import experience
    plan, _ = experience._read(path / "plan.json")
    state, _ = experience._read(path / "run.json")
    stages = plan.get("stages") or []
    if not stages:
        raise ValueError("The saved run has no calculation stages.")
    structure = experience._source(path, stages[0]["source_path"], plan.get("sources") or {})
    query = {"formula": structure.composition.reduced_formula,
             "lattice_angstrom": structure.lattice.matrix.tolist(),
             "sites": [{"element": site.specie.symbol, "fractional_coordinates": site.frac_coords.tolist()} for site in structure]}
    checked = experience.retrieve_experience(path.parent, query, run_paths=[path], limit=1)
    if not checked:
        raise ValueError("This run cannot be checked. Keep its original inputs, plan and outputs together.")
    case = checked[0]
    raw = {"source_run": str(path), "context_sha256": case["context_sha256"], "structure": query}
    identifier = "local-" + hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:24]
    outcome = "succeeded" if all(stage["accepted"] for stage in case["stages"]) else "failed"
    record = {"id": identifier, "kind": "case", "title": f"{query['formula']} saved run", "status": "candidate",
              "topics": [], "tasks": list(dict.fromkeys(stage["task"] for stage in case["stages"])),
              "formulas": [query["formula"]], "methods": [],
              "summary": "Selected saved run. Matching inputs and results are checked again before use.",
              "conditions": ["Keep the original run directory available."],
              "limitations": ["Result checks do not establish property convergence or a ground state."],
              "outcome": outcome, "sources": [], "evidence": [], "import_type": "run"}
    return [(validate_record(record), raw)]


def retrieve_imported_runs(runs_root, structure, *, parameters=None, tasks=None, limit=4):
    """Recheck selected saved runs with the same matcher used for current history."""
    from .experience import retrieve_experience
    if runs_root is None or type(limit) is not int or limit <= 0:
        return []
    root = Path(runs_root).expanduser().resolve() / _LOCAL
    result, seen = [], set()
    for record in load_catalog(runs_root):
        if record["origin"] != "local" or record.get("import_type") != "run" or not record["evidence_valid"]:
            continue
        if record["status"] in {"rejected", "deprecated", "shadow_verified"}:
            continue
        try:
            source = _read_json(root / record["evidence"][0]["path"])
            path = Path(source["source_run"])
            current = retrieve_experience(path.parent, structure, parameters=parameters, tasks=tasks, run_paths=[path], limit=1)
            for case in current:
                if case["case_id"] not in seen:
                    case["source"] = "checked_imported_run"
                    result.append(case)
                    seen.add(case["case_id"])
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if len(result) >= min(limit, MAX_RESULTS):
            break
    return result

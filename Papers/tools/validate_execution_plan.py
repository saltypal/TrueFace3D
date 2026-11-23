"""Validate planning coverage, references and gate dependencies, not model behavior."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

PAPERS = Path(__file__).resolve().parents[1]
PLAN = PAPERS / "ExecutionPlan"


def main():
    errors = []
    corpus = json.loads((PAPERS / "evidence/corpus.json").read_text(encoding="utf-8"))
    expected = {paper["id"] for paper in corpus}
    matrix = (PLAN / "02_PAPER_ACTION_MATRIX.md").read_text(encoding="utf-8")
    entries = re.findall(r"^## (\d{2}_\w+)\s*$", matrix, re.MULTILINE)
    if set(entries) != expected or len(entries) != len(expected):
        errors.append("Paper action matrix must contain each corpus ID exactly once")
    state = json.loads((PLAN / "plan_state.json").read_text(encoding="utf-8"))
    if state["execution_days"] != 14 or state["retraining_in_14_day_scope"]:
        errors.append("Scope must remain fourteen pretrained-first days without retraining")
    if {c["id"] for c in state["components"]} != {f"C{i}" for i in range(1, 7)}:
        errors.append("Expected exactly six component IDs")
    gates = {g["id"]: g for g in state["gates"]}
    if len(gates) != len(state["gates"]):
        errors.append("Gate IDs must be unique")
    allowed_gate_states = set(state["gate_state_semantics"]["allowed_states"])
    for gate in gates.values():
        if gate["state"] not in allowed_gate_states:
            errors.append(f"Unknown gate state: {gate['id']}")
        references = list(gate.get("optional_inputs", {}))
        for conditional in gate.get("conditional_requirements", {}).values():
            references.extend(conditional)
        if any(reference not in gates for reference in references):
            errors.append(f"Unknown optional/conditional gate: {gate['id']}")
    visiting, visited = set(), set()

    def visit(identifier):
        if identifier not in gates:
            errors.append(f"Unknown gate dependency: {identifier}")
            return
        if identifier in visiting:
            errors.append(f"Cyclic gate dependency at {identifier}")
            return
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in gates[identifier]["requires"]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in gates:
        visit(identifier)
    for component in state["components"]:
        if component["gate"] not in gates or not (PLAN / component["design"]).is_file():
            errors.append(f"Missing component design or gate: {component['id']}")
    checked_links = 0
    markdown_files = sorted(PLAN.rglob("*.md"))
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        if len(re.findall(r"^```", content, re.MULTILINE)) % 2:
            errors.append(f"Unbalanced code fences: {path.name}")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"^[a-zA-Z]+://|^#", target):
                continue
            target = unquote(target.split("#", 1)[0].strip("<>"))
            checked_links += 1
            if not (path.parent / target).exists():
                errors.append(f"Missing local link in {path.name}: {target}")
    audit = json.loads((PLAN / "evidence/release_audit_2026-10-23.json").read_text(encoding="utf-8"))
    checks = audit.get("checks", audit.get("release_checks", []))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "static_plan_coverage_links_json_and_dependency_DAG_only",
        "paper_entries": len(entries),
        "components": len(state["components"]),
        "gates": len(gates),
        "markdown_documents": len(markdown_files),
        "checked_local_links": checked_links,
        "release_checks": len(checks),
        "plan_state_sha256": hashlib.sha256((PLAN / "plan_state.json").read_bytes()).hexdigest(),
        "errors": errors,
        "model_execution_validated": False,
        "external_link_health_validated_by_this_script": False,
    }
    (PLAN / "evidence/plan_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()

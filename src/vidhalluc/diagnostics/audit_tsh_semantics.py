"""TSH answer-format analysis and semantic-order extraction."""

import json, re, sys, importlib.util
from pathlib import Path
from ..paths import DATA_DIR, OFFICIAL_DIR


def semantic_order(text):
    s = text.strip().lower()
    clean = re.sub(r"\s+", "", s).strip(".,:;")
    if clean in ["a", "b", "ab", "ba"]:
        return clean.upper(), "exact"
    evidence = []
    labels = re.findall(
        r"^\s*(?:\d+[.)]|[-*])\s*(?:\*\*)?\s*action\s+([ab])\b", s, re.M
    )
    if len(labels) == 2 and len(set(labels)) == 2:
        evidence.append("".join(labels).upper())
    for relation in ["before", "after"]:
        pairs = re.findall(
            r"\baction\s+([ab])\b[^\n.]{0,80}?\b"
            + relation
            + r"\b[^\n.]{0,40}?\baction\s+([ab])\b",
            s,
        )
        for a, b in pairs:
            if a != b:
                evidence.append((a + b if relation == "before" else b + a).upper())
    if len(set(evidence)) == 1:
        return evidence[0], "consistent numbered list / explicit relation"
    if len(set(evidence)) > 1:
        return None, "conflicting statements"
    labels = re.findall(r"\baction\s+([ab])\b", s)
    if len(labels) == 2 and len(set(labels)) == 2:
        return "".join(labels).upper(), "single mention of each action"
    if len(set(labels)) == 1 and not re.search(
        r"\b(not|never|neither|unclear|absent)\b|no clear", s
    ):
        return labels[0].upper(), "one action mentioned, no explicit negation"
    return None, "unresolved"


def main():
    out = Path(sys.argv[1])
    gt = json.loads((DATA_DIR / "tsh.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "official_tsh", OFFICIAL_DIR / "eval_tsh.py"
    )
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    total = resolved = correct = official_correct = 0
    for path in sorted(out.glob("predictions_*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            if row["subset"] != "tsh":
                continue
            total += 1
            response = row["prediction"]["response"]
            target = gt[row["key"]]["Correct Answer"]
            answer, _ = semantic_order(response)
            official_correct += (
                official.model_answer_to_correct_answer(response) == target
            )
            resolved += answer is not None
            correct += answer == target
    report = dict(
        total=total,
        official_correct=official_correct,
        semantic_resolved=resolved,
        semantic_correct_resolved=correct,
        semantic_wrong_resolved=resolved - correct,
        semantic_unresolved=total - resolved,
    )
    (out / "tsh_semantic_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

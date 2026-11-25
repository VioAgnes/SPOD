"""MCQ answer-format analysis for compact multi-choice outputs."""

import json, re, sys
from pathlib import Path
from ..paths import DATA_DIR, OFFICIAL_DIR


def extract(text, choices, compact=False):
    if compact and re.fullmatch(r"\s*[A-D]{2,4}[.!]?\s*", text):
        return set(re.findall("[A-D]", text))
    result = set(re.findall(r"\b[A-D]\b", text))
    if result:
        return result
    for k, v in choices.items():
        if v.lower() in text.lower():
            return {k.upper()}
    return set()


def main():
    out = Path(sys.argv[1])
    gt = json.loads((DATA_DIR / "ach_mcq.json").read_text())
    counts = dict(standard_correct=0, compact_aware_correct=0, compact_only_correct=0)
    total = 0
    examples = []
    for path in sorted(out.glob("predictions_*.jsonl")):
        for row in map(json.loads, path.read_text().splitlines()):
            if row["subset"] != "ach_mcq":
                continue
            total += 1
            target = gt[row["key"]][row["video"]]
            gold = set(re.findall("[A-D]", target["Correct Answer"].upper()))
            response = row["prediction"]["response"]
            standard = extract(response, target["Choices"])
            compact = extract(response, target["Choices"], True)
            counts["standard_correct"] += standard == gold
            counts["compact_aware_correct"] += compact == gold
            if compact == gold and standard != gold:
                counts["compact_only_correct"] += 1
                examples.append(
                    dict(id=row["id"], response=response, gold=sorted(gold))
                )
    result = dict(
        total=total,
        counts=counts,
        examples=examples,
    )
    (out / "mcq_semantic_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

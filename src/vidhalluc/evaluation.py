"""VidHalluc scoring with local SimCSE weights and cached CLS embeddings."""

import argparse, ast, copy, json, math, types
from pathlib import Path
from .paths import DATA_DIR, MODEL_DIR, OFFICIAL_DIR
from .config import load_benchmark, load_config
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

OFF = OFFICIAL_DIR


def official(name):
    path = OFF / f"eval_{name}.py"
    tree = ast.parse(path.read_text())
    # Keep the official functions unchanged, avoid CLI execution/network model loading.
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    module = types.ModuleType("official_" + name)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


def parse_sth(text, gt):
    # Exact official main() parsing; malformed classification maps to No downstream.
    if ", Locations: " in text:
        a, b = text.split(", Locations: ", 1)
        pred = a.split(": ", 1)[-1].split(",")[0].strip()
        locations = b.strip()
    else:
        pred = text.split(": ", 1)[-1].split(",")[0].strip()
        locations = ""
    return {
        "GT Scene change": gt["Scene change"],
        "GT Locations": gt["Locations"],
        "Model Scene change": pred,
        "Model Locations": locations,
    }


@torch.inference_mode()
def main(a=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--config", default="configs/spod.json")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args() if a is None else a
    out = Path(a.directory)
    doc = load_benchmark()
    name = load_config(a.config)["name"]
    names = [name]
    pred = {}
    for path in sorted(out.glob("predictions_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r["id"] not in pred, r["id"]
            pred[r["id"]] = r
    expected = {r["id"] for r in doc["rows"]}
    assert set(pred) <= expected
    complete = {
        s: all(r["id"] in pred for r in doc["rows"] if r["subset"] == s)
        for s in doc["counts"]
    }
    if not a.partial:
        assert set(pred) == expected, ("Missing inputs", len(expected - set(pred)))
    mods = {s: official(s) for s in ["bqa", "mcq", "tsh", "sth"]}
    data = DATA_DIR
    gt = {s: json.loads((data / f"{s}.json").read_text()) for s in doc["counts"]}
    exports = {}
    sth_samples = {}
    summary = {n: {} for n in names}
    scores = {n: {} for n in names}
    for name in names:
        exp = {s: copy.deepcopy(gt[s]) for s in doc["counts"] if complete[s]}
        for r in doc["rows"]:
            s = r["subset"]
            if not complete[s]:
                continue
            v = pred[r["id"]]["prediction"]
            text = v["response"].strip()
            assert (
                v["all_valid_steps_intervened"]
                and v["prefill_changed"] == v["prefill_tokens"]
            )
            if s == "ach_binaryqa":
                exp[s][r["key"]][r["qi"]]["a"][r["video"]] = text
            elif s == "ach_mcq":
                exp[s][r["key"]][r["video"]]["Model Answer"] = text
            elif s == "tsh":
                exp[s][r["key"]]["Model Answer"] = text
            else:
                exp[s][r["key"]] = text
        dest = out / "official_inputs" / name
        dest.mkdir(parents=True, exist_ok=True)
        for s, obj in exp.items():
            (dest / f"{s}.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2)
            )
        exports[name] = exp
        if complete["ach_binaryqa"]:
            total, correct = mods["bqa"].compute(
                gt["ach_binaryqa"], exp["ach_binaryqa"]
            )
            summary[name]["BQA"] = 100 * correct / total
            scores[name]["BQA"] = dict(
                correct=correct,
                total=total,
                unit="all clips per question must be correct",
            )
        if complete["ach_mcq"]:
            mods["mcq"].match_model_answer(exp["ach_mcq"])
            total, correct = mods["mcq"].compute_MCQ(exp["ach_mcq"])
            summary[name]["MCQ"] = 100 * correct / total
            scores[name]["MCQ"] = dict(correct=correct, total=total)
        if complete["tsh"]:
            rows = exp["tsh"].values()
            correct = sum(
                mods["tsh"].model_answer_to_correct_answer(r["Model Answer"])
                == r["Correct Answer"]
                for r in rows
            )
            total = len(exp["tsh"])
            summary[name]["TSH"] = 100 * correct / total
            scores[name]["TSH"] = dict(correct=correct, total=total)
        if complete["sth"]:
            sth_samples[name] = [
                parse_sth(exp["sth"][key], val) for key, val in gt["sth"].items()
            ]
    if complete["sth"]:
        mp = MODEL_DIR / "sup-simcse-roberta-large"
        model = AutoModel.from_pretrained(str(mp)).to("cuda").eval()
        tok = AutoTokenizer.from_pretrained(str(mp))
        sentences = set()
        for rs in sth_samples.values():
            for r in rs:
                if (
                    r["GT Scene change"].lower()
                    == r["Model Scene change"].lower()
                    == "yes"
                ):
                    for field in ["GT Locations", "Model Locations"]:
                        scenes = mods["sth"].extract_scenes(r[field])
                        sentences.update(s for s in scenes if s)
        sentences = sorted(sentences)
        cache = {}
        for i in range(0, len(sentences), 16):
            ss = sentences[i : i + 16]
            inp = tok(ss, return_tensors="pt", padding=True, truncation=True).to("cuda")
            emb = model(**inp).last_hidden_state[:, 0, :].float().cpu().numpy()
            for s, e in zip(ss, emb):
                cache[s] = e[None, :]
        # Validate batching against the official single-sentence embedding path.
        max_error = 0.0
        for s in sentences[:5]:
            inp = tok(s, return_tensors="pt", truncation=True).to("cuda")
            e = model(**inp).last_hidden_state[:, 0, :].float().cpu().numpy()
            err = float(np.max(np.abs(e - cache[s])))
            max_error = max(max_error, err)
            assert np.allclose(e, cache[s], atol=3e-5, rtol=3e-4), err
        mods["sth"].get_simcse_embedding = lambda s: cache[s]
        for name, rs in sth_samples.items():
            metric = mods["sth"].evaluate_model(
                rs, similarity_threshold_high=0.8, similarity_threshold_low=0.5
            )
            summary[name]["STH"] = 100 * metric["Overall Score"]
            scores[name]["STH"] = metric | dict(
                total=len(rs),
                malformed_scene_fields=sum(
                    r["Model Scene change"].lower() not in ["yes", "no"] for r in rs
                ),
                embedding_batch_max_abs_error=max_error,
            )
        buf = __import__("io").BytesIO()
        np.savez_compressed(
            buf,
            sentences=np.array(sentences),
            embeddings=np.concatenate([cache[s] for s in sentences]),
        )
        (out / "simcse_embeddings.npz").write_bytes(buf.getvalue())
    for name in names:
        if all(complete.values()):
            summary[name]["Overall"] = (
                sum(summary[name][k] for k in ["BQA", "MCQ", "STH", "TSH"]) / 4
            )
    result = dict(
        complete=all(complete.values()),
        completed_subsets=complete,
        received_predictions=len(pred),
        expected_predictions=len(expected),
        metrics=summary[name],
        details=scores[name],
        method=load_config(a.config),
        protocol={
            "official_commit": doc["official_revision"],
            "STH_formula": "100*(0.6*((MCC+1)/2)^2 +0.4*normalized_description_score), SimCSE threshold0.5",
            "Overall": "Unweighted arithmetic mean of BQA,MCQ,STH,TSH",
            "sampling": "16 uniform frames; native Qwen video; max_pixels200704; greedy generation max256",
        },
    )
    fn = (
        "official_metrics.json"
        if result["complete"]
        else "partial_official_metrics.json"
    )
    (out / fn).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

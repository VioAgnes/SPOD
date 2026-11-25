"""SPOD rank-8 video hallucination inference."""

import argparse, json, time, contextlib, traceback
from pathlib import Path
from .paths import DATA_DIR, MODEL_DIR, DECODER_CACHE, OUTPUT_DIR
from .config import load_benchmark, load_config
from .decoding.decoder_cache import reader as VideoReader
from collections import defaultdict
import numpy as np
import torch
from decord import cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .interventions.evidence import PromptEvidence
from .interventions.method import Intervention


@torch.inference_mode()
def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR / "default"))
    ap.add_argument("--config", default="configs/spod.json")
    ap.add_argument("--prompts-per-batch", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    args = ap.parse_args() if args is None else args
    doc = load_benchmark()
    method_config = load_config(args.config)
    if args.prompts_per_batch < 1 or not 0 <= args.shard < args.shards:
        ap.error("Invalid batch size or shard index")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predfile = out / f"predictions_{args.shard}.jsonl"
    rows = doc["rows"]
    if args.smoke:
        selected = []
        for subset in doc["counts"]:
            sub = [r for r in rows if r["subset"] == subset]
            selected.extend(sub[:8])
        rows = selected
    groups = defaultdict(list)
    for r in rows:
        groups[(r["folder"], r["video"])].append(r)
    keys = sorted(groups, key=lambda k: ({"STH": 1, "TSH": 2, "ACH": 0}[k[0]], k[1]))[
        args.shard :: args.shards
    ]
    rows = [r for key in keys for r in groups[key]]
    assert 0 <= args.shard < args.shards
    done = {}
    if predfile.exists():
        for line in predfile.read_text().splitlines():
            r = json.loads(line)
            assert r["id"] not in done
            done[r["id"]] = r
    assert set(done) <= set(r["id"] for r in rows)
    config = dict(
        arguments=vars(args),
        method=method_config,
        ids=[r["id"] for r in rows],
        frames=16,
        max_pixels=200704,
        max_new_tokens=256,
        official_revision=doc["official_revision"],
        protocol="Official prompts and complete annotations; same-video prompt batching, original-size visual encoding shared; full hidden positions and every valid decoding step edited before LM head; no KV rewriting.",
    )
    cp = out / f"config_{args.shard}.json"
    if cp.exists():
        assert json.loads(cp.read_text()) == config, "Resume config mismatch"
    else:
        cp.write_text(json.dumps(config, indent=2))
    for folder, vid in keys:
        assert (DATA_DIR / "videos" / folder / f"{vid}.mp4").is_file(), vid
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    mp = str(MODEL_DIR / "Qwen2.5-VL-7B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        mp, min_pixels=4 * 28 * 28, max_pixels=200704, use_fast=False
    )
    processor.tokenizer.padding_side = "left"
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    gen = dict(
        max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None
    )
    original = model.visual.forward
    visual_cache = {}

    def shared(h, grid_thw, **kw):
        n = len(grid_thw)
        grid = grid_thw[:1]
        assert torch.equal(grid_thw, grid.repeat(n, 1))
        if "features" not in visual_cache:
            count = int(grid[0].prod())
            assert h.shape[0] % count == 0
            for j in range(1, h.shape[0] // count):
                assert torch.equal(h[:count], h[j * count : (j + 1) * count])
            visual_cache["features"] = original(h[:count], grid_thw=grid, **kw)
            visual_cache["grid"] = grid.clone()
        assert torch.equal(grid, visual_cache["grid"])
        return visual_cache["features"].repeat(n, 1)

    model.visual.forward = shared
    model.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:, :],))
    started = time.time()
    initial = len(done)
    last_print = 0
    errors = []

    def canonical(ids):
        ids = ids.tolist()
        n = next((i + 1 for i, v in enumerate(ids) if v in eos), len(ids))
        return ids[:n]

    def intervene(inp, qids):
        # Collect fixed evidence without producing an unedited answer.
        with contextlib.ExitStack() as stack:
            evidence = []
            for index, question in enumerate(qids):
                prompt_input = type(inp)(
                    {
                        key: value[index : index + 1]
                        for key, value in inp.items()
                        if key in ["input_ids", "video_grid_thw"]
                    }
                )
                evidence.append(
                    stack.enter_context(
                        PromptEvidence(model, prompt_input, question, index, 1)
                    )
                )
            model(**inp, use_cache=False)
        with Intervention(model, evidence, method_config) as intervention:
            generated = model.generate(**inp, **gen)
        return generated, evidence, intervention

    for ki, (folder, vid) in enumerate(keys):
        tasks = [r for r in groups[(folder, vid)] if r["id"] not in done]
        if not tasks:
            continue
        visual_cache.clear()
        path = DATA_DIR / "videos" / folder / f"{vid}.mp4"
        try:
            vr = VideoReader(str(path), ctx=cpu(0), num_threads=2)
            frames = vr.get_batch(np.linspace(0, len(vr) - 1, 16, dtype=int)).asnumpy()
            fps = 16 / (len(vr) / vr.get_avg_fps())
            del vr
            decoder_meta = DECODER_CACHE / f"{vid}.json"
            decoder_info = (
                json.loads(decoder_meta.read_text())
                if decoder_meta.exists()
                else {"backend": "isolated Decord"}
            )
            for offset in range(0, len(tasks), args.prompts_per_batch):
                chunk = tasks[offset : offset + args.prompts_per_batch]
                texts = []
                qids = []
                for r in chunk:
                    texts.append(
                        processor.apply_chat_template(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "video", "video": str(path)},
                                        {"type": "text", "text": r["prompt"]},
                                    ],
                                }
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                    qids.append(
                        processor.tokenizer.encode(
                            r["prompt"], add_special_tokens=False
                        )
                    )
                inp = processor(
                    text=texts,
                    videos=[frames] * len(chunk),
                    fps=[fps] * len(chunk),
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                length = inp.input_ids.shape[1]
                generated, ev, hook = intervene(inp, qids)
                tokens = generated[:, length:]
                answers = processor.batch_decode(tokens, skip_special_tokens=True)
                for j, r in enumerate(chunk):
                    record = {
                        k: r[k] for k in ["id", "subset", "key", "video", "folder"]
                    }
                    record["attention"] = ev[j].audit
                    record["decoder"] = decoder_info
                    ids = canonical(tokens[j])
                    audit = hook.parts[j].records
                    assert len(audit) == tokens.shape[1]
                    assert audit[0]["shape"][1] == length and all(
                        a["shape"][1] == 1 for a in audit[1:]
                    )
                    changed = all(
                        a["changed"][0] == a["shape"][1] for a in audit[: len(ids)]
                    )
                    assert changed
                    record["prediction"] = dict(
                        response=answers[j].strip(),
                        new_tokens=len(ids),
                        cap=len(ids) == 256 and ids[-1] not in eos,
                        prefill_tokens=length,
                        prefill_changed=audit[0]["changed"][0],
                        valid_forward_count=len(ids),
                        all_valid_steps_intervened=changed,
                    )
                    with predfile.open("a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done[r["id"]] = record
                del inp, generated, tokens, hook, ev
                if len(done) - last_print >= 40:
                    elapsed = time.time() - started
                    print(
                        "PROGRESS",
                        len(done),
                        len(rows),
                        "videos",
                        ki + 1,
                        len(keys),
                        "seconds",
                        round(elapsed),
                        "prompts_per_sec",
                        round((len(done) - initial) / max(elapsed, 1), 2),
                        flush=True,
                    )
                    last_print = len(done)
        except Exception as exc:
            err = dict(
                video=vid,
                folder=folder,
                error=repr(exc),
                traceback=traceback.format_exc(),
            )
            errors.append(err)
            with (out / f"errors_{args.shard}.jsonl").open("a") as f:
                f.write(json.dumps(err) + "\n")
            print("VIDEO_ERROR", vid, repr(exc), flush=True)
            if isinstance(exc, AssertionError):
                raise
    status = dict(
        completed=len(done),
        expected=len(rows),
        errors=errors,
        seconds=time.time() - started,
        complete=len(done) == len(rows),
    )
    (out / f"completion_{args.shard}.json").write_text(json.dumps(status, indent=2))
    print("FINAL", json.dumps(status), flush=True)
    assert status["complete"], "Missing predictions; inspect errors and resume"


if __name__ == "__main__":
    main()

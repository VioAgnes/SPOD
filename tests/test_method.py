"""Numerical and output-format checks for the rank-8 method."""

import json
import ast
import contextlib
import io
import time
import traceback
from collections import defaultdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for numerical checks")
class ProjectionTests(unittest.TestCase):
    def test_single_method_inference_record(self):
        from vidhalluc.config import load_config
        import numpy as np

        class Inputs(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        class Processor:
            tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])

            def apply_chat_template(self, *args, **kwargs):
                return "prompt"

            def __call__(self, **kwargs):
                return Inputs(
                    input_ids=torch.tensor([[1, 3, 4]]),
                    attention_mask=torch.ones(1, 3),
                    video_grid_thw=torch.tensor([[1, 1, 1]]),
                    second_per_grid_ts=[1.0],
                )

            def batch_decode(self, *args, **kwargs):
                return ["AB"]

        class Model:
            def __init__(self):
                self.generations = self.prefills = 0
                self.visual = SimpleNamespace(forward=lambda *args, **kwargs: None)
                self.lm_head = SimpleNamespace(
                    register_forward_pre_hook=lambda hook: None
                )
                self.generation_config = SimpleNamespace(eos_token_id=2)

            def eval(self):
                return self

            def __call__(self, **kwargs):
                self.prefills += 1

            def generate(self, **kwargs):
                self.generations += 1
                return torch.tensor([[1, 3, 4, 7, 2]])

        class Evidence:
            def __init__(self, *args):
                self.audit = {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        class Method(Evidence):
            def __init__(self, *args):
                self.parts = [
                    SimpleNamespace(
                        records=[
                            dict(shape=[1, 3, 8], changed=[3]),
                            dict(shape=[1, 1, 8], changed=[1]),
                        ]
                    )
                ]

        class Reader:
            def __init__(self, *args, **kwargs):
                pass

            def __len__(self):
                return 16

            def get_avg_fps(self):
                return 16

            def get_batch(self, indices):
                return SimpleNamespace(asnumpy=lambda: np.zeros((16, 4, 4, 3)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "videos/TSH/clip.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            model = Model()
            doc = dict(
                official_revision="test",
                counts={"tsh": 1},
                rows=[
                    dict(
                        id="test",
                        subset="tsh",
                        key="clip",
                        video="clip",
                        folder="TSH",
                        prompt="question",
                    )
                ],
            )
            namespace = dict(
                torch=torch,
                np=np,
                argparse=__import__("argparse"),
                json=json,
                time=time,
                contextlib=contextlib,
                traceback=traceback,
                Path=Path,
                defaultdict=defaultdict,
                DATA_DIR=root,
                OUTPUT_DIR=root,
                MODEL_DIR=root,
                DECODER_CACHE=root / "cache",
                load_benchmark=lambda: doc,
                load_config=load_config,
                PromptEvidence=Evidence,
                Intervention=Method,
                VideoReader=Reader,
                cpu=lambda value: value,
                Qwen2_5_VLForConditionalGeneration=SimpleNamespace(
                    from_pretrained=lambda *a, **kw: model
                ),
                AutoProcessor=SimpleNamespace(
                    from_pretrained=lambda *a, **kw: Processor()
                ),
            )
            source = ast.parse(
                (ROOT / "src/vidhalluc/inference.py").read_text(encoding="utf-8")
            )
            main = next(
                node
                for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
            exec(
                compile(
                    ast.Module(body=[main], type_ignores=[]), "<inference-test>", "exec"
                ),
                namespace,
            )
            args = SimpleNamespace(
                config=str(ROOT / "configs/spod.json"),
                output_dir=str(root / "out"),
                prompts_per_batch=1,
                shard=0,
                shards=1,
                smoke=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["main"](args)
            record = json.loads((root / "out/predictions_0.jsonl").read_text())
            self.assertEqual(model.generations, 1)
            self.assertEqual(model.prefills, 1)
            self.assertEqual(record["prediction"]["response"], "AB")
            self.assertTrue(record["prediction"]["all_valid_steps_intervened"])
            self.assertNotIn("variants", record)

    def test_rank8_projection_and_full_token_coverage(self):
        from vidhalluc.config import load_config
        from vidhalluc.interventions.method import Intervention
        from vidhalluc.interventions.projection import decompose

        torch.manual_seed(42)
        prompt = SimpleNamespace(
            frame22=torch.randn(5, 48),
            text22=torch.randn(8, 48),
            visual22=torch.randn(40, 48),
            groups=5,
            weights={"fused": torch.ones(5) / 5},
            audit={},
        )
        config = load_config(ROOT / "configs/spod.json")
        method = Intervention(SimpleNamespace(), [prompt], config)
        temporal, prior = method.build_bases(prompt)
        self.assertEqual(
            temporal.shape[1], 12
        )  # Four temporal + eight spatial directions.
        self.assertEqual(prior.shape[1], 4)
        torch.testing.assert_close(
            temporal.T @ prior, torch.zeros(12, 4), atol=1e-5, rtol=0
        )
        hidden = torch.randn(1, 7, 48)
        hT, hP, hR = decompose(hidden[0], temporal, prior)
        torch.testing.assert_close(hT + hP + hR, hidden[0])
        expected = config["gT"] * hT + config["gP"] * hP + config["gR"] * hR
        output = method.final(None, (), SimpleNamespace(last_hidden_state=hidden))
        torch.testing.assert_close(
            output.last_hidden_state[0], expected, atol=1e-5, rtol=1e-5
        )
        self.assertEqual(method.pending[0][1].tolist(), [7])

    def test_attention_preserves_mask_and_restores_sdpa(self):
        from vidhalluc.interventions.attention import wrap_attention
        import torch.nn.functional as functional

        sdpa = functional.scaled_dot_product_attention
        q = k = torch.zeros(1, 1, 3, 4)
        v = torch.eye(3)[None, None]
        mask = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
        calls = {}
        config = {"video_gain": 1.25}

        def original():
            return functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        wrapped = wrap_attention(
            original, 8, [SimpleNamespace(visual=torch.tensor([0]))], config, calls
        )
        output = wrapped()
        self.assertIs(functional.scaled_dot_product_attention, sdpa)
        self.assertEqual(calls, {8: 1})
        torch.testing.assert_close(output[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(
            output[0, 0, 2], torch.tensor([1.25, 1.0, 1.0]) / 3.25
        )


class DiagnosticTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is required for the scoring entry point")
    def test_scoring_reads_single_prediction(self):
        import copy
        import importlib.util
        from vidhalluc.config import load_config

        spec = importlib.util.spec_from_file_location(
            "test_tsh", ROOT / "src/vidhalluc/evaluators/eval_tsh.py"
        )
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subsets = ("tsh", "sth", "ach_binaryqa", "ach_mcq")
            for subset in subsets:
                annotation = (
                    {"clip": {"Correct Answer": "AB"}} if subset == "tsh" else {}
                )
                (root / f"{subset}.json").write_text(
                    json.dumps(annotation), encoding="utf-8"
                )
            record = dict(
                id="tsh",
                subset="tsh",
                key="clip",
                prediction=dict(
                    response="AB",
                    all_valid_steps_intervened=True,
                    prefill_changed=3,
                    prefill_tokens=3,
                ),
            )
            (root / "predictions_0.jsonl").write_text(
                json.dumps(record), encoding="utf-8"
            )
            benchmark = dict(
                counts={s: 1 for s in subsets},
                official_revision="test",
                rows=[dict(id=s, subset=s, key="clip") for s in subsets],
            )
            namespace = dict(
                torch=torch,
                argparse=__import__("argparse"),
                json=json,
                Path=Path,
                copy=copy,
                DATA_DIR=root,
                load_benchmark=lambda: benchmark,
                load_config=load_config,
                official=lambda name: evaluator if name == "tsh" else SimpleNamespace(),
            )
            source = ast.parse(
                (ROOT / "src/vidhalluc/evaluation.py").read_text(encoding="utf-8")
            )
            main = next(
                node
                for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == "main"
            )
            exec(
                compile(
                    ast.Module(body=[main], type_ignores=[]), "<scoring-test>", "exec"
                ),
                namespace,
            )
            args = SimpleNamespace(
                directory=str(root),
                config=str(ROOT / "configs/spod.json"),
                partial=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["main"](args)
            result = json.loads((root / "partial_official_metrics.json").read_text())
            self.assertEqual(result["metrics"], {"TSH": 100.0})
            self.assertEqual(result["method"]["spatial_rank"], 8)

    def test_single_prediction_diagnostics(self):
        from vidhalluc.diagnostics import audit_mcq_semantics as mcq
        from vidhalluc.diagnostics import audit_tsh_semantics as tsh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ach_mcq.json").write_text(
                json.dumps(
                    {
                        "clip": {
                            "v": {
                                "Correct Answer": "AC",
                                "Choices": {"A": "first", "C": "third"},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tsh.json").write_text(
                json.dumps({"order": {"Correct Answer": "AB"}}), encoding="utf-8"
            )
            records = [
                dict(
                    id="m",
                    subset="ach_mcq",
                    key="clip",
                    video="v",
                    prediction={"response": "AC"},
                ),
                dict(id="t", subset="tsh", key="order", prediction={"response": "AB"}),
            ]
            (root / "predictions_0.jsonl").write_text(
                "\n".join(map(json.dumps, records)), encoding="utf-8"
            )
            for module in (mcq, tsh):
                with (
                    patch.object(module, "DATA_DIR", root),
                    patch.object(sys, "argv", ["audit", str(root)]),
                ):
                    module.main()
            self.assertEqual(
                json.loads((root / "mcq_semantic_audit.json").read_text())["counts"][
                    "compact_only_correct"
                ],
                1,
            )
            self.assertEqual(
                json.loads((root / "tsh_semantic_audit.json").read_text())[
                    "semantic_correct_resolved"
                ],
                1,
            )

"""Isolated video decoding with cached PyAV fallback."""

import sys, json, subprocess, tempfile, os, uuid, shutil
from pathlib import Path
from ..paths import DECODER_CACHE
from types import SimpleNamespace
import faulthandler

faulthandler.enable()
import numpy as np
import torch  # Load before Decord.
import decord

cache = DECODER_CACHE


class CachedReader:
    def __init__(self, path):
        with np.load(path) as a:
            self.data = {k: a[k] for k in a.files}

    def __len__(self):
        return int(self.data["count"])

    def get_avg_fps(self):
        return float(self.data["fps"])

    def get_batch(self, ids):
        assert np.array_equal(ids, self.data["indices"])
        return SimpleNamespace(asnumpy=lambda: self.data["frames"])


def reader(path, *args, **kw):
    cache.mkdir(parents=True, exist_ok=True)
    cp = cache / (Path(path).stem + ".npz")
    if cp.exists():
        print("DECODER_FALLBACK", str(path), str(cp), flush=True)
        return CachedReader(cp)
    print("DECODE_ISOLATED", str(path), flush=True)
    fd, tmp = tempfile.mkstemp(prefix="vidhalluc_frames_", suffix=".npz")
    os.close(fd)
    try:
        reason = None
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vidhalluc.decoding.decode_isolated",
                    str(path),
                    tmp,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            if r.returncode:
                reason = "Decord error: " + r.stderr[-500:]
        except subprocess.TimeoutExpired:
            reason = "Decord timeout after30s"
        if reason is None:
            return CachedReader(tmp)
        token = uuid.uuid4().hex
        staged = Path(tempfile.gettempdir()) / (Path(path).stem + "." + token + ".npz")
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "vidhalluc.decoding.decode_av_fallback",
                str(path),
                str(staged),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if r.returncode:
            raise RuntimeError("Fallback failed: " + r.stderr[-500:])
        meta = json.loads(staged.with_suffix(".json").read_text())
        meta["fallback_reason"] = reason
        meta["decoder_wrapper"] = "decoder_cache.py"
        staged.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        published = cache / (Path(path).stem + "." + token + ".npz")
        shutil.copyfile(staged, published)
        os.replace(published, cp)
        published_meta = cache / (Path(path).stem + "." + token + ".json")
        shutil.copyfile(staged.with_suffix(".json"), published_meta)
        os.replace(published_meta, cp.with_suffix(".json"))
        staged.unlink()
        staged.with_suffix(".json").unlink()
        print("DECODER_FALLBACK_CREATED", str(path), reason, flush=True)
        return CachedReader(cp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

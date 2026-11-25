"""Uniform video-frame sampling with PyAV."""

import sys, json
import av, numpy as np
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
meta = {
    "source": str(source),
    "backend": "PyAV fallback",
    "policy": "16 uniformly sampled decoded frames.",
}
frames = []
with av.open(str(source)) as container:
    stream = container.streams.video[0]
    meta["original_declared_count"] = stream.frames
    fps = float(stream.average_rate or 30.0)
    try:
        for f in container.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
    except Exception as exc:
        meta["decode_error"] = repr(exc)
assert frames, ("No decodable frames", str(source))
ids = np.linspace(0, len(frames) - 1, 16, dtype=int)
sample = np.stack([frames[i] for i in ids])
meta.update(
    count=len(frames), av_decoded_count=len(frames), fps=fps, indices=ids.tolist()
)
np.savez(dest, frames=sample, indices=ids, count=len(frames), fps=fps)
dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))

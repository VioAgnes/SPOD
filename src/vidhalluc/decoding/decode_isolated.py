"""Uniform video-frame sampling in an isolated Decord process."""

import sys, json
import numpy as np
from decord import VideoReader, cpu

v = VideoReader(sys.argv[1], ctx=cpu(0), num_threads=2)
n = len(v)
fps = v.get_avg_fps()
ids = np.linspace(0, n - 1, 16, dtype=int)
frames = v.get_batch(ids).asnumpy()
np.savez(sys.argv[2], frames=frames, indices=ids, count=n, fps=fps)

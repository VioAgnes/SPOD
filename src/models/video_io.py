"""
Video I/O and path mapping utilities for VideoMMMU dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import numpy as np

from decord import VideoReader, cpu


# ----------------------------------------------------------------------
# Dataset path mapping
# ----------------------------------------------------------------------

SUBJECT_TO_CATEGORY = {
    # Art
    "Art": "Art", "Art_Theory": "Art", "Design": "Art", "Music": "Art",
    # Science
    "Biology": "Science", "Chemistry": "Science", "Geography": "Science",
    "Math": "Science", "Physics": "Science",
    # Humanities
    "History": "Humanities", "Literature": "Humanities",
    "Sociology": "Humanities", "Psychology": "Humanities",
    # Engineering
    "Agriculture": "Engineering", "Architecture_and_Engineering": "Engineering",
    "Computer_Science": "Engineering", "Electronics": "Engineering",
    "Energy_and_Power": "Engineering", "Materials": "Engineering",
    "Mechanical_Engineering": "Engineering",
    # Medicine
    "Basic_Medical_Science": "Medicine", "Clinical_Medicine": "Medicine",
    "Diagnostics_and_Laboratory_Medicine": "Medicine", "Pharmacy": "Medicine",
    "Public_Health": "Medicine",
    # Business
    "Accounting": "Business", "Economics": "Business", "Finance": "Business",
    "Manage": "Business", "Marketing": "Business",
}


def extract_subject(sample_id: str) -> str:
    """Extract subject name from VideoMMMU sample ID.

    ID format: prefix_subject_number
    Examples:
        dev_Biology_3       -> Biology
        test_Computer_Science_1 -> Computer_Science
        new_Architecture_and_Engineering_2 -> Architecture_and_Engineering
    """
    parts = sample_id.split("_")
    subject_parts = parts[1:-1]
    return "_".join(subject_parts)


def get_cache_dir(subject: str) -> str:
    """Map subject name to VideoMMMU cache directory category."""
    return SUBJECT_TO_CATEGORY.get(subject, "Unknown")


def find_local_video(
    sample_id: str,
    video_root: str | Path,
    subset: Optional[str] = None,
    strict: bool = False,
) -> Optional[str]:
    """Find local video file for a VideoMMMU sample.

    Args:
        sample_id: VideoMMMU sample ID (e.g., "test_Biology_1").
        video_root: Root directory of VideoMMMU videos.
        subset: One of "perception", "comprehension", "adaptation".
            If None, searches all categories.
        strict: If True (for adaptation), raises error when video not found.
            If False, returns None when not found.

    Returns:
        Absolute path to the video file, or None if not found.

    Raises:
        FileNotFoundError: When strict=True and video is missing.
        ValueError: When strict=True and a question_only video is detected.
    """
    video_root = Path(video_root)
    subject = extract_subject(sample_id)
    category = get_cache_dir(subject)
    video_path = video_root / category / f"{sample_id}.mp4"

    if video_path.exists():
        if "question_only" in str(video_path):
            raise ValueError(
                f"[CRITICAL] question_only video should not be used for "
                f"VideoMMMU evaluation.\n  sample_id: {sample_id}\n  "
                f"path: {video_path}\n  Use real videos under "
                f"{video_root}/<category>/"
            )
        return str(video_path)

    if strict:
        raise FileNotFoundError(
            f"[CRITICAL] Video not found: {video_path}\n"
            f"  sample_id: {sample_id}, category: {category}\n"
            f"  Please download the video from the original source."
        )
    return None


# ----------------------------------------------------------------------
# Video loading
# ----------------------------------------------------------------------

def load_video(video_path: str, num_frames: int = 32) -> np.ndarray:
    """Load video and uniformly sample `num_frames` frames.

    Mirrors the uniform sampling strategy used in LLaVA-OneVision / Qwen2.5-VL.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    frame_idx = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(frame_idx)
    return frames.asnumpy() if hasattr(frames, "asnumpy") else np.asarray(frames)


# ----------------------------------------------------------------------
# Vision token length estimation (architecture-specific)
# ----------------------------------------------------------------------

def estimate_vision_seq_len(
    video_grid_thw,
    video_inputs,
    arch: Literal["qwen2_vl", "videoLlama2", "videoLlava", "llama3_vl"] = "qwen2_vl",
) -> int:
    """Estimate the number of vision tokens after vision encoder + merger.

    Different models use different vision encoders with different downsampling:
    - Qwen2.5-VL: SigLIP + VisionMerger (2x temporal pooling)
    - VideoLlama2: SigLIP + Q-Former/Merger (typically 14x spatial)
    - VideoLlava: CLIP + MLP (typically 14x spatial)
    - Llama-3.2-Vision: SigLIP + projection (14x spatial)

    Args:
        video_grid_thw: Grid dimensions from model's vision processor.
        video_inputs: Raw pixel values.
        arch: Model architecture type.

    Returns:
        Estimated number of vision tokens.
    """
    if video_grid_thw is not None and video_grid_thw.numel() >= 3:
        thw = video_grid_thw[0] if video_grid_thw.dim() == 2 else video_grid_thw
        T, H, W = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
        
        # Architecture-specific pooling factors
        if arch == "qwen2_vl":
            # Qwen2.5-VL: VisionMerger pools T by 2x
            return T * (H // 2) * (W // 2)
        else:
            # VideoLlama2, VideoLlava, Llama-3.2-Vision: SigLIP/CLIP with 14x pooling
            return T * (H // 14) * (W // 14)
    
    if video_inputs:
        T = sum(v.shape[0] for v in video_inputs)
        H, W = video_inputs[0].shape[2], video_inputs[0].shape[3]
        
        if arch == "qwen2_vl":
            return T * (H // 14) * (W // 14)
        else:
            return T * (H // 14) * (W // 14)
    
    return 0


def detect_model_arch(model_id: str) -> Literal["qwen2_vl", "videoLlama2", "videoLlava", "llama3_vl"]:
    """Detect model architecture from model ID.
    
    Args:
        model_id: HuggingFace model ID or local path.
        
    Returns:
        Architecture type string.
    """
    model_id_lower = model_id.lower()
    
    if "qwen" in model_id_lower and "vl" in model_id_lower:
        return "qwen2_vl"
    elif "videollama2" in model_id_lower or "videollama_2" in model_id_lower:
        return "videoLlama2"
    elif "videollava" in model_id_lower or "videolla_va" in model_id_lower:
        return "videoLlava"
    elif "llama" in model_id_lower and "vision" in model_id_lower:
        return "llama3_vl"
    else:
        # Default to qwen2_vl
        return "qwen2_vl"

"""
VideoMMMU Evaluation Template
=============================

Reference benchmark. See paper for details.

Benchmark overview
-----------------
VideoMMMU is the first benchmark that assesses knowledge acquisition from
educational videos. It evaluates how well LMMs learn new knowledge from videos
and apply that knowledge in practice.

Dataset composition:
  - 300 lecture-style videos covering 6 disciplines
  - 30 subjects across: Art, Business, Science, Medicine, Humanities, Engineering
  - 900 question-answer pairs (3 QA per video)
  - Three cognitive-level subsets:
      perception    Visual information extraction (ASR/OCR-based questions)
      comprehension  Concept understanding and problem-solving comprehension
      adaptation     Knowledge application to new scenarios

Key metric: Δknowledge
----------------------
VideoMMMU introduces Δknowledge, which measures a model's improvement after
watching the relevant educational video:

    Δknowledge = (Acc_after - Acc_before) / (100% - Acc_before) × 100%

where Acc_before is measured on the "question_only" baseline (2-second image-only
videos containing just the question image).

Evaluation procedure
--------------------
1. Load video frames uniformly at NUM_FRAMES intervals
2. Construct MCQ prompt from question + options
3. Run model inference (SPOD or baseline)
4. Extract answer letter from response (A/B/C/D)
5. Compare against ground-truth answer and compute accuracy

For adaptation track: the question image is appended as the last frame of the
video with a prompt indicating its location.

Question format
---------------
Options are provided as choice dictionaries or plain strings:
  [{"choice": "A", "text": "option text"}, ...]
  or ["option A", "option B", "option C", "option D"]

Full prompt format:
  "{question}\nA. {option_A}\nB. {option_B}\nC. {option_C}\nD. {option_D}"

Answer extraction:
  1. Match first uppercase letter [A-D]
  2. Otherwise, match first letter of response against A/B/C/D
  3. Return the matched letter or first token

File structure (VideoMMMU parquet files)
----------------------------------------
video_root/
  Perception/    test-00000-of-00001.parquet
  Comprehension/ test-00000-of-00001.parquet
  Adaptation/    test-00000-of-00001.parquet

Parquet schema (HuggingFace dataset columns):
  id        str   - Sample ID: "{split}_{subject}_{number}"
  video     str   - Video URL or local path
  question  str   - Question text
  options   list  - List of choice dicts or strings
  answer    str   - Ground-truth answer letter: "A"/"B"/"C"/"D"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union


# =============================================================================
# Subset definitions
# =============================================================================

SUBSET_CONFIG: dict[str, dict] = {
    "perception": {
        "parquet": "Perception/test-00000-of-00001.parquet",
        "description": (
            "Perception — identify key information from video (ASR/OCR-based "
            "questions about visual elements, labels, figures)"
        ),
        "cognitive_level": "Perception",
        "example_tasks": ["Read text from slides", "Identify visible labels", "Count objects"],
    },
    "comprehension": {
        "parquet": "Comprehension/test-00000-of-00001.parquet",
        "description": (
            "Comprehension — understand underlying concepts and reasoning strategies "
            "from the educational video"
        ),
        "cognitive_level": "Comprehension",
        "example_tasks": ["Explain a concept", "Apply a formula", "Analyze cause and effect"],
    },
    "adaptation": {
        "parquet": "Adaptation/test-00000-of-00001.parquet",
        "description": (
            "Adaptation — apply knowledge learned from the video to solve "
            "new problems or analyze new case studies"
        ),
        "cognitive_level": "Adaptation",
        "example_tasks": [
            "Solve new exam questions",
            "Analyze unseen case studies",
            "Apply formulas to new scenarios",
        ],
        "note": (
            "The question image is appended as the last video frame. "
            "The prompt notes: 'The last frame contains the question image.'"
        ),
    },
}


# =============================================================================
# Question prompt construction
# =============================================================================

def make_question_prompt(
    question: str,
    options: list,
    adaptation_note: bool = False,
) -> str:
    """Build the full question prompt for VideoMMMU MCQ.

    Args:
        question: The question text.
        options: List of options. Each option can be:
            - dict with keys 'choice'/'text' (e.g. {"choice": "A", "text": "..."})
            - plain string (e.g. "Option A text")
        adaptation_note: If True, add a note that the last frame is the question image.

    Returns:
        Formatted prompt string.
    """
    if not options:
        prompt_text = question
    elif isinstance(options[0], dict):
        opt_lines = "\n".join(
            f"{o.get('choice', chr(65+i))}. {o.get('text', str(o))}"
            for i, o in enumerate(options)
        )
        prompt_text = f"{question}\n{opt_lines}"
    else:
        opt_lines = "\n".join(
            f"{chr(65+i)}. {o}" for i, o in enumerate(options)
        )
        prompt_text = f"{question}\n{opt_lines}"

    if adaptation_note:
        prompt_text += "\nNote: The last frame contains the question image."

    return prompt_text


# Default prompt builder
QUESTION_PROMPT_TEMPLATES = {
    "perception": lambda q, opts: make_question_prompt(q, opts, adaptation_note=False),
    "comprehension": lambda q, opts: make_question_prompt(q, opts, adaptation_note=False),
    "adaptation": lambda q, opts: make_question_prompt(q, opts, adaptation_note=True),
}


# =============================================================================
# Answer extraction
# =============================================================================

def extract_mcq_answer(response: Optional[str]) -> Optional[str]:
    """Extract the selected answer letter (A/B/C/D) from model response.

    Mirrors the original VideoMMMU evaluation logic:
    1. Match first uppercase letter [A-D] in response
    2. If no match, use the first character of the response

    Args:
        response: Raw model output string.

    Returns:
        One of "A", "B", "C", "D" or None.
    """
    if response is None or not response.strip():
        return None

    resp = response.strip()
    first_char = resp[0].upper()
    if first_char in ("A", "B", "C", "D", "E"):
        return first_char
    return first_char if first_char.isalpha() else None


# =============================================================================
# Metric definitions
# =============================================================================

def compute_accuracy(num_correct: int, num_total: int) -> float:
    """Compute accuracy as a fraction."""
    if num_total == 0:
        return 0.0
    return num_correct / num_total


def compute_delta_knowledge(
    acc_adaptation: float,
    acc_question_only: float,
) -> float:
    """Compute Δknowledge as defined in the VideoMMMU paper.

    Δknowledge = (Acc_after - Acc_before) / (100% - Acc_before) × 100%

    Args:
        acc_adaptation: Accuracy on the adaptation track (0.0–1.0).
        acc_question_only: Accuracy on the question_only baseline (0.0–1.0).

    Returns:
        Delta knowledge as a fraction (0.0–1.0).
    """
    if acc_question_only >= 1.0:
        return 0.0
    return (acc_adaptation - acc_question_only) / (1.0 - acc_question_only)


# Map subset -> metric name
METRIC_DEFINITIONS: dict[str, str] = {
    "perception": "Accuracy on perception-level questions",
    "comprehension": "Accuracy on comprehension-level questions",
    "adaptation": "Accuracy on adaptation-level questions (Acc_after)",
    "question_only": "Baseline accuracy without video context (Acc_before)",
    "delta_knowledge": "Knowledge gain metric: Δknowledge = (Acc_after - Acc_before) / (1 - Acc_before)",
}


# =============================================================================
# Parquet data helpers
# =============================================================================

def parse_options(options_field) -> list:
    """Parse the 'options' field from a VideoMMMU parquet row.

    Handles both JSON strings and already-parsed lists.
    """
    import json
    if isinstance(options_field, list):
        return options_field
    if isinstance(options_field, str):
        try:
            return json.loads(options_field)
        except Exception:
            return [options_field]
    return []


# =============================================================================
# Per-subset model config overrides
# =============================================================================

# Per-subset FDA/TDS hyperparameters (tuned for VideoMMMU)
SUBSET_MODEL_CONFIG: dict[str, dict] = {
    "perception": {
        "fda_layer_start": 9,
        "fda_layer_end": 28,
        "tds_deep_layer": 22,
        "temporal_dims": 16,
        "prior_dims": 8,
    },
    "comprehension": {
        "fda_layer_start": 9,
        "fda_layer_end": 25,
        "tds_deep_layer": 20,
        "temporal_dims": 16,
        "prior_dims": 8,
    },
    "adaptation": {
        "fda_layer_start": 9,
        "fda_layer_end": 22,
        "tds_deep_layer": 18,
        "temporal_dims": 12,
        "prior_dims": 6,
    },
}

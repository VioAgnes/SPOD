"""
VidHalluc Evaluation Template
=============================

Reference: https://github.com/CyL97/VidHalluc

Benchmark overview
-----------------
VidHalluc evaluates video hallucination in VideoLLMs across four subsets:
  ach_binaryqa   Action hallucination (binary QA)
  ach_mcq       Action hallucination (multiple choice)
  sth           State-of-things hallucination
  tsh           Transition-of-state hallucination

Each subset uses different question formats and answer extraction strategies.

Evaluation procedure (mirrors original benchmark)
--------------------------------------------------
1. Load video frames uniformly at NUM_FRAMES intervals
2. Construct question prompt from question template
3. Run model inference (SPOD or baseline)
4. Extract answer using subset-specific extraction strategy
5. Compare against ground truth and compute accuracy

Question formats
----------------
ach_binaryqa:
  Template: "{question}\nPlease answer yes or no."
  GT format: {"clip_id": "Yes"/"No"}
  Pred extraction: First word "Yes"/"No" (case-insensitive)

ach_mcq:
  Template: "{question}\n{choices}"
  GT format: {"clip_id": "A"/"B"/"C"/"D"}
  Pred extraction: First uppercase letter [A-D]

sth:
  Template: "{question}\nPlease answer yes or no."
  GT format: {"clip_id": "Yes"/"No"}
  Pred extraction: Same as ach_binaryqa

tsh:
  Template: "{question}\nPlease answer yes or no."
  GT format: {"clip_id": "Yes"/"No"}
  Pred extraction: Same as ach_binaryqa

File structure
-------------
question_root/
  ach_binaryqa.json  # {video_key: [{q, a: {clip_id: answer}}]}
  ach_mcq.json       # {video_key: [{q, a: {clip_id: answer}, choices: {A, B, C, D}}]}
  sth.json           # {video_key: [{q, a: {clip_id: answer}}]}
  tsh.json           # {video_key: [{q, a: {clip_id: answer}}]}

video_root/
  ACH_ffmpeg_fast/   # ach_binaryqa and ach_mcq videos
  ACH_reencoded_test/ # backup ach videos
  STH/               # sth videos
  TSH/               # tsh videos
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


# =============================================================================
# Subset definitions
# =============================================================================

SUBSET_FILES: dict[str, str] = {
    "ach_binaryqa": "ach_binaryqa.json",
    "ach_mcq": "ach_mcq.json",
    "sth": "sth.json",
    "tsh": "tsh.json",
}

SUBSET_DESCRIPTIONS: dict[str, str] = {
    "ach_binaryqa": "Action hallucination — binary Yes/No questions about object actions",
    "ach_mcq": "Action hallucination — multiple-choice questions with 4 options",
    "sth": "State-of-things hallucination — binary questions about persistent object states",
    "tsh": "Transition-of-state hallucination — binary questions about state changes",
}

VIDEO_SUBFOLDERS: dict[str, str] = {
    "ach_binaryqa": "ACH_ffmpeg_fast",
    "ach_mcq": "ACH_ffmpeg_fast",
    "sth": "STH",
    "tsh": "TSH",
}


# =============================================================================
# Question format templates
# =============================================================================

def make_binary_question(question: str) -> str:
    """Format a binary (Yes/No) question for VidHalluc subsets."""
    return f"{question.strip()}\nPlease answer yes or no."


def make_mcq_question(question: str, choices: dict[str, str]) -> str:
    """Format a multiple-choice question for ach_mcq subset.

    Args:
        question: The question text.
        choices: Dict mapping option letters to text, e.g. {"A": "...", "B": "...", ...}

    Returns:
        Formatted question string with choices.
    """
    choice_lines = "\n".join(f"{k}. {v}" for k, v in choices.items())
    return f"{question.strip()}\n{choice_lines}"


# =============================================================================
# Answer extraction strategies
# =============================================================================

def extract_binary_answer(response: Optional[str]) -> Optional[str]:
    """Extract Yes/No answer from model response for binary QA subsets.

    Mirrors the original VidHalluc extraction logic:
    - Returns 'Yes.' if response starts with 'yes' (case-insensitive)
    - Returns 'No.' if response starts with 'no' (case-insensitive)
    - Returns None if neither matches
    """
    if response is None:
        return None
    resp = response.strip().lower()
    if resp.startswith("yes"):
        return "Yes."
    if resp.startswith("no"):
        return "No."
    return None


def extract_mcq_answer(response: Optional[str]) -> Optional[str]:
    """Extract the selected answer letter from MCQ response.

    Mirrors the original VidHalluc eval_mcq.py logic:
    1. Search for pattern [A-D] as a whole word
    2. If not found, match choice text against response
    3. Return None if no match

    Returns:
        One of "A", "B", "C", "D" or None.
    """
    import re
    if response is None:
        return None
    resp = response.strip()

    # Strategy 1: match first [A-D] letter
    choices = re.findall(r'\b[A-D]\b', resp)
    if choices:
        return choices[0].upper()

    # Strategy 2: match choice text
    return None


def extract_ach_answer(response: Optional[str]) -> Optional[str]:
    """Alias for ach_binaryqa — uses binary extraction."""
    return extract_binary_answer(response)


def extract_sth_answer(response: Optional[str]) -> Optional[str]:
    """Alias for sth — uses binary extraction."""
    return extract_binary_answer(response)


def extract_tsh_answer(response: Optional[str]) -> Optional[str]:
    """Alias for tsh — uses binary extraction."""
    return extract_binary_answer(response)


# Map subset -> answer extraction function
ANSWER_EXTRACTION: dict[str, callable] = {
    "ach_binaryqa": extract_binary_answer,
    "ach_mcq": extract_mcq_answer,
    "sth": extract_sth_answer,
    "tsh": extract_tsh_answer,
}


# =============================================================================
# Question format configuration
# =============================================================================

@dataclass
class SubsetConfig:
    """Configuration for a VidHalluc subset."""
    name: str
    json_file: str
    video_subfolder: str
    question_template: callable  # (question, *args) -> formatted question string
    answer_extractor: callable   # (response) -> extracted answer
    metric: str                  # "accuracy" or "per-video accuracy"


@dataclass
class VidHallucConfig:
    """Full VidHalluc benchmark configuration."""
    subsets: dict[str, SubsetConfig]

    def get_subset(self, name: str) -> SubsetConfig:
        if name not in self.subsets:
            raise ValueError(f"Unknown subset: {name}. Available: {list(self.subsets.keys())}")
        return self.subsets[name]

    def get_subset_names(self) -> list[str]:
        return list(self.subsets.keys())


def build_vidhalluc_config() -> VidHallucConfig:
    """Build the standard VidHalluc evaluation configuration."""
    subsets = {
        "ach_binaryqa": SubsetConfig(
            name="ach_binaryqa",
            json_file="ach_binaryqa.json",
            video_subfolder="ACH_ffmpeg_fast",
            question_template=lambda q, **kw: make_binary_question(q),
            answer_extractor=extract_binary_answer,
            metric="accuracy",
        ),
        "ach_mcq": SubsetConfig(
            name="ach_mcq",
            json_file="ach_mcq.json",
            video_subfolder="ACH_ffmpeg_fast",
            question_template=lambda q, choices, **kw: make_mcq_question(q, choices),
            answer_extractor=extract_mcq_answer,
            metric="accuracy",
        ),
        "sth": SubsetConfig(
            name="sth",
            json_file="sth.json",
            video_subfolder="STH",
            question_template=lambda q, **kw: make_binary_question(q),
            answer_extractor=extract_sth_answer,
            metric="accuracy",
        ),
        "tsh": SubsetConfig(
            name="tsh",
            json_file="tsh.json",
            video_subfolder="TSH",
            question_template=lambda q, **kw: make_binary_question(q),
            answer_extractor=extract_tsh_answer,
            metric="accuracy",
        ),
    }
    return VidHallucConfig(subsets=subsets)


# Default config instance
vidhalluc_config = build_vidhalluc_config()


# Backward-compatible flat exports
QUESTION_FORMATS = {
    "ach_binaryqa": lambda q: make_binary_question(q),
    "ach_mcq": make_mcq_question,
    "sth": lambda q: make_binary_question(q),
    "tsh": lambda q: make_binary_question(q),
}

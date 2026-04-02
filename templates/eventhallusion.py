"""
EventHallusion Evaluation Template
=================================

Reference: https://github.com/Stevetich/EventHallusion

Benchmark overview
-----------------
EventHallusion evaluates event hallucination in VideoLLMs across three subsets:
  entire     Videos with a single consistent event throughout
  interleave  Videos with multiple interleaved events
  misleading  Videos designed to mislead using prior biases

Each split uses different GPT-4o judging prompts for description consistency.

Evaluation procedure (mirrors original benchmark)
------------------------------------------------
1. Load video frames uniformly at NUM_FRAMES intervals
2. Generate a detailed video description (excluding 'misleading' split)
3. Ask binary Yes/No questions about events in the video
4. For non-misleading: judge description consistency with GPT-4o
5. Compare answers and compute accuracy metrics

Binary QA prompts
-----------------
All subsets use the same binary question format:
  "{question}\nPlease answer yes or no."

GPT-4o judging prompts (for description consistency)
----------------------------------------------------
entire:
  System: Evaluate if model's description is consistent with ground-truth event.
  Prompt: "Imagine you are a referee... {model_output}\n{ground_truth_event}"
  Expected answer: "yes." or "no." (first word)

interleave:
  System: Determine if a specific event is mentioned in the model description.
  Prompt: "... {model_output}\n{unexpected_event}"
  Expected answer: "yes." or "no." (first word)

misleading:
  (No description generation or judging — purely binary QA)

File structure
--------------
video_root/
  videos/
    entire/      # entire_N.mp4
    interleave/  # interleave_N.mp4
    misleading/  # misleading_N.mp4

question_root/ (from EventHallusion/questions/)
  entire_questions.json      # [{id, questions: [{question, answer}], event_info: {...}}]
  mix_questions.json         # interleave subset
  misleading_questions.json   # misleading subset

Prediction output format
-------------------------
{
  "split_name": {
    "video_id": {
      "desc": "model description or ''",
      "judgement": "GPT-4o judgement text",
      "qa": [
        {"question": "...", "answer": "Yes.", "prediction": "Yes."},
        ...
      ]
    }
  }
}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


# =============================================================================
# Subset definitions
# =============================================================================

QUESTION_FILE_MAP: dict[str, str] = {
    "entire": "entire_questions.json",
    "interleave": "mix_questions.json",
    "misleading": "misleading_questions.json",
}

VIDEO_FILENAME_PREFIX: dict[str, str] = {
    "entire": "entire",
    "interleave": "interleave",
    "misleading": "misleading",
}

VIDEO_SUBFOLDER: dict[str, str] = {
    "entire": "videos/entire",
    "interleave": "videos/interleave",
    "misleading": "videos/misleading",
}

SPLIT_DESCRIPTIONS: dict[str, str] = {
    "entire": "Single-event videos — one consistent event throughout the entire video",
    "interleave": "Multi-event videos — multiple events interleaved in temporal order",
    "misleading": "Misleading videos — designed to activate prior biases that lead to incorrect answers",
}


# =============================================================================
# Binary QA extraction
# =============================================================================

def extract_yes_no(response: Optional[str]) -> Optional[str]:
    """Extract Yes/No answer from model response.

    Mirrors the original EventHallusion eval_binary.py logic:
    - Returns 'Yes.' if response starts with 'yes' (case-insensitive)
    - Returns 'No.' if response starts with 'no' (case-insensitive)
    - Returns None if neither matches

    Args:
        response: Raw model output string.

    Returns:
        'Yes.' or 'No.' on match, else None.
    """
    if response is None:
        return None
    resp = response.lower().strip()
    if resp.startswith("yes"):
        return "Yes."
    if resp.startswith("no"):
        return "No."
    return None


# Alias for compatibility
YES_NO_ANSWER = extract_yes_no


# =============================================================================
# GPT-4o judging prompts (from original EventHallusion constants.py)
# =============================================================================

MISLEADING_SYSTEM = (
    'Imagine you are a referee tasked with evaluating a model\'s output. '
    'The model will output a detailed description of a video. '
    'You will receive both the model\'s output and the ground truth event. '
    'Your task is to determine whether the model\'s description is consistent with the ground truth event. '
    'If you find any other descriptions unrelated to the ground truth event, answer "no." '
    'Otherwise, answer "yes." '
    'You need only focus on the consistency of the event and action. '
    'Do not judge the description of specific object, environment, atmosphere, and so on. '
    'Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. '
)

ENTIRE_SYSTEM = (
    'Imagine you are a referee tasked with evaluating a model\'s output. '
    'The model will output a detailed description of a video. '
    'You will receive both the model\'s output and a ground-truth event. '
    'Your task is to determine whether the event described in the model\'s output is consistent with the ground-truth event. '
    'If true, answer "yes." If it is not consistent with the ground-truth event, answer "no." '
    'You need only focus on the consistency of the event and action. '
    'Do not judge the description of specific object, environment, atmosphere, and so on. '
    'Please answer yes or no in the first word of your reply! Then, provide your analysis and reasoning. '
)

INTERLEAVE_SYSTEM = (
    'Imagine you are a referee tasked with evaluating a model\'s output. '
    'The model will output a detailed description of a video. '
    'You will receive the output of the tested model and a special event. '
    'You need to determine whether this special event is mentioned in the output of the model. '
    'If mentioned, you need to answer "yes", otherwise answer "no". '
    'You need only focus on the consistency of the event and action. '
    'Do not judge the description of specific object, environment, atmosphere, and so on. '
    'Please answer yes or no in the first word of your reply! Then, provide your analysis. '
)


def make_misleading_prompt(model_output: str, ground_truth_event: str) -> str:
    """Build GPT-4o judging prompt for 'misleading' split.

    Args:
        model_output: Model-generated video description.
        ground_truth_event: Ground-truth event caption from the dataset.

    Returns:
        Formatted prompt string.
    """
    return f"{MISLEADING_SYSTEM}Model output: {model_output}\nGround-truth event: {ground_truth_event}"


def make_entire_prompt(model_output: str, ground_truth_event: str) -> str:
    """Build GPT-4o judging prompt for 'entire' split.

    Args:
        model_output: Model-generated video description.
        ground_truth_event: Ground-truth event caption from the dataset.

    Returns:
        Formatted prompt string.
    """
    return f"{ENTIRE_SYSTEM}Model output: {model_output}\nGround-truth event: {ground_truth_event}"


def make_interleave_prompt(model_output: str, unexpected_event: str) -> str:
    """Build GPT-4o judging prompt for 'interleave' split.

    Args:
        model_output: Model-generated video description.
        unexpected_event: The event to check for in the description.

    Returns:
        Formatted prompt string.
    """
    return f"{INTERLEAVE_SYSTEM}Output: {model_output}\nUnexpected event: {unexpected_event}"


PROMPTS = {
    "entire": make_entire_prompt,
    "interleave": make_interleave_prompt,
    "misleading": make_misleading_prompt,
}


# =============================================================================
# Event info extraction helpers
# =============================================================================

def get_event_caption(event_info: dict) -> str:
    """Extract the primary event caption from event_info dict."""
    return event_info.get("caption", "")


def get_unexpected_event(event_info: dict) -> str:
    """Extract the unexpected/alternative event from event_info dict."""
    return event_info.get("unexpected", "")


# =============================================================================
# Evaluation configuration
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for an EventHallusion split."""
    name: Literal["entire", "interleave", "misleading"]
    question_file: str
    video_subfolder: str
    video_filename_prefix: str
    description: str
    has_description: bool  # Whether to generate description (False for misleading)
    has_judging: bool       # Whether to use GPT-4o judging (False for misleading)


EVENTHALLUSION_CONFIG: dict[str, SplitConfig] = {
    "entire": SplitConfig(
        name="entire",
        question_file="entire_questions.json",
        video_subfolder="videos/entire",
        video_filename_prefix="entire",
        description=SPLIT_DESCRIPTIONS["entire"],
        has_description=True,
        has_judging=True,
    ),
    "interleave": SplitConfig(
        name="interleave",
        question_file="mix_questions.json",
        video_subfolder="videos/interleave",
        video_filename_prefix="interleave",
        description=SPLIT_DESCRIPTIONS["interleave"],
        has_description=True,
        has_judging=True,
    ),
    "misleading": SplitConfig(
        name="misleading",
        question_file="misleading_questions.json",
        video_subfolder="videos/misleading",
        video_filename_prefix="misleading",
        description=SPLIT_DESCRIPTIONS["misleading"],
        has_description=False,
        has_judging=False,
    ),
}

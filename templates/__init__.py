"""
Evaluation Templates for SPOD Benchmark Evaluation
=================================================

This directory contains question templates and evaluation prompts for the three
benchmarks supported by SPOD:

  vidhalluc/    - VidHalluc (video hallucination benchmark)
  eventhallusion/ - EventHallusion (event hallucination benchmark)
  videommmu/    - VideoMMMU (multi-discipline video QA benchmark)

Each subdirectory mirrors the original benchmark structure for easy reference
and reproducibility. See subdirectory READMEs for details.
"""

from .vidhalluc import (
    SUBSET_FILES,
    SUBSET_DESCRIPTIONS,
    QUESTION_FORMATS,
    ANSWER_EXTRACTION,
)

from .eventhallusion import (
    QUESTION_FILE_MAP,
    VIDEO_FILENAME_PREFIX,
    SPLIT_DESCRIPTIONS,
    PROMPTS,
    YES_NO_ANSWER,
)

from .videommmu import (
    SUBSET_CONFIG,
    QUESTION_PROMPT_TEMPLATES,
    METRIC_DEFINITIONS,
)

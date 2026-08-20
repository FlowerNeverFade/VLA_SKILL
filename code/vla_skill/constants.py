from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


# Keep large datasets and checkpoints outside Git.  Set VLA_SKILL_ROOT when
# the data workspace is different from the repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("VLA_DATA_ROOT", os.environ.get("VLA_SKILL_ROOT", REPO_ROOT)))

DEFAULT_BASE_MODEL_PATH = _path_from_env("VLA_BASE_MODEL_PATH", DATA_ROOT / "model" / "pi05_base")
DEFAULT_SKILL_ROOT = _path_from_env("VLA_SKILL_DIR", DATA_ROOT / "skill")
DEFAULT_OUTPUT_ROOT = _path_from_env("VLA_OUTPUT_ROOT", DATA_ROOT / "outputs" / "pi05_skill_lora")
DEFAULT_LOCAL_TOKENIZER_PATH = _path_from_env(
    "PI05_TOKENIZER_PATH", DATA_ROOT / "model" / "google" / "paligemma-3b-pt-224"
)
DEFAULT_TOKENIZER_REPO = "leo009/paligemma-3b-pt-224"

DEFAULT_SEED = 1000
DEFAULT_CHUNK_SIZE = 50
DEFAULT_N_ACTION_STEPS = 50

MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
IMAGE_RESOLUTION = (224, 224)

IMAGE_FIELD_BY_CAMERA = {
    "base_0_rgb": "observation.images.base_0_rgb",
    "left_wrist_0_rgb": "observation.images.left_wrist_0_rgb",
    "right_wrist_0_rgb": "observation.images.right_wrist_0_rgb",
}

MODEL_CAMERA_NAMES = tuple(IMAGE_FIELD_BY_CAMERA.keys())
DEFAULT_SKILL_CAMERA_NAMES = MODEL_CAMERA_NAMES
CAMERA_NAMES = DEFAULT_SKILL_CAMERA_NAMES
IMAGE_FIELDS = tuple(IMAGE_FIELD_BY_CAMERA.values())

LORA_GROUPS = ("A", "B", "C", "D")
LORA_DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
}

EXPECTED_LORA_TARGET_COUNTS = {
    "A": 36,
    "B": 72,
    "C": 130,
    "D": 4,
}

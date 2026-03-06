import os

DATASET_ROOT = os.environ.get("DATASET_ROOT", "../")

JSONS_DIR = os.path.join(DATASET_ROOT, "data", "jsons")

SPLITS_DIR = os.path.join(DATASET_ROOT, "data", "splits")

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")

BIAS_LABEL_MAP = {
    0: "left",
    1: "center",
    2: "right",
}

BIAS_TEXT_MAP = {
    "left": 0,
    "center": 1,
    "right": 2,
}

LOG_LEVEL = "INFO"

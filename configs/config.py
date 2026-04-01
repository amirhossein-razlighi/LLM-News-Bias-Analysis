import os
from pathlib import Path

# Default to the NLP_Project directory (parent of this configs/ folder)
_DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)
DATASET_ROOT = os.environ.get("DATASET_ROOT", _DEFAULT_ROOT)

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

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CONTENT_TRUNCATE_CHARS = 512

EMBEDDING_TEXT_FIELD = "title_content"

# Community detection settings (incident clustering)
# cosine similarity threshold for grouping articles into incidents
COMMUNITY_DETECTION_THRESHOLD = 0.70
MIN_COMMUNITY_SIZE = 3                  # minimum articles per incident cluster
# minimum distinct leanings (left/center/right) per incident
MIN_BIAS_DIVERSITY = 2
# cap on articles per leaning class per incident
MAX_ARTICLES_PER_LEANING = 3

# Output filenames
INCIDENTS_JSONL = "prepared_incidents.jsonl"

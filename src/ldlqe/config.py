from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_COLUMNS = [
    "news_id",
    "title",
    "url",
    "publisher",
    "label",
    "story",
    "hostname",
    "timestamp",
]

LABEL_MAP = {
    "b": "business",
    "e": "entertainment",
    "m": "health",
    "t": "science_tech",
}

KEYWORD_RULES = {
    "business": [
        "fed",
        "stocks",
        "market",
        "bank",
        "shares",
        "oil",
        "economy",
        "trade",
        "profit",
        "investor",
    ],
    "entertainment": [
        "film",
        "movie",
        "star",
        "music",
        "show",
        "tv",
        "actor",
        "album",
        "hollywood",
        "festival",
    ],
    "health": [
        "cancer",
        "health",
        "disease",
        "doctor",
        "medical",
        "drug",
        "hospital",
        "virus",
        "study finds",
        "patient",
    ],
    "science_tech": [
        "google",
        "apple",
        "facebook",
        "microsoft",
        "tech",
        "robot",
        "software",
        "internet",
        "smartphone",
        "science",
    ],
}

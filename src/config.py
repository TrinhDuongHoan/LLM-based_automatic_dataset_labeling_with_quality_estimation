SENTIMENT_LABELS = ["negative", "neutral", "positive"]
TOPIC_LABELS = ["lecturer", "curriculum", "facility", "others"]
SEED = 42
TEXT_COL = "sentence"
SENTIMENT_COL = "sentiment"
TOPIC_COL = "topic"

MODEL_CANDIDATES = {
    "phobert": "vinai/phobert-base",
    "visobert": "uitnlp/visobert",
}
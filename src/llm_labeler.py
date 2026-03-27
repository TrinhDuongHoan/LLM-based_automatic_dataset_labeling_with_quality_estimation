from __future__ import annotations

import json
import re
import time
import pandas as pd
from openai import OpenAI
from .prompts import build_joint_prompt
from .metrics import evaluate_classification
from .config import SENTIMENT_LABELS, TOPIC_LABELS


def parse_json_object(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def run_llm_joint(
    train_df,
    val_df,
    model_name,
    api_base,
    api_key,
    shots=0,
    temperature=0.0,
):
    client = OpenAI(base_url=api_base, api_key=api_key)

    few_shot_rows = None
    if shots > 0:
        few_shot_rows = train_df.sample(min(shots, len(train_df)), random_state=42).to_dict("records")

    outputs = []
    latencies = []
    valid = 0

    for sentence in val_df["sentence"].tolist():
        prompt = build_joint_prompt(sentence, few_shot_rows=few_shot_rows)
        start = time.time()
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "Bạn là bộ gán nhãn dữ liệu chính xác."},
                {"role": "user", "content": prompt},
            ],
        )
        latency = time.time() - start
        latencies.append(latency)
    return merged, (sent_metrics, sent_report, sent_cm), (topic_metrics, topic_report, topic_cm), meta
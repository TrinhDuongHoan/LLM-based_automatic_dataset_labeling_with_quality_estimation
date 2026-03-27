from __future__ import annotations

import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from .config import TEXT_COL
from .metrics import evaluate_classification

def train_hf_classifier(
    train_df,
    val_df,
    task,
    labels,
    model_name,
    output_dir,
    epochs=4,
    batch_size=8,
    learning_rate=2e-5,
    max_length=256,
    use_fast=False,
):

    label2id = {label: i for i, label in enumerate(labels)}

    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true = [id2label[int(x)] for x in labels_np]
        y_pred = [id2label[int(x)] for x in preds]
        metrics, _, _ = evaluate_classification(y_true, y_pred, labels)
        return metrics

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    pred_output = trainer.predict(val_ds)
    pred_ids = pred_output.predictions.argmax(axis=-1)
    y_pred = [id2label[int(x)] for x in pred_ids]
    y_true = val_df[task].tolist()
    metrics, report, cm = evaluate_classification(y_true, y_pred, labels)
    return trainer, y_pred, metrics, report, cm
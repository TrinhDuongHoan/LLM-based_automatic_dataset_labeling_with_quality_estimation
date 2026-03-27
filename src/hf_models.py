from __future__ import annotations
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    train_data = train_df[[TEXT_COL, task]].rename(columns={task: "label"}).copy()
    val_data = val_df[[TEXT_COL, task]].rename(columns={task: "label"}).copy()
    train_data["label"] = train_data["label"].map(label2id)
    val_data["label"] = val_data["label"].map(label2id)

    train_ds = Dataset.from_pandas(train_data, preserve_index=False)
    val_ds = Dataset.from_pandas(val_data, preserve_index=False)

    def preprocess(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=max_length)

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
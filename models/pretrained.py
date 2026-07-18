from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, EarlyStoppingCallback

from ..src.pretrained.preprocessed import DataCollatorForMultipleChoice
from ..src.pretrained.utils import compute_metrics


def build_model(model_name_or_path, device):
    return AutoModelForMultipleChoice.from_pretrained(model_name_or_path).to(device)


def build_training_args(output_dir, use_bf16, seed):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",

        save_only_model=True,
        save_total_limit=1,

        learning_rate=1e-5,
        weight_decay=0.01,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        warmup_ratio=0.15,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.05,
        logging_steps=20,
        bf16=use_bf16,
        fp16=not use_bf16,
        load_best_model_at_end=True,
        metric_for_best_model="map3",
        greater_is_better=True,
        report_to="none",
        seed=seed,
    )


def build_trainer(model, training_args, train_ds, val_ds, tokenizer):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
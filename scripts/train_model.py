import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
import evaluate
import os
from pathlib import Path

# Set cache location BEFORE any Hugging Face operations

def build_datasets(data_dir: str) -> DatasetDict:
    """Load imagefolder dataset with train/val splits from the given directory."""
    ds = load_dataset(
        "imagefolder",
        data_dir=data_dir,
    )

    # Expecting train and val splits under dataset/train and dataset/val
    if not isinstance(ds, DatasetDict):
        raise ValueError("Expected a DatasetDict with 'train' and 'validation' or 'val' splits.")

    # Normalize possible split naming to 'train' and 'validation'
    if "validation" not in ds and "val" in ds:
        ds = DatasetDict({"train": ds["train"], "validation": ds["val"]})

    if "train" not in ds or "validation" not in ds:
        raise ValueError("Dataset must contain 'train' and 'validation' splits.")

    return ds


@dataclass
class ImageTransform:
    processor: Any

    def __call__(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        images = [Image.open(p).convert("RGB") if isinstance(p, str) else p for p in examples["image"]]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the plant leaf classifier using a local imagefolder dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Path to the dataset directory containing 'train' and 'val' subfolders.",
    )
    parser.add_argument("--output_dir", type=str, default="./my_leaf_model", help="Where to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Train/eval batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    base_model = "google/vit-base-patch16-224"

    # Load dataset
    dataset = build_datasets(args.data_dir)

    # Extract class names from dataset features
    class_names: List[str] = dataset["train"].features["label"].names
    num_labels = len(class_names)

    # Load processor and base model, adapting for new label space
    processor = AutoImageProcessor.from_pretrained(base_model)
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  # allow head resizing
    )

    label2id = {c: i for i, c in enumerate(class_names)}
    id2label = {i: c for i, c in enumerate(class_names)}
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Prepare transforms
    transform = ImageTransform(processor)

    def with_transform(example_batch):
        return transform(example_batch)

    train_ds = dataset["train"].with_transform(with_transform)
    eval_ds = dataset["validation"].with_transform(with_transform)

    # Metrics
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,  # Important for image inputs
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final model and processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Model and processor saved to {args.output_dir}")


if __name__ == "__main__":
    main()



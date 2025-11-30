#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


def load_tmdb_splits(
    movies_path: str,
    test_size: float = 0.3,
    val_fraction_of_temp: float = 0.5,
    seed: int = 42,
):
    """
    Load the TMDB movies CSV and create train/val/test splits exactly like the baseline:
      - drop rows with missing overview or vote_average
      - train: 70%, val: 15%, test: 15% (via 0.3 + 0.5)
    """
    movies = pd.read_csv(movies_path)

    data = movies.copy()
    data = data.dropna(subset=["vote_average", "overview"])
    data = data.reset_index(drop=True)

    train_df, temp_df = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_fraction_of_temp,
        random_state=seed,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Keep only what we actually need for this script
    cols = ["id", "overview", "vote_average"]
    train_df = train_df[cols]
    val_df = val_df[cols]
    test_df = test_df[cols]

    return train_df, val_df, test_df


def make_hf_datasets(train_df, val_df, test_df):
    """
    Wrap pandas DataFrames into a Hugging Face DatasetDict.
    """
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    return DatasetDict(
        train=train_ds,
        validation=val_ds,
        test=test_ds,
    )


def build_model_and_tokenizer(model_name: str):
    """
    Load tokenizer + model for regression.
    AutoModelForSequenceClassification with num_labels=1 + problem_type='regression'.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,  
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    return tokenizer, model



def tokenize_function_builder(tokenizer, max_length: int):
    def tokenize_function(batch):
        # batch["overview"] is a list of strings
        enc = tokenizer(
            batch["overview"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Regression label: vote_average
        # HF Trainer expects 'labels' field
        enc["labels"] = batch["vote_average"]
        return enc

    return tokenize_function


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds.squeeze(-1)
    labels = labels.astype(np.float32)

    mse = np.mean((preds - labels) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - labels)))

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TMDB rating prediction with a Hugging Face pretrained model + regression head."
    )
    parser.add_argument(
        "--movies_path",
        type=str,
        default="dataset/tmdb_5000_movies.csv",
        help="Path to tmdb_5000_movies.csv (same as baseline).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="HF model name or path (e.g. 'microsoft/deberta-v3-base', 'roberta-large').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hf_regressor_output",
        help="Where to save checkpoints, logs, etc.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Train/eval batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load data and splits (same logic as baseline)
    print("Loading TMDB dataset and creating splits...")
    train_df, val_df, test_df = load_tmdb_splits(
        movies_path=args.movies_path,
        seed=args.seed,
    )

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))

    # 2) Wrap into HF datasets
    raw_datasets = make_hf_datasets(train_df, val_df, test_df)

    # 3) Build tokenizer + model
    print(f"Loading model and tokenizer: {args.model_name}")
    tokenizer, model = build_model_and_tokenizer(args.model_name)

    # 4) Tokenize
    tokenize_fn = tokenize_function_builder(tokenizer, args.max_length)
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=["overview"],  # we only need the tokenized fields + labels
    )

    # 5) TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_steps=100,
        report_to="none",  # disable WandB/etc by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) Train
    print("Starting training...")
    trainer.train()

    # 7) Eval on validation + test
    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()

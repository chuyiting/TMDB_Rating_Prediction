#!/usr/bin/env python

import os
import ast
import argparse
from collections import Counter

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

# ---------------------------------------------------------------------
# Global defaults for cast usage
# ---------------------------------------------------------------------
TOP_K_CAST = 5       # top K cast members per movie based on "order"
TOP_N_ACTORS = 100   # keep only the top-N most frequent actors globally


# ---------------------------------------------------------------------
# Utilities for reading credits and building LLM text input
# ---------------------------------------------------------------------
def parse_cast(cast_str):
    """
    Parse the 'cast' column from tmdb_5000_credits.csv.

    The cast column is a JSON-like string of list[dict], each dict having keys
    like {'cast_id': ..., 'character': ..., 'name': ..., 'order': ...}.
    """
    if pd.isna(cast_str):
        return []
    try:
        return ast.literal_eval(cast_str)
    except (ValueError, SyntaxError):
        return []


def get_top_k_actors(cast_list, k):
    """
    From a parsed cast list (list of dicts), return up to K actor names ordered
    by the 'order' field (ascending).
    """
    if not isinstance(cast_list, list):
        return []
    cast_sorted = sorted(cast_list, key=lambda x: x.get("order", 1e9))
    top_k = cast_sorted[:k]
    return [member.get("name") for member in top_k if "name" in member]


def build_movies_with_cast_text(
    movies_path: str,
    credits_path: str,
    top_k_cast: int = TOP_K_CAST,
    top_n_actors: int = TOP_N_ACTORS,
):
    """
    Load movies + credits and build a DataFrame that includes:

      - id
      - overview
      - vote_average
      - top_cast_names: list[str] of top-K cast names that are in global top-N
      - text_for_llm: "Overview: ...\nCast: a, b, c, ..."

    This is the DataFrame we will then split into train/val/test.
    """
    # Load raw CSVs
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Base movie table: drop rows without overview or rating
    data = movies.copy()
    data = data.dropna(subset=["vote_average", "overview"])
    data = data.reset_index(drop=True)

    # Parse cast JSON from credits
    credits_parsed = credits.copy()
    credits_parsed["cast_parsed"] = credits_parsed["cast"].apply(parse_cast)

    # Build (movie_id, actor_name) for each movie's top-K ordered cast
    records = []
    for row in credits_parsed.itertuples(index=False):
        movie_id = row.movie_id
        top_actors = get_top_k_actors(row.cast_parsed, top_k_cast)
        for actor_name in top_actors:
            records.append({"movie_id": movie_id, "actor_name": actor_name})

    top_cast_df = pd.DataFrame(records)

    # Join with ratings to limit to movies that are in `data`
    movie_ratings = data[["id", "vote_average"]].rename(columns={"id": "movie_id"})
    merged = top_cast_df.merge(movie_ratings, on="movie_id", how="inner")

    # Global top-N actor names by frequency
    actor_counts = merged["actor_name"].value_counts()
    top_actor_names = set(actor_counts.head(top_n_actors).index)

    # Keep only actors in the top-N set
    valid_top_cast = top_cast_df[top_cast_df["actor_name"].isin(top_actor_names)]

    # For each movie, keep its list of top cast names intersected with top-N
    movie_top_cast = (
        valid_top_cast
        .groupby("movie_id")["actor_name"]
        .apply(list)
        .reset_index(name="top_cast_names")
    )

    # Merge back into `data`
    data_with_cast = data.merge(
        movie_top_cast,
        left_on="id",
        right_on="movie_id",
        how="left",
    )

    # Build final text_for_llm
    def make_text(row):
        overview = row["overview"] if isinstance(row["overview"], str) else ""
        names = row["top_cast_names"]
        if isinstance(names, list) and len(names) > 0:
            cast_str = ", ".join(names)
            return f"Overview: {overview}\nCast: {cast_str}"
        else:
            return f"Overview: {overview}"

    data_with_cast["text_for_llm"] = data_with_cast.apply(make_text, axis=1)

    return data_with_cast


# ---------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------
def load_tmdb_splits(
    movies_path: str,
    credits_path: str,
    test_size: float = 0.3,
    val_fraction_of_temp: float = 0.5,
    seed: int = 42,
    top_k_cast: int = TOP_K_CAST,
    top_n_actors: int = TOP_N_ACTORS,
):
    """
    Load movies + credits, add cast info into text_for_llm, then do train/val/test splits.

    - Drop rows with missing overview or vote_average
    - text_for_llm = overview + top-K cast names (restricted to top-N actors)
    - train: 70%, val: 15%, test: 15%
    """
    data_with_cast = build_movies_with_cast_text(
        movies_path=movies_path,
        credits_path=credits_path,
        top_k_cast=top_k_cast,
        top_n_actors=top_n_actors,
    )

    train_df, temp_df = train_test_split(
        data_with_cast,
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

    # Keep only what the HF pipeline needs
    cols = ["id", "text_for_llm", "vote_average"]
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


# ---------------------------------------------------------------------
# Model + tokenizer + metrics
# ---------------------------------------------------------------------
def build_model_and_tokenizer(model_name: str):
    """
    Load tokenizer + model for regression.

    NOTE: use_fast=False to avoid certain DeBERTa tokenizer issues.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,  # important for some DeBERTa variants
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        problem_type="regression",
    )
    return tokenizer, model


def tokenize_function_builder(tokenizer, max_length: int):
    def tokenize_function(batch):
        enc = tokenizer(
            batch["text_for_llm"],       # overview + cast
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Regression label: vote_average
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


# ---------------------------------------------------------------------
# Main: CLI + training
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TMDB rating prediction with a Hugging Face pretrained model + cast-aware input."
    )
    parser.add_argument(
        "--movies_path",
        type=str,
        default="dataset/tmdb_5000_movies.csv",
        help="Path to tmdb_5000_movies.csv.",
    )
    parser.add_argument(
        "--credits_path",
        type=str,
        default="dataset/tmdb_5000_credits.csv",
        help="Path to tmdb_5000_credits.csv.",
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
        default="./hf_cast_regressor_output",
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
    parser.add_argument(
        "--top_k_cast",
        type=int,
        default=TOP_K_CAST,
        help="Per-movie: use up to this many top-ordered cast members.",
    )
    parser.add_argument(
        "--top_n_actors",
        type=int,
        default=TOP_N_ACTORS,
        help="Globally: keep only this many most frequent actors.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 1) Load data and splits (movies + credits + cast-based text)
    print("Loading TMDB dataset and creating splits with cast-aware text...")
    train_df, val_df, test_df = load_tmdb_splits(
        movies_path=args.movies_path,
        credits_path=args.credits_path,
        seed=args.seed,
        top_k_cast=args.top_k_cast,
        top_n_actors=args.top_n_actors,
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
        remove_columns=["id", "text_for_llm"],  # labels are in "labels"
    )

    # 5) TrainingArguments & Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",    # eval (and print MSE) every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=100,              # training loss every 100 steps
        report_to="none",               # disable wandb/etc by default
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

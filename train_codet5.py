"""
Fine-tune Flan-T5-base on the Neo4j text2cypher dataset.

Usage:
    python train_codet5.py                        # train with defaults
    python train_codet5.py --epochs 10 --lr 5e-5  # override hyperparams
    python train_codet5.py --output_dir ./my_model # custom save path

Requirements:
    pip install transformers datasets accelerate sentencepiece
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "neo4j/text2cypher-2024v1"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256

INSTRUCTION_PREFIX = (
    "Generate Cypher statement to query a graph database.\n"
    "Schema: {schema}\nQuestion: {question}\nCypher: "
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Flan-T5 on text2cypher")
    p.add_argument("--output_dir", type=str, default="./flan_t5_text2cypher")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_input_len", type=int, default=MAX_INPUT_LEN)
    p.add_argument("--max_target_len", type=int, default=MAX_TARGET_LEN)
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    return p.parse_args()


def build_input_text(example):
    schema = example.get("schema", "")
    question = example.get("question", "")
    return INSTRUCTION_PREFIX.format(schema=schema, question=question)


def preprocess(examples, tokenizer, max_input_len, max_target_len):
    inputs = [
        INSTRUCTION_PREFIX.format(schema=s, question=q)
        for s, q in zip(examples["schema"], examples["question"])
    ]
    targets = examples["cypher"]

    model_inputs = tokenizer(
        inputs, max_length=max_input_len, padding=False, truncation=True
    )
    labels = tokenizer(
        targets, max_length=max_target_len, padding=False, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    args = parse_args()

    log.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    log.info("Loading dataset: %s", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME)

    # The dataset may have different split names; handle common cases
    if "train" in dataset:
        train_ds = dataset["train"]
    else:
        # Single split — do an 90/10 split
        split = dataset[list(dataset.keys())[0]].train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        dataset["validation"] = split["test"]

    eval_ds = dataset.get("validation") or dataset.get("test")

    log.info("Train size: %d, Eval size: %s", len(train_ds), len(eval_ds) if eval_ds else "N/A")

    # Check expected columns exist
    expected_cols = {"schema", "question", "cypher"}
    actual_cols = set(train_ds.column_names)
    if not expected_cols.issubset(actual_cols):
        log.warning(
            "Expected columns %s but got %s. Available: %s",
            expected_cols, expected_cols - actual_cols, actual_cols,
        )
        log.info("Sample row: %s", train_ds[0])
        raise SystemExit(
            "Dataset columns don't match expected format. "
            "Check the dataset and update the column names in this script."
        )

    log.info("Tokenizing dataset...")
    tokenize_fn = lambda examples: preprocess(
        examples, tokenizer, args.max_input_len, args.max_target_len
    )
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    eval_tokenized = (
        eval_ds.map(tokenize_fn, batched=True, remove_columns=eval_ds.column_names)
        if eval_ds
        else None
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=args.fp16,
        eval_strategy="epoch" if eval_tokenized else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if eval_tokenized else False,
        metric_for_best_model="eval_loss" if eval_tokenized else None,
        logging_steps=50,
        predict_with_generate=False,
        report_to="none",
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    log.info("Starting training...")
    trainer.train()

    log.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Done! Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

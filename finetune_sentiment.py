"""Fine-tune an instruction model for financial sentiment with QLoRA.

The heavy ML dependencies are imported only when training or evaluation starts,
so prompt and dataset helpers remain fast to test.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET = "takala/financial_phrasebank"
DEFAULT_DATASET_CONFIG = "sentences_allagree"
LABELS = ("negative", "neutral", "positive")
SYSTEM_PROMPT = (
    "Classify the sentiment of the financial news text. "
    "Respond with exactly one label: negative, neutral, or positive."
)


@dataclass(frozen=True)
class FineTuneConfig:
    model_name: str = DEFAULT_MODEL
    dataset_name: str = DEFAULT_DATASET
    dataset_config: str = DEFAULT_DATASET_CONFIG
    output_dir: str = "artifacts/qwen3-finance-sentiment"
    seed: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    max_length: int = 256
    epochs: float = 3.0
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def validated(self) -> "FineTuneConfig":
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size and validation_size must sum to less than 1")
        for name in (
            "max_length",
            "batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
            "lora_alpha",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be between 0 and 1")
        return self


def label_name(value: int | str) -> str:
    """Normalize Financial PhraseBank labels to their text representation."""

    if isinstance(value, int):
        try:
            return LABELS[value]
        except IndexError as exc:
            raise ValueError(f"Unknown sentiment label index: {value}") from exc
    normalized = value.strip().lower()
    if normalized not in LABELS:
        raise ValueError(f"Unknown sentiment label: {value!r}")
    return normalized


def training_messages(sentence: str, label: int | str) -> list[dict[str, str]]:
    text = sentence.strip()
    if not text:
        raise ValueError("sentence must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": label_name(label)},
    ]


def inference_messages(sentence: str) -> list[dict[str, str]]:
    text = sentence.strip()
    if not text:
        raise ValueError("sentence must not be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def normalize_prediction(text: str) -> str:
    """Return a valid label only when the generated answer is unambiguous."""

    normalized = text.strip().lower().strip(" .,:;!?\"'")
    if normalized in LABELS:
        return normalized
    matches = [label for label in LABELS if label in normalized.split()]
    return matches[0] if len(matches) == 1 else "unknown"


def classification_metrics(
    references: Iterable[str], predictions: Iterable[str]
) -> dict[str, object]:
    y_true = list(references)
    y_pred = list(predictions)
    if len(y_true) != len(y_pred):
        raise ValueError("references and predictions must have equal lengths")
    if not y_true:
        raise ValueError("at least one prediction is required")

    per_label = {}
    for label in LABELS:
        true_positive = sum(
            expected == label and predicted == label
            for expected, predicted in zip(y_true, y_pred)
        )
        false_positive = sum(
            expected != label and predicted == label
            for expected, predicted in zip(y_true, y_pred)
        )
        false_negative = sum(
            expected == label and predicted != label
            for expected, predicted in zip(y_true, y_pred)
        )
        precision = true_positive / (true_positive + false_positive or 1)
        recall = true_positive / (true_positive + false_negative or 1)
        f1 = 2 * precision * recall / (precision + recall or 1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(expected == label for expected in y_true),
        }

    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(LABELS),
        "unknown_predictions": sum(value == "unknown" for value in y_pred),
        "per_label": per_label,
    }


def load_and_split_dataset(config: FineTuneConfig):
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train",
        trust_remote_code=False,
    )
    split = dataset.train_test_split(
        test_size=config.test_size,
        seed=config.seed,
        stratify_by_column="label",
    )
    validation_fraction = config.validation_size / (1 - config.test_size)
    train_validation = split["train"].train_test_split(
        test_size=validation_fraction,
        seed=config.seed,
        stratify_by_column="label",
    )
    return train_validation["train"], train_validation["test"], split["test"]


def prepare_training_dataset(dataset, tokenizer):
    def render(example):
        return {
            "text": tokenizer.apply_chat_template(
                training_messages(example["sentence"], example["label"]),
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    return dataset.map(render, remove_columns=dataset.column_names)


def train(config: FineTuneConfig):
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    set_seed(config.seed)
    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    train_dataset, validation_dataset, test_dataset = load_and_split_dataset(config)
    train_dataset = prepare_training_dataset(train_dataset, tokenizer)
    validation_dataset = prepare_training_dataset(validation_dataset, tokenizer)

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    training_args = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=config.max_length,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=bf16_supported,
        fp16=not bf16_supported,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        seed=config.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )
    return trainer.model, tokenizer, test_dataset


def evaluate_model(model, tokenizer, dataset, batch_size: int = 16):
    import torch

    references = [label_name(value) for value in dataset["label"]]
    predictions = []
    model.eval()
    for start in range(0, len(dataset), batch_size):
        sentences = dataset[start : start + batch_size]["sentence"]
        prompts = [
            tokenizer.apply_chat_template(
                inference_messages(sentence),
                tokenize=False,
                add_generation_prompt=True,
            )
            for sentence in sentences
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        predictions.extend(
            normalize_prediction(text)
            for text in tokenizer.batch_decode(generated, skip_special_tokens=True)
        )
    return classification_metrics(references, predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--output-dir", default=FineTuneConfig.output_dir)
    parser.add_argument("--seed", type=int, default=FineTuneConfig.seed)
    parser.add_argument("--test-size", type=float, default=FineTuneConfig.test_size)
    parser.add_argument(
        "--validation-size", type=float, default=FineTuneConfig.validation_size
    )
    parser.add_argument("--max-length", type=int, default=FineTuneConfig.max_length)
    parser.add_argument("--epochs", type=float, default=FineTuneConfig.epochs)
    parser.add_argument(
        "--learning-rate", type=float, default=FineTuneConfig.learning_rate
    )
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=FineTuneConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--lora-rank", type=int, default=FineTuneConfig.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=FineTuneConfig.lora_alpha)
    parser.add_argument(
        "--lora-dropout", type=float, default=FineTuneConfig.lora_dropout
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    values = vars(args).copy()
    values.pop("skip_evaluation")
    return FineTuneConfig(**values).validated()


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    model, tokenizer, test_dataset = train(config)
    if not args.skip_evaluation:
        metrics = evaluate_model(model, tokenizer, test_dataset)
        metrics_path = Path(config.output_dir) / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

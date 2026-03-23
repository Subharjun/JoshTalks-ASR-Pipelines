"""
Q1 - Step 2: Fine-tune Whisper-small on Hindi ASR data
JoshTalks AI Researcher Intern Assessment

Uses HuggingFace Transformers + Datasets.
GPU recommended (free Colab T4 works).
Run: python finetune.py
"""

import os
import torch  # type: ignore
import evaluate  # type: ignore
import numpy as np  # type: ignore
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from datasets import DatasetDict, load_from_disk, Audio  # type: ignore
from transformers import (  # type: ignore
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME    = "openai/whisper-small"
LANGUAGE      = "Hindi"
TASK          = "transcribe"
DATA_DIR      = Path("data/processed/hindi_asr_dataset")
OUTPUT_DIR    = Path("results/whisper-small-hi")
SAMPLE_RATE   = 16_000
MAX_AUDIO_SEC = 30
BATCH_SIZE    = 16      # reduce to 8 if OOM on GPU
GRAD_ACCUM    = 2
LEARNING_RATE = 1e-5
WARMUP_STEPS  = 200
MAX_STEPS     = 4000    # ~3 epochs on ~10h data
EVAL_STEPS    = 500
SAVE_STEPS    = 500

# ── Device detection (CUDA > MPS > CPU) ──────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
    FP16   = True    # CUDA supports FP16
    BF16   = False
elif torch.backends.mps.is_available():
    DEVICE = "mps"   # Apple Silicon GPU (M1/M2/M3/M4)
    FP16   = False   # MPS does NOT support FP16 training
    BF16   = False   # BF16 not yet stable on MPS either
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    DEVICE = "cpu"
    FP16   = False
    BF16   = False
    print("⚠️  No GPU found — using CPU (training will be slow)")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── LOAD PROCESSOR ───────────────────────────────────────────────────────────
print("Loading Whisper processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=LANGUAGE, task=TASK
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE, task=TASK
)

# ─── DATA PREPARATION ─────────────────────────────────────────────────────────
def prepare_dataset(batch):
    """Convert audio URL + text into Whisper input features + labels."""
    # NOTE: For real training, audio must be downloaded locally.
    # Here we assume the dataset has a local 'audio' column with Audio feature.
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    ).input_features[0]

    # Tokenize the target text
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


# ─── DATA COLLATOR ────────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Input features
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Labels — pad to max length in batch
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token with -100 so loss ignores padding
        labels = labels_batch["input_ids"].masked_fill(  # type: ignore
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS if present (Whisper adds it during generation)
        if (
            labels[:, 0] == self.decoder_start_token_id  # type: ignore
        ).all():
            labels = labels[:, 1:]  # type: ignore

        batch["labels"] = labels
        return batch


# ─── WER METRIC ───────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 2)}


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ── Load dataset ──────────────────────────────────────────────────────────
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. Run preprocess.py first."
        )
    print(f"Loading dataset from {DATA_DIR}...")
    ds: DatasetDict = load_from_disk(str(DATA_DIR))

    # Add Audio column (resample to 16kHz)
    ds = ds.cast_column("audio_url", Audio(sampling_rate=SAMPLE_RATE))

    # Apply feature extraction
    ds = ds.map(
        prepare_dataset,
        remove_columns=ds.column_names["train"],
        num_proc=1,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading Whisper-small...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.generation_config.language  = LANGUAGE.lower()
    model.generation_config.task      = TASK
    model.generation_config.forced_decoder_ids = None

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=FP16,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"\nSaving best model to {OUTPUT_DIR}/best_model...")
    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    processor.save_pretrained(str(OUTPUT_DIR / "best_model"))
    print("Done.")


if __name__ == "__main__":
    main()

"""
Q1 - Step 3: Evaluate Whisper-small (baseline vs fine-tuned) on FLEURS Hindi
JoshTalks AI Researcher Intern Assessment

Outputs a structured WER comparison table.
Run: python evaluate.py
"""

import torch  # type: ignore
import evaluate  # type: ignore
import json
from pathlib import Path
from datasets import load_dataset, Audio  # type: ignore
from transformers import (  # type: ignore
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)
from tqdm import tqdm  # type: ignore

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BASELINE_MODEL  = "openai/whisper-small"
FINETUNED_MODEL = "results/whisper-small-hi/best_model"
SAMPLE_RATE     = 16_000
BATCH_SIZE      = 8
RESULTS_DIR     = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

wer_metric = evaluate.load("wer")


# ─── LOAD FLEURS HINDI TEST SET ───────────────────────────────────────────────
def load_fleurs_hindi():
    print("Loading FLEURS Hindi test split...")
    ds = load_dataset(
        "google/fleurs",
        "hi_in",
        split="test",
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"  {len(ds)} test samples loaded.")
    return ds


# ─── RUN INFERENCE WITH A MODEL ───────────────────────────────────────────────
def run_inference(model_id: str, test_ds, label: str) -> dict:
    """
    Run Whisper inference on all test samples.
    Returns dict with predictions, references, and WER.
    """
    print(f"\nEvaluating: {label} ({model_id})")

    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        device=device,
        generate_kwargs={"language": "hindi", "task": "transcribe"},
    )

    predictions, references = [], []

    for sample in tqdm(test_ds, desc=f"  [{label}]"):
        audio    = sample["audio"]["array"]
        ref_text = sample["transcription"]

        pred = asr_pipeline({"array": audio, "sampling_rate": SAMPLE_RATE})
        predictions.append(pred["text"].strip())
        references.append(ref_text.strip())

    wer = 100 * wer_metric.compute(
        predictions=predictions, references=references
    )

    return {
        "model":       label,
        "model_id":    model_id,
        "num_samples": len(references),
        "wer":         round(wer, 2),
        "predictions": predictions,
        "references":  references,
    }


# ─── PRINT WER TABLE ──────────────────────────────────────────────────────────
def print_wer_table(results: list[dict]):
    print("\n" + "=" * 60)
    print("  WER Evaluation on FLEURS Hindi Test Set")
    print("=" * 60)
    print(f"  {'Model':<35} {'WER (%)':>8}  {'Samples':>8}")
    print("-" * 60)
    for r in results:
        print(f"  {r['model']:<35} {r['wer']:>8.2f}  {r['num_samples']:>8}")
    print("=" * 60)

    # Delta
    if len(results) == 2:
        baseline_wer  = results[0]["wer"]
        finetuned_wer = results[1]["wer"]
        delta = baseline_wer - finetuned_wer
        print(f"\n  WER Reduction after fine-tuning: {delta:+.2f}%")
        if delta > 0:
            rel = delta / baseline_wer * 100
            print(f"  Relative improvement: {rel:.1f}%")
    print()


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
def save_results(results: list[dict]):
    out = []
    for r in results:
        out.append({
            "model":       r["model"],
            "wer":         r["wer"],
            "num_samples": r["num_samples"],
        })

    path = RESULTS_DIR / "wer_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")

    # Save predictions for error analysis
    for r in results:
        safe = r["model"].replace(" ", "_").replace("/", "-")
        pred_path = RESULTS_DIR / f"predictions_{safe}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for pred, ref in zip(r["predictions"], r["references"]):
                f.write(json.dumps(
                    {"prediction": pred, "reference": ref}, ensure_ascii=False
                ) + "\n")
        print(f"Predictions saved to {pred_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    test_ds = load_fleurs_hindi()

    results = []

    # 1. Baseline pretrained Whisper-small
    results.append(run_inference(
        BASELINE_MODEL, test_ds, label="Whisper-small (pretrained baseline)"
    ))

    # 2. Fine-tuned Whisper-small (only if available)
    if Path(FINETUNED_MODEL).exists():
        results.append(run_inference(
            FINETUNED_MODEL, test_ds, label="Whisper-small (fine-tuned on Hindi)"
        ))
    else:
        print(f"\n[INFO] Fine-tuned model not found at {FINETUNED_MODEL}.")
        print("       Run finetune.py first, then re-run evaluate.py.")

    print_wer_table(results)
    save_results(results)


if __name__ == "__main__":
    main()

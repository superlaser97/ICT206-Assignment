from __future__ import annotations

# Standard library
import json
import random
import sys
import subprocess
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch

# Hugging Face
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_ROOT / "dataset" / "mermaid_50k.jsonl"

MODEL_ID = "LiquidAI/LFM2.5-350M"

OUTPUT_ROOT = "outputs"
RUN_PREFIX = "run-"
RUN_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"

RESULTS_DIR = "results"
RESULTS_BATCH_SIZE = 8

# Sampling parameters
TEMPERATURE = 0.1
TOP_K = 50
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 256

# Seed used to randomize dataset rows
RANDOM_SEED = 206
# Percentage of dataset to use
DATA_USAGE_FRACTION = 0.14

# Divide the used rows in the ratio training : validation : evaluation
SPLIT_RATIOS = (30, 5, 1)

# LoRA hyperparameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Which layers get adapters.
LORA_TARGET_MODULES = [
    "w1",
    "w2",
    "w3",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "in_proj",
]

# --- Training hyperparameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2.0e-4
WARMUP_STEPS = 10
MAX_SEQUENCE_LENGTH = 1024
EVAL_STEPS = 50
SAVE_STEPS = 250

# =============================================================================
# 2. PROMPT
# =============================================================================
SYSTEM_INSTRUCTION = (
    "You convert process descriptions into Mermaid flowchart syntax. "
    "Reply with the Mermaid diagram only."
)

# The user turn. `{description}` is filled in per row. The trailing
# "Mermaid diagram:" line tells the model exactly where its answer begins.
PROMPT_TEMPLATE = (
    "Convert the following process description into Mermaid flowchart syntax.\n"
    "Reply with the Mermaid diagram only, with no explanation and no code fences.\n"
    "\n"
    "Description:\n"
    "{description}\n"
    "\n"
    "Mermaid diagram:\n"
)

# =============================================================================
# 3. HELPERS
# =============================================================================
def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dictionaries (one per line).

    JSONL = "JSON Lines": each line is a complete, self-contained JSON object.
    It streams nicely and one bad line never breaks the whole file for readers
    that skip it. Blank lines are ignored here for robustness.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_prompt(description: str) -> str:
    return PROMPT_TEMPLATE.format(description=description.strip())

def shuffled_rows() -> list[dict]:
    if not SOURCE_FILE.exists():
        sys.exit("Source dataset not found")

    rows = load_jsonl(SOURCE_FILE)
    random.Random(RANDOM_SEED).shuffle(rows)
    return rows


def split_sizes(total: int) -> tuple[int, int, int]:
    if not 0.0 < DATA_USAGE_FRACTION <= 1.0:
        sys.exit(f"DATA_USAGE_FRACTION must be in (0, 1], got {DATA_USAGE_FRACTION}.")
    if len(SPLIT_RATIOS) != 3 or any(weight <= 0 for weight in SPLIT_RATIOS):
        sys.exit(f"SPLIT_RATIOS must be three positive weights, got {SPLIT_RATIOS}.")

    used = min(total, round(total * DATA_USAGE_FRACTION))
    w_train, w_val, w_eval = SPLIT_RATIOS
    denominator = w_train + w_val + w_eval
    train_n = used * w_train // denominator
    val_n = used * w_val // denominator
    eval_n = used - train_n - val_n
    if min(train_n, val_n, eval_n) <= 0:
        sys.exit(
            f"Split too small: {used} used rows cannot be divided by "
            f"{SPLIT_RATIOS} into non-empty parts."
        )
    return train_n, val_n, eval_n


def prepare_datasets() -> tuple[Dataset, Dataset | None]:
    rows = shuffled_rows()
    print(f"Loaded {len(rows):,} rows from {SOURCE_FILE.name}")

    train_n, val_n, eval_n = split_sizes(len(rows))
    train_rows = rows[:train_n]
    val_rows = rows[train_n : train_n + val_n]
    print(
        f"Split: {len(rows):,} total -> use {train_n + val_n + eval_n:,} "
        f"(train {train_n:,} / val {val_n:,} / eval {eval_n:,}), seed {RANDOM_SEED}"
    )

    def to_conversation(row: dict) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": build_prompt(row["description"])},
                {"role": "assistant", "content": row["mermaid"]},
            ]
        }

    train_data = Dataset.from_list([to_conversation(r) for r in train_rows])
    eval_data = (
        Dataset.from_list([to_conversation(r) for r in val_rows]) if val_rows else None
    )
    return train_data, eval_data


def load_tokenizer_and_model():
    print(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if not use_bf16:
        print("WARNING: no CUDA GPU found; falling back to CPU in float32 (slow).")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",   # spread layers across available GPUs / CPU RAM
        dtype=compute_dtype,
    )
    return tokenizer, model, use_bf16


# =============================================================================
# 4. Main process
# =============================================================================
def main() -> None:
    started = datetime.now()
    run_name = f"{RUN_PREFIX}{started.strftime(RUN_TIMESTAMP_FORMAT)}"
    output_dir = PROJECT_ROOT / OUTPUT_ROOT / run_name
    adapter_dir = output_dir / "adapter"

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print(f"Base model : {MODEL_ID}")
    print(f"Output     : {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"Precision  : {'bf16' if torch.cuda.is_available() else 'fp32 (CPU)'}")
    print(f"Data       : {DATA_USAGE_FRACTION:.0%} used, ratios {SPLIT_RATIOS}  epochs {NUM_EPOCHS}  seed {RANDOM_SEED}")
    print("=" * 70)

    train_data, eval_data = prepare_datasets()
    tokenizer, model, use_bf16 = load_tokenizer_and_model()

    # LoRA config
    peft_config = LoraConfig(
        r=LORA_RANK,                # width of A and B (bigger = more capacity)
        lora_alpha=LORA_ALPHA,      # volume knob on the adapter's effect
        lora_dropout=LORA_DROPOUT,  # randomly ignore this share of the adapter
        bias="none",                # do not also train the layers' bias terms
        task_type="CAUSAL_LM",      # this is a text-generation (causal LM) task
        target_modules=LORA_TARGET_MODULES,  # which layers get an adapter
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        eval_strategy="steps" if eval_data is not None else "no",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=8,
        prediction_loss_only=True,   # for eval we only need the loss number
        save_strategy="steps" if eval_data is not None else "epoch",
        save_steps=SAVE_STEPS,
        max_length=MAX_SEQUENCE_LENGTH,
        bf16=use_bf16,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
        dataloader_num_workers=0,
        report_to="none",
        assistant_only_loss=True,
    )

    # Trainer + sanity check
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainer.model.print_trainable_parameters()
    if trainable == 0:
        sys.exit(
            f"No trainable parameters for {MODEL_ID}. The LoRA target modules "
            f"{LORA_TARGET_MODULES} did not match any layer. Fix LORA_TARGET_MODULES."
        )

    # Train & Save
    print("Starting training...")
    trainer.train()

    # Save the adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapter to {adapter_dir.relative_to(PROJECT_ROOT)}")

    # Test generating one diagram
    try:
        sanity_generate(trainer.model, tokenizer, eval_data)
    except Exception as error:  # never let a showcase step fail the run
        print(f"(Skipping sample generation: {error})")

    # Write raw results (loss CSVs and base-vs-fine-tuned comparison)
    try:
        write_results(trainer.model, tokenizer, output_dir)
    except Exception as error:
        print(f"(Skipping results generation: {error})")

    print("Done.")


def sanity_generate(model, tokenizer, eval_data: Dataset | None) -> None:
    if eval_data is None or len(eval_data) == 0:
        return

    # Take one held-out example's user turn as the prompt.
    example = eval_data[0]["messages"]
    prompt_messages = [t for t in example if t["role"] != "assistant"]

    # apply_chat_template renders the conversation into the exact text the model was trained on
    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("-" * 70)
    print("Sample generation on a held-out description")
    print("-" * 70)
    print(answer.strip())
    print("-" * 70)

def heldout_rows() -> list[dict]:
    rows = shuffled_rows()
    train_n, val_n, eval_n = split_sizes(len(rows))
    start = train_n + val_n
    return rows[start : start + eval_n]


def _write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(",".join("" if cell is None else str(cell) for cell in row) + "\n")


def _find_latest_state(run_dir: Path) -> dict | None:
    best, best_step = None, -1
    for state_file in run_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(state_file.parent.name.rsplit("-", 1)[-1])
        except ValueError:
            step = -1
        if step > best_step:
            try:
                best = json.loads(state_file.read_text(encoding="utf-8"))
                best_step = step
            except Exception:
                continue
    if best is None:
        root_state = run_dir / "trainer_state.json"
        if root_state.exists():
            best = json.loads(root_state.read_text(encoding="utf-8"))
    return best


def write_loss_csvs(run_dir: Path, results_dir: Path) -> None:
    state = _find_latest_state(run_dir)
    if not state:
        print("(No trainer_state.json found; skipping loss CSVs.)")
        return
    train_rows: list[tuple] = [("step", "loss")]
    eval_rows: list[tuple] = [("step", "eval_loss")]
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if "loss" in entry:
            train_rows.append((step, entry["loss"]))
        if "eval_loss" in entry:
            eval_rows.append((step, entry["eval_loss"]))
    _write_csv(results_dir / "train_loss.csv", train_rows)
    _write_csv(results_dir / "eval_loss.csv", eval_rows)


def _render_prompts(tokenizer, rows: list[dict]) -> list[str]:
    texts = []
    for row in rows:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": build_prompt(row["description"])},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return texts


def generate_jsonl(model, tokenizer, rows, output_path: Path, *, use_adapter: bool) -> None:
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    model.eval()
    set_seed(RANDOM_SEED)

    with output_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), RESULTS_BATCH_SIZE):
            batch = rows[start : start + RESULTS_BATCH_SIZE]
            inputs = tokenizer(
                _render_prompts(tokenizer, batch), return_tensors="pt", padding=True
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # disable_adapter() makes PEFT use the raw frozen weights.
            context = model.disable_adapter() if not use_adapter else nullcontext()
            with context, torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,               # see "sampling settings" in section 1
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row, full_output, prompt_ids in zip(batch, generated, inputs["input_ids"]):
                raw = tokenizer.decode(
                    full_output[prompt_ids.shape[0] :], skip_special_tokens=True
                )
                handle.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "description": row["description"],
                            "reference_mermaid": row.get("mermaid"),
                            "generated": raw.strip(),
                            "raw_generated": raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def write_comparison_csv(results_dir: Path) -> None:
    rows: list[tuple] = [("model", "passed", "total", "pass_rate_percent")]
    for name in ("base", "finetuned"):
        score_path = results_dir / f"{name}.score.json"
        if not score_path.exists():
            continue
        score = json.loads(score_path.read_text(encoding="utf-8"))
        rows.append(
            (name, score.get("passed"), score.get("total"), score.get("pass_rate_percent"))
        )
    _write_csv(results_dir / "comparison.csv", rows)


def run_judge(results_dir: Path) -> None:
    judge = PROJECT_ROOT / "judge.mjs"
    if not judge.exists():
        print(f"(Judge not found at {judge.relative_to(PROJECT_ROOT)}; skipping scoring.)")
        return
    targets = [str(results_dir / "base.jsonl"), str(results_dir / "finetuned.jsonl")]
    try:
        subprocess.run(["node", str(judge), *targets], cwd=str(PROJECT_ROOT), check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"(Skipping judge - is Node installed and `pnpm install` run? {error})")
        return
    write_comparison_csv(results_dir)


def write_results(model, tokenizer, run_dir: Path) -> None:
    results_dir = run_dir / RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Writing results (loss CSVs + base-vs-fine-tuned comparison)...")
    write_loss_csvs(run_dir, results_dir)

    rows = heldout_rows()
    generate_jsonl(model, tokenizer, rows, results_dir / "base.jsonl", use_adapter=False)
    generate_jsonl(model, tokenizer, rows, results_dir / "finetuned.jsonl", use_adapter=True)

    run_judge(results_dir)
    print(f"Wrote results to {results_dir.relative_to(PROJECT_ROOT)}")

# Entry point
if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUN_PREFIX = "run-"
MODEL_ID = "LiquidAI/LFM2.5-350M"
MAX_NEW_TOKENS = 256

# Sampling parameters
TEMPERATURE = 0.1
TOP_K = 50
REPETITION_PENALTY = 1.05

SYSTEM_INSTRUCTION = (
    "You convert process descriptions into Mermaid flowchart syntax. "
    "Reply with the Mermaid diagram only."
)
PROMPT_TEMPLATE = (
    "Convert the following process description into Mermaid flowchart syntax.\n"
    "Reply with the Mermaid diagram only, with no explanation and no code fences.\n"
    "\n"
    "Description:\n"
    "{description}\n"
    "\n"
    "Mermaid diagram:\n"
)

USAGE = 'usage: python test_model.py "process description"'


def find_latest_adapter() -> Path | None:
    if not OUTPUT_ROOT.is_dir():
        return None
    runs = sorted(
        (p for p in OUTPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith(RUN_PREFIX)),
        key=lambda p: p.name,
    )
    for run in reversed(runs):
        adapter = run / "adapter"
        if adapter.is_dir():
            return adapter
    return None


def generate(model, tokenizer, description: str) -> tuple[str, float, int]:
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(description=description.strip()),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.perf_counter() - start

    n_new = out.shape[1] - prompt_len
    answer = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return answer, elapsed, n_new


def report(label: str, answer: str, elapsed: float, n_new: int) -> None:
    rate = n_new / elapsed if elapsed > 0 else 0.0
    print(label)
    print("-" * 42)
    print(answer)
    print(f"# {n_new} tokens in {elapsed:.2f} s ({rate:.1f} tok/s)")
    print()


def main() -> None:
    argv = sys.argv[1:]
    words = [arg for arg in argv if not arg.startswith("--")]
    if not words:
        sys.exit(USAGE)
    description = " ".join(words)

    adapter_dir = find_latest_adapter()
    if adapter_dir is None:
        sys.exit(
            "No training runs found under outputs/.\n"
            "Train first: python train.py"
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.float32
    )
    model = PeftModel.from_pretrained(base, adapter_dir)

    # Base model (no fine-tuning)
    with model.disable_adapter():
        report("Base model (no fine-tuning)",
               *generate(model, tokenizer, description))

    # Fine-tuned model (LoRA adapter)
    report("Fine-tuned model (LoRA adapter)",
           *generate(model, tokenizer, description))


if __name__ == "__main__":
    main()

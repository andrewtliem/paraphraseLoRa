# -*- coding: utf-8 -*-
"""Fine-tune a QLoRA adapter for academic writing-style revision.

This script is a sanitized template based on a Colab workflow.
It trains a LoRA adapter that revises rough or generic academic text into a
target writing style while preserving the original meaning. The goal is not
only to make text different; it is to improve academic phrasing, structure,
precision, and tone without adding unsupported claims.

Replace all dummy paths and style labels before running.
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# =========================
# Configuration
# =========================
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
RANDOM_SEED = 3407

# Replace this with your own JSONL dataset path.
# Example for Colab + Google Drive:
# DATASET_PATH = "/content/drive/MyDrive/my-writing-style/data/writing_style_examples.jsonl"
DATASET_PATH = "data/writing_style_examples.jsonl"

# Replace this with your desired output directory.
# Example for Colab + Google Drive:
# OUTPUT_DIR = "/content/drive/MyDrive/my-writing-style/output/writing-style-qwen-lora"
OUTPUT_DIR = "outputs/writing-style-qwen-lora"

# Generic label used in prompts. Keep this generic for public repositories.
STYLE_OWNER = "the user's"


def load_base_model():
    """Load a 4-bit base model and tokenizer with Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer


def add_lora_adapter(model):
    """Attach a LoRA adapter for QLoRA fine-tuning."""
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )


def prepare_dataset(tokenizer):
    """Load JSONL data, split it, and convert chat messages into text."""
    raw = load_dataset("json", data_files=DATASET_PATH, split="train")

    split_1 = raw.train_test_split(test_size=0.15, seed=RANDOM_SEED)
    train_data = split_1["train"]
    temp_data = split_1["test"]

    split_2 = temp_data.train_test_split(test_size=0.5, seed=RANDOM_SEED)
    valid_data = split_2["train"]
    test_data = split_2["test"]

    Path("data/splits").mkdir(parents=True, exist_ok=True)
    train_path = "data/splits/train.jsonl"
    valid_path = "data/splits/validation.jsonl"
    test_path = "data/splits/test.jsonl"

    train_data.to_json(train_path)
    valid_data.to_json(valid_path)
    test_data.to_json(test_path)

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_path,
            "validation": valid_path,
        },
    )

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    return dataset.map(formatting_prompts_func, batched=True)


def train():
    """Train and save the LoRA adapter."""
    model, tokenizer = load_base_model()
    model = add_lora_adapter(model)
    dataset = prepare_dataset(tokenizer)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        optim="adamw_8bit",
        seed=RANDOM_SEED,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        padding_free=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return model, tokenizer


def clear_gpu_memory():
    """Clear Python and CUDA memory between training and inference."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lora_for_inference(model_dir: str = OUTPUT_DIR):
    """Load the trained LoRA adapter for inference."""
    clear_gpu_memory()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_revision_messages(text: str, mode: str = "sentence"):
    """Build chat messages for sentence or paragraph revision."""
    unit = "paragraph" if mode == "paragraph" else "sentence"
    return [
        {
            "role": "system",
            "content": (
                f"You revise academic {unit}s into {STYLE_OWNER} academic writing style. "
                "Improve wording, sentence structure, academic tone, precision, and flow while preserving the original meaning. "
                "Use precise terminology and formal academic structure. "
                "Do not add new claims, examples, assumptions, citations, or explanations. "
                f"Return only one complete rewritten {unit}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Revise the following {unit} into my academic writing style. "
                "Do not merely make it different; improve academic phrasing while preserving the original meaning.\n\n"
                f"Original {unit}:\n{text}\n\nRevised {unit}:"
            ),
        },
    ]


def generate_revision(
    model,
    tokenizer,
    text: str,
    mode: str = "sentence",
    max_new_tokens: int = 180,
    temperature: float = 0.45,
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
):
    """Generate a style-transfer rewrite."""
    messages = build_revision_messages(text, mode=mode)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
    return generated.strip()


def split_sentences(paragraph: str):
    """Simple sentence splitter for ordinary academic paragraphs."""
    paragraph = paragraph.strip()
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def revise_paragraph_sentence_by_sentence(model, tokenizer, paragraph: str, pause: float = 0.2):
    """Rewrite a paragraph sentence by sentence for better control."""
    revised_sentences = []
    for index, sentence in enumerate(split_sentences(paragraph), start=1):
        revised = generate_revision(
            model,
            tokenizer,
            sentence,
            mode="sentence",
            max_new_tokens=180,
        )
        revised_sentences.append(revised)
        print(f"Sentence {index} done.")
        time.sleep(pause)
    return " ".join(revised_sentences)


def demo_inference(model, tokenizer):
    """Run small dummy examples after training or loading an adapter."""
    sentence = (
        "Dynamic bandwidth allocation changes the amount of bandwidth given to users "
        "according to changing network demand."
    )
    paragraph = (
        "The system gives time slots to devices. When traffic changes, fixed allocation "
        "can reduce efficiency."
    )

    print("SENTENCE TEST:")
    print(generate_revision(model, tokenizer, sentence, mode="sentence"))

    print("\nPARAGRAPH TEST:")
    print(generate_revision(model, tokenizer, paragraph, mode="paragraph", max_new_tokens=350))

    print("\nSENTENCE-BY-SENTENCE PARAGRAPH TEST:")
    print(revise_paragraph_sentence_by_sentence(model, tokenizer, paragraph))


if __name__ == "__main__":
    # Train the adapter. Comment this line if you only want inference from an existing adapter.
    train()

    # Load the trained adapter and run demo inference.
    inference_model, inference_tokenizer = load_lora_for_inference(OUTPUT_DIR)
    demo_inference(inference_model, inference_tokenizer)

# Fine-Tune a QLoRA Revision Model for Your Own Academic Writing Style

This repository accompanies the blog article [Fine-Tuning LoRA for Academic Writing Style](https://blog.atlverse.xyz/blog/fine-tuning-lora-academic-writing-style/).

It shows a small, practical workflow for fine-tuning an instruction model to **revise and reshape academic text into your own writing style**, using your own dataset of writing examples.

The example uses:

- [Unsloth](https://github.com/unslothai/unsloth) for efficient QLoRA fine-tuning
- `Qwen/Qwen2.5-3B-Instruct` as the base model
- Hugging Face `datasets`
- TRL `SFTTrainer`
- A JSONL chat-style dataset

> This is a template. Replace the dummy paths, dataset examples, and style description with your own.

## What This Does

The training script fine-tunes a small LoRA adapter so the model learns a writing-revision task:

```text
Input:  rough, generic, or draft academic sentence/paragraph
Output: the same meaning revised into your academic writing style
```

This is **not only about making the text look different**. The goal is to revise wording, sentence structure, academic tone, precision, and flow while preserving the original meaning. The model should not invent new technical content, citations, results, or claims.

## Complete Workflow Included

The repository includes the full fine-tuning and inference flow in [`fine_tune_qlora_writing_style.py`](./fine_tune_qlora_writing_style.py), not only dataset examples.

| Step | Where it is in the code | What it does |
|---|---|---|
| Load base model | `load_base_model()` | Loads `Qwen/Qwen2.5-3B-Instruct` with Unsloth in 4-bit mode. |
| Add QLoRA adapter | `add_lora_adapter(model)` | Adds LoRA adapters to attention and MLP projection layers. |
| Load dataset | `prepare_dataset(tokenizer)` | Loads `.jsonl` or `.json` chat-style data with Hugging Face `load_dataset("json", ...)`. |
| Split train/validation/test | `prepare_dataset(tokenizer)` | Splits the dataset into train, validation, and test portions. |
| Save split files | `prepare_dataset(tokenizer)` | Writes `data/splits/train.jsonl`, `validation.jsonl`, and `test.jsonl`. |
| Apply chat template | `formatting_prompts_func(...)` | Converts `messages` into model-ready chat text using `tokenizer.apply_chat_template(...)`. |
| Configure training | `SFTConfig(...)` inside `train()` | Sets batch size, accumulation, learning rate, eval steps, save steps, bf16/fp16, and optimizer. |
| Fine-tune model | `trainer.train()` | Runs supervised fine-tuning with TRL `SFTTrainer`. |
| Save LoRA model | `model.save_pretrained(OUTPUT_DIR)` | Saves the trained LoRA adapter. |
| Save tokenizer | `tokenizer.save_pretrained(OUTPUT_DIR)` | Saves the tokenizer with the adapter. |
| Load trained model | `load_lora_for_inference(...)` | Reloads the saved LoRA adapter for inference. |
| Use the model | `generate_revision(...)` | Revises a sentence or paragraph using the trained adapter. |
| Paragraph workflow | `revise_paragraph_sentence_by_sentence(...)` | Revises longer paragraphs sentence by sentence for more control. |
| Demo inference | `demo_inference(...)` | Runs example sentence, paragraph, and sentence-by-sentence tests. |

The main script intentionally keeps the workflow in one file so it can be copied into Colab and run top-to-bottom.

## Dataset Format

Prepare either a `.jsonl` file, where each line contains one chat-style training example with `messages`, or a `.json` file containing an array of the same objects. The training script can read either format through Hugging Face `load_dataset("json", ...)`.

Each example should contain:

1. A `system` message that defines the revision task.
2. A `user` message containing the rough, generic, or draft text.
3. An `assistant` message containing your desired revised version.

The repository includes the same small dummy dataset in two formats:

```text
examples/writing_style_examples.jsonl  # line-delimited JSON, useful for training
examples/writing_style_examples.json   # regular JSON array, easier to upload to tools that reject .jsonl
```

A single training row looks like this:

```jsonl
{"messages": [{"role": "system", "content": "You revise generic academic text into the user's academic writing style while preserving meaning."}, {"role": "user", "content": "Revise this sentence in my academic writing style:\n\nThe system gives bandwidth to users when traffic changes."}, {"role": "assistant", "content": "The system dynamically allocates bandwidth in response to variations in user traffic demand."}]}
```

A paragraph-level example looks like this:

```jsonl
{"messages": [{"role": "system", "content": "You revise generic academic text into the user's academic writing style while preserving meaning."}, {"role": "user", "content": "Revise this paragraph in my academic writing style:\n\nDynamic bandwidth allocation is a technique that changes how much bandwidth each device receives. It is useful because network traffic is not always the same. When more users send data, the system can change the allocation."}, {"role": "assistant", "content": "Dynamic bandwidth allocation adjusts the bandwidth assigned to each device according to variations in network traffic demand. This mechanism is useful because traffic conditions are not static. As more users transmit data, the system can modify the allocation to better match the current network state."}]}
```

For your real training run, copy one example file and replace the dummy rows with your own paired examples:

```bash
mkdir -p data
cp examples/writing_style_examples.jsonl data/writing_style_examples.jsonl
# or, if your upload/tooling system does not accept .jsonl:
cp examples/writing_style_examples.json data/writing_style_examples.json
```

Then edit one of these files:

```text
data/writing_style_examples.jsonl
data/writing_style_examples.json
```

If a platform says `.jsonl` is unsupported, use the `.json` version. It contains the same records as a normal JSON array.

### Dataset Writing Tips

Good training examples are not only “different wording.” They should show the model what *good revision* means in your style:

| Input should contain | Assistant output should show |
|---|---|
| rough academic wording | clearer academic structure |
| generic explanation | more precise terminology |
| weak sentence flow | stronger sentence organization |
| same technical meaning | no new unsupported claims |
| sentence and paragraph examples | consistent tone across lengths |

Recommended starting size:

- **Tiny test:** 20–50 examples, just to verify the pipeline works.
- **Useful first adapter:** 200–500 examples.
- **Better style consistency:** 1,000+ carefully cleaned examples.

Keep private names, unpublished results, confidential institutional data, and sensitive source text out of the public dataset.

## Colab Setup

If you are running this in Google Colab, start with:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then install dependencies:

```bash
pip install unsloth
pip install --upgrade transformers datasets trl accelerate bitsandbytes
```

## Training

The training flow is implemented in the `train()` function:

```python
def train():
    model, tokenizer = load_base_model()
    model = add_lora_adapter(model)
    dataset = prepare_dataset(tokenizer)

    training_args = SFTConfig(...)

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
```

Dataset splitting happens before training:

```python
split_1 = raw.train_test_split(test_size=0.15, seed=RANDOM_SEED)
train_data = split_1["train"]
temp_data = split_1["test"]

split_2 = temp_data.train_test_split(test_size=0.5, seed=RANDOM_SEED)
valid_data = split_2["train"]
test_data = split_2["test"]
```

This gives approximately:

```text
85.0% train
 7.5% validation
 7.5% test
```

Edit these variables in `fine_tune_qlora_writing_style.py`:

```python
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "/content/drive/MyDrive/path/to/writing_style_examples.jsonl"
# For a quick local smoke test, you can also use either:
# DATASET_PATH = "examples/writing_style_examples.jsonl"
# DATASET_PATH = "examples/writing_style_examples.json"
OUTPUT_DIR = "/content/drive/MyDrive/path/to/writing-style-qwen-lora"
STYLE_OWNER = "the user's"
```

Then run:

```bash
python fine_tune_qlora_writing_style.py
```

## Inference

After training, the script reloads the saved LoRA adapter and uses it for revision.

Model loading happens here:

```python
inference_model, inference_tokenizer = load_lora_for_inference(OUTPUT_DIR)
```

The loader uses:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)
```

Text revision happens through:

```python
generate_revision(model, tokenizer, text, mode="sentence")
generate_revision(model, tokenizer, text, mode="paragraph")
```

The inference prompts are written as **revision prompts**, not as generic paraphrase prompts.

Example inference goal:

```text
Original:
Dynamic bandwidth allocation changes the amount of bandwidth given to users based on demand.

Revised:
Dynamic bandwidth allocation adjusts the assigned bandwidth in response to variations in user demand.
```

## Safety and Quality Notes

When fine-tuning a personal writing-style model:

1. **Use only writing you have permission to use.**
2. **Remove private, unpublished, or sensitive information from the dataset.**
3. **Do not train on confidential student, patient, client, or institutional data.**
4. **Keep meaning preservation as the main objective.**
5. **Treat the output as a revision draft, not an automatically correct final text.**
6. **Review outputs manually before using them in academic writing.**
7. **Do not use the model to fabricate citations, results, or technical claims.**

## Repository Contents

```text
fine_tune_qlora_writing_style.py  # training + inference template
sample_data.jsonl                 # tiny dummy dataset example
examples/writing_style_examples.jsonl # larger dummy dataset example, JSONL format
examples/writing_style_examples.json  # same dummy dataset, regular JSON format
requirements.txt                  # Python packages
README.md                         # this guide
```

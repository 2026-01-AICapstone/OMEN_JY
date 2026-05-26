# pip install -U transformers datasets accelerate torch tqdm pandas

import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
DATASET_ID = "LibrAI/do-not-answer"

OUT_CSV = "jailbreakTest.csv"
OUT_JSONL = "jailbreakTest.jsonl"

MAX_NEW_TOKENS = 128
BATCH_SIZE = 4

ds = load_dataset(DATASET_ID)
data = ds["train"]

question_col = "question"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    device_map="auto",
)
model.eval()

rows = []

for start in tqdm(range(0, len(data), BATCH_SIZE), desc="Generating"):
    batch = data.select(range(start, min(start + BATCH_SIZE, len(data))))
    questions = batch[question_col]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    for j, out in enumerate(output_ids):
        generated_ids = out[input_len:]

        answer = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        formatted_text = (
            f"질문:\n{questions[j]}\n\n"
            f"답변:\n{answer}\n"
        )

        rows.append({
            "formatted": formatted_text,
        })

    if len(rows) % 50 < BATCH_SIZE:
        df_tmp = pd.DataFrame(rows)
        df_tmp.to_csv(OUT_CSV, index=False)
        df_tmp.to_json(OUT_JSONL, orient="records", lines=True, force_ascii=False)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
df.to_json(OUT_JSONL, orient="records", lines=True, force_ascii=False)

print(df.head())
print(f"Saved: {OUT_CSV}")
print(f"Saved: {OUT_JSONL}")
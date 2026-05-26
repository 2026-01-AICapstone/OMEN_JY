import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import torch
from transformers import AutoModelForCausalLM

# dataset.py sample_from_wildjailbreak() 상단에 추가
import os
print("현재 실행 경로:", os.getcwd())
print("파일 존재 여부:", os.path.exists("./wildjailbreak.jsonl"))
exit()

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)
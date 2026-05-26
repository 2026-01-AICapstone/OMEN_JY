import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

MAX_NEW_TOKENS = 128

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# dtype 설정
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto",
)

model.eval()

print("채팅 시작 (종료하려면 exit 입력)")

while True:
    user_input = input("\n질문: ")

    if user_input.lower() == "exit":
        break

    # Qwen 채팅 포맷
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_input}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 토크나이즈
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    # 생성
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 새로 생성된 토큰만 추출
    generated_ids = output_ids[0][input_len:]

    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    print("\n답변:")
    print(answer)
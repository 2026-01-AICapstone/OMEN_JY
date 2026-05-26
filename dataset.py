# import random
# from typing import Dict
# import jsonlines
# import torch
# import transformers
# from datasets import load_dataset
# from torch.utils.data import Dataset
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
#
# random.seed(0)
#
#
# class RepBendingDataset(Dataset):
#
#     def __init__(self,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  num_examples,
#                  mode,
#                  max_length,
#                  model_name_or_path,
#                  dataset_path,
#                  split=None,
#                  is_online=False,
#                  ):
#         super().__init__()
#
#         self.model_name_or_path = model_name_or_path.lower()
#         self.max_length = max_length
#         self.tokenizer = tokenizer
#         self.num_examples = num_examples
#
#         self.data_safe_samples = []
#         self.data_unsafe_samples = []
#         self.data_unsafe_labels = []
#         self.retain_set = []
#         self.unsafe_prompt_pair_unsafe_answer = []
#         self.unsafe_prompt_pair_safe_answer = []
#
#         # =========================================================
#         # TEMPLATE
#         # =========================================================
#         def set_tags_templates():
#             one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
#
#             user_tag, assistant_tag = None, None
#
#             if "qwen" in self.model_name_or_path:
#                 print("USING QWEN TEMPLATE")
#                 user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
#                 assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
#             elif "llama" in self.model_name_or_path:
#                 print("USING LLAMA TEMPLATE")
#                 user_tag = "<|user|>\n"
#                 assistant_tag = "<|assistant|>\n"
#             else:
#                 raise NotImplementedError()
#
#             self.user_tag = user_tag
#             self.assistant_tag = assistant_tag
#             return one_shot_template
#
#         # =========================================================
#         # ULTRACHAT (SAFE FIXED VERSION)
#         # =========================================================
#
#         # def sample_from_ultrachat():
#         #     print("Loading UltraChat (non-streaming mode)...")
#         #
#         #     ds = load_dataset(
#         #         "HuggingFaceH4/ultrachat_200k",
#         #         split=f"train_sft[:{self.num_examples * 3}]"  # 여유분 확보
#         #     )
#         #
#         #     safe_samples = []
#         #     retain = []
#         #
#         #     for example in ds:
#         #         if len(safe_samples) >= self.num_examples:
#         #             break
#         #
#         #         messages = example["messages"]
#         #         if len(messages) < 2:
#         #             continue
#         #
#         #         instruction = messages[0]["content"]
#         #         response = messages[1]["content"]
#         #
#         #         safe_samples.append(
#         #             self.one_shot_template.format(
#         #                 user_tag=self.user_tag,
#         #                 assistant_tag=self.assistant_tag,
#         #                 instruction=instruction,
#         #                 response=response
#         #             )
#         #         )
#         #         retain.append(instruction)
#         #
#         #     self.data_safe_samples = safe_samples
#         #     self.retain_set = retain
#         #     print(f"UltraChat loaded: {len(safe_samples)}")
#
#         def sample_from_ultrachat():
#             print("Loading safe samples via HuggingFace API...")
#             import requests
#
#             url = "https://datasets-server.huggingface.co/rows"
#             safe_samples = []
#             retain = []
#             batch_size = 100  # API 최대 허용치
#
#             offset = 0
#             while len(safe_samples) < self.num_examples:
#                 params = {
#                     "dataset": "HuggingFaceH4/ultrachat_200k",
#                     "config": "default",
#                     "split": "train_sft",
#                     "offset": offset,
#                     "length": batch_size
#                 }
#
#                 response = requests.get(url, params=params, timeout=60)
#                 data = response.json()
#                 rows = data.get("rows", [])
#
#                 if not rows:
#                     break
#
#                 for row in rows:
#                     messages = row.get("row", {}).get("messages", [])
#                     if len(messages) < 2:
#                         continue
#
#                     instruction = messages[0].get("content", "")
#                     response_text = messages[1].get("content", "")
#
#                     if not instruction or not response_text:
#                         continue
#
#                     safe_samples.append(
#                         self.one_shot_template.format(
#                             user_tag=self.user_tag,
#                             assistant_tag=self.assistant_tag,
#                             instruction=instruction,
#                             response=response_text
#                         )
#                     )
#                     retain.append(instruction)
#
#                     if len(safe_samples) >= self.num_examples:
#                         break
#
#                 offset += batch_size
#                 print(f"  {len(safe_samples)}/{self.num_examples} 로드 중...")
#
#             self.data_safe_samples = safe_samples
#             self.retain_set = retain
#             print(f"Safe samples loaded: {len(safe_samples)}")
#
#
#
#         # =========================================================
#         # DO NOT ANSWER (SAFE LIMITED)
#         # =========================================================
#         def sample_from_do_not_answer_with_kmeans():
#             print("Loading Do-Not-Answer via HuggingFace API...")
#             import requests
#             from sklearn.cluster import KMeans
#             from sklearn.feature_extraction.text import TfidfVectorizer
#
#             url = "https://datasets-server.huggingface.co/rows"
#             raw_unsafe = []
#             raw_inst = []
#             seed_dict = {}
#
#             offset = 0
#             batch_size = 100
#
#             while len(raw_unsafe) < 500:
#                 params = {
#                     "dataset": "LibrAI/do-not-answer",
#                     "config": "default",
#                     "split": "train",
#                     "offset": offset,
#                     "length": batch_size
#                 }
#
#                 response = requests.get(url, params=params, timeout=60)
#                 data = response.json()
#                 rows = data.get("rows", [])
#
#                 if not rows:
#                     break
#
#                 for row in rows:
#                     r = row.get("row", {})
#                     instruction = r.get("question", r.get("prompt", ""))
#                     response_text = r.get("response", "")
#                     category = r.get("risk_area", "Unknown")
#
#                     if not instruction:
#                         continue
#
#                     formatted = self.one_shot_template.format(
#                         user_tag=self.user_tag,
#                         assistant_tag=self.assistant_tag,
#                         instruction=instruction,
#                         response=response_text
#                     )
#
#                     raw_unsafe.append(formatted)
#                     raw_inst.append(instruction)
#
#                     if category not in seed_dict:
#                         seed_dict[category] = instruction
#
#                     if len(raw_unsafe) >= 500:
#                         break
#
#                 offset += batch_size
#                 print(f"  Unsafe {len(raw_unsafe)}/500 로드 중...")
#
#             # KMeans 클러스터링
#             vectorizer = TfidfVectorizer(max_features=1000)
#             X = vectorizer.fit_transform(raw_inst)
#
#             pseudo_labels = KMeans(
#                 n_clusters=5,
#                 random_state=42,
#                 n_init=1
#             ).fit_predict(X)
#
#             self.data_unsafe_samples = raw_unsafe
#             self.data_unsafe_labels = pseudo_labels.tolist()
#
#             print(f"Unsafe loaded: {len(raw_unsafe)}")
#
#         # =========================================================
#         # WILDJAILBREAK (SAFE LIMITED)
#         # =========================================================
#         def sample_from_wildjailbreak():
#             print("Loading WildJailbreak (safe 500)...")
#
#             train_data = []
#
#             try:
#                 with jsonlines.open("./wildjailbreak.jsonl") as f:
#                     for i, line in enumerate(f):
#                         train_data.append(line)
#                         if i >= 500:
#                             break
#             except:
#                 print("wildjailbreak.jsonl not found")
#                 return
#
#             unsafe_pair = []
#             safe_pair = []
#
#             template = self.one_shot_template
#
#             for d in train_data:
#
#                 prompt = d.get("prompt", "")
#                 harmful = d.get("harmful_answer", "")
#                 harmless = d.get("harmless_answer", "")
#                 dtype = d.get("prompt_type", "")
#
#                 if "harmful" in dtype.lower():
#                     if harmful and harmless:
#                         unsafe_pair.append(
#                             template.format(
#                                 user_tag=self.user_tag,
#                                 assistant_tag=self.assistant_tag,
#                                 instruction=prompt,
#                                 response=harmful
#                             )
#                         )
#
#                         safe_pair.append(
#                             template.format(
#                                 user_tag=self.user_tag,
#                                 assistant_tag=self.assistant_tag,
#                                 instruction=prompt,
#                                 response=harmless
#                             )
#                         )
#
#                 if len(unsafe_pair) >= 500:
#                     break
#
#             self.unsafe_prompt_pair_unsafe_answer = unsafe_pair
#             self.unsafe_prompt_pair_safe_answer = safe_pair
#
#             print(f"WildJailbreak loaded: {len(unsafe_pair)}")
#
#         # =========================================================
#         # RUN PIPELINE
#         # =========================================================
#         self.one_shot_template = set_tags_templates()
#
#         sample_from_ultrachat()
#         sample_from_do_not_answer_with_kmeans()
#         sample_from_wildjailbreak()
#
#         self.tokenizer.padding_side = "right"
#
#     # =========================================================
#     def __len__(self):
#         return min(
#             len(self.data_safe_samples),
#             len(self.data_unsafe_samples),
#             len(self.retain_set),
#             len(self.unsafe_prompt_pair_unsafe_answer)
#         )
#
#     # =========================================================
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         safe = self.data_safe_samples[i]
#         unsafe = self.data_unsafe_samples[i]
#         label = self.data_unsafe_labels[i]
#         retain = self.retain_set[i]
#         unsafe_pair = self.unsafe_prompt_pair_unsafe_answer[i]
#         safe_pair = self.unsafe_prompt_pair_safe_answer[i]
#
#         def tok(x, max_len):
#             return self.tokenizer(
#                 x,
#                 max_length=max_len,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt"
#             )
#
#         half = self.max_length // 2
#
#         # <SEPARATOR> 기준으로 request/response 분리
#         safe_req, safe_res = safe.split("<SEPARATOR>")
#         unsafe_req, unsafe_res = unsafe.split("<SEPARATOR>")
#         ur_req, ur_res = unsafe_pair.split("<SEPARATOR>")  # unsafe request + unsafe response
#         sr_req, sr_res = safe_pair.split("<SEPARATOR>")  # unsafe request + safe response
#
#         # 토크나이징
#         safe_in = tok(safe_req, half)
#         safe_out = tok(safe_res, half)
#         unsafe_in = tok(unsafe_req, half)
#         unsafe_out = tok(unsafe_res, half)
#         retain_tok = tok(retain, self.max_length)
#         ur_in = tok(ur_req, half)
#         ur_out = tok(ur_res, half)
#         sr_out = tok(sr_res, half)
#
#         return {
#             # safe sample (safe request + safe response)
#             "ids_safe_sample": torch.cat([safe_in["input_ids"], safe_out["input_ids"]], dim=1),
#             "mask_safe_sample": torch.cat([safe_in["attention_mask"], safe_out["attention_mask"]], dim=1),
#             "mask_safe_sample_request": safe_in["attention_mask"],
#             "mask_safe_sample_response": safe_out["attention_mask"],
#
#             # unsafe sample (unsafe request + unsafe response)
#             "ids_unsafe_sample": torch.cat([unsafe_in["input_ids"], unsafe_out["input_ids"]], dim=1),
#             "mask_unsafe_sample": torch.cat([unsafe_in["attention_mask"], unsafe_out["attention_mask"]], dim=1),
#             "mask_unsafe_sample_request": unsafe_in["attention_mask"],
#             "mask_unsafe_sample_response": unsafe_out["attention_mask"],
#
#             # retain set (일반 지식 보존용)
#             "ids_retain": retain_tok["input_ids"],
#             "mask_retain": retain_tok["attention_mask"],
#
#             # unsafe request + unsafe response 쌍
#             "ids_unsafe_request_unsafe_response": torch.cat([ur_in["input_ids"], ur_out["input_ids"]], dim=1),
#             "mask_unsafe_request_unsafe_response": torch.cat([ur_in["attention_mask"], ur_out["attention_mask"]],
#                                                              dim=1),
#             "mask_unsafe_response_for_unsafe_request": ur_out["attention_mask"],
#
#             # unsafe request + safe response 쌍
#             "ids_unsafe_request_safe_response": torch.cat([ur_in["input_ids"], sr_out["input_ids"]], dim=1),
#             "mask_unsafe_request_safe_response": torch.cat([ur_in["attention_mask"], sr_out["attention_mask"]], dim=1),
#             "mask_safe_response_for_unsafe_request": sr_out["attention_mask"],
#
#             # unsafe request 단독
#             "ids_unsafe_request": ur_in["input_ids"],
#             "mask_unsafe_request": ur_in["attention_mask"],
#
#             # CRUSH 클러스터 라벨
#             "labels": torch.tensor(label, dtype=torch.long).unsqueeze(0),
#         }

"""
dataset.py
==========
RepBendingDataset (CRUSH 변경 후 버전)

핵심 설계 변경 (이전 버전 대비):
  - K-Means + TF-IDF + seed 1개 클러스터링 로직 제거
  - DNA의 risk_area 정답 라벨은 검증용으로만 보존 (data_unsafe_true_labels)
  - 학습용 pseudo-label은 외부(clustering.py)에서 주입 (set_pseudo_labels())
  - WJ 데이터에는 라벨 부여하지 않음 (pull/push 학습에서 제외)
  - DNA의 raw request prompt 리스트(data_unsafe_request_prompts)를 보존 →
    clustering.py가 base 모델로 hidden 추출 시 사용

데이터 흐름:
  1. RepBendingDataset 생성 시: 전체 데이터 로드, true_labels만 보존, pseudo_labels는 미설정
  2. train.py에서 base 모델로 clustering.py 실행 → pseudo_labels + centroids 생성
  3. dataset.set_pseudo_labels(pseudo_labels) 호출 → __getitem__이 pseudo-label 반환
  4. 학습 시작
"""

import random
import os
from typing import Dict, List, Optional

import jsonlines
import torch
import transformers
from torch.utils.data import Dataset
import numpy as np
import requests
import time

random.seed(0)


class RepBendingDataset(Dataset):

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_examples,
                 mode,
                 max_length,
                 model_name_or_path,
                 dataset_path=None,
                 split=None,
                 is_online=False,
                 ):
        super().__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.num_examples = num_examples

        # ---------- 데이터 컨테이너 ----------
        self.data_safe_samples = []                  # harmless prompt + harmless answer
        self.data_unsafe_samples = []                # harmful prompt + harmful answer (DNA)
        self.data_unsafe_request_prompts = []        # DNA request만 (clustering.py용)
        self.data_unsafe_true_labels = []            # DNA risk_area 정답 (검증용)
        self.data_unsafe_labels = None               # pseudo-label (외부 주입)

        self.retain_set = []
        self.unsafe_prompt_pair_unsafe_answer = []   # WJ harmful + harmful
        self.unsafe_prompt_pair_safe_answer = []     # WJ harmful + harmless

        # =========================================================
        # TEMPLATE
        # =========================================================
        def set_tags_templates():
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
            user_tag, assistant_tag = None, None
            if "qwen" in self.model_name_or_path:
                print("USING QWEN TEMPLATE")
                user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
            elif "llama" in self.model_name_or_path:
                print("USING LLAMA TEMPLATE")
                user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
                assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # 추가
            elif "gemma" in self.model_name_or_path.lower():
                print("USING GEMMA TEMPLATE")

                user_tag = "<start_of_turn>user\n"
                assistant_tag = "<end_of_turn>\n<start_of_turn>model\n"
                
                
            else:
                raise NotImplementedError(f"Model {self.model_name_or_path} not supported")
            self.user_tag = user_tag
            self.assistant_tag = assistant_tag
            return one_shot_template

        # =========================================================
        # API 헬퍼
        # =========================================================
        def fetch_rows(url, params, max_retries=3):
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, params=params, timeout=120)
                    return response.json().get("rows", [])
                except requests.exceptions.ReadTimeout:
                    print(f"  타임아웃 발생, {attempt + 1}번째 재시도...")
                    time.sleep(5)
            return []

        # =========================================================
        # ULTRACHAT
        # =========================================================
        def sample_from_ultrachat():
            print("Loading UltraChat via HuggingFace API (939개)...")
            url = "https://datasets-server.huggingface.co/rows"
            safe_samples = []
            retain = []
            offset = 0

            while len(safe_samples) < 939:
                params = {
                    "dataset": "HuggingFaceH4/ultrachat_200k",
                    "config": "default",
                    "split": "train_sft",
                    "offset": offset,
                    "length": 100
                }
                rows = fetch_rows(url, params)
                if not rows:
                    break

                for row in rows:
                    messages = row.get("row", {}).get("messages", [])
                    if len(messages) < 2:
                        continue
                    instruction = messages[0].get("content", "")
                    response_text = messages[1].get("content", "")
                    if not instruction or not response_text:
                        continue

                    safe_samples.append(
                        self.one_shot_template.format(
                            user_tag=self.user_tag,
                            assistant_tag=self.assistant_tag,
                            instruction=instruction,
                            response=response_text
                        )
                    )
                    retain.append(instruction)

                    if len(safe_samples) > 939:
                        break

                offset += 100
                print(f"  UltraChat {len(safe_samples)}/939 로드 중...")

            self.data_safe_samples.extend(safe_samples)
            self.retain_set.extend(retain)
            print(f"UltraChat loaded: {len(safe_samples)}")

        # =========================================================
        # WILDJAILBREAK (라벨 미부여)
        # =========================================================
        def sample_from_wildjailbreak():
            print("Loading WildJailbreak...")
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            jsonl_path = os.path.join(BASE_DIR, "wildjailbreak.jsonl")

            if not os.path.exists(jsonl_path):
                print("wildjailbreak.jsonl not found, skipping...")
                return

            train_data = []
            import json
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        train_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            wj_safe_samples = []
            wj_unsafe_pair = []
            wj_safe_pair = []
            template = self.one_shot_template

            for d in train_data:
                prompt = d.get("prompt", "")
                harmful = d.get("harmful_answer", "")
                harmless = d.get("harmless_answer", "")
                dtype = d.get("prompt_type", "")
                if not prompt:
                    continue

                if "harmful" in dtype.lower():
                    if harmful and harmless and len(wj_unsafe_pair) < 939:
                        wj_unsafe_pair.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmful
                            )
                        )
                        wj_safe_pair.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmless
                            )
                        )

                elif "harmless" in dtype.lower():
                    if harmless and len(wj_safe_samples) < 939:
                        wj_safe_samples.append(
                            template.format(
                                user_tag=self.user_tag,
                                assistant_tag=self.assistant_tag,
                                instruction=prompt,
                                response=harmless
                            )
                        )
                        self.retain_set.append(prompt)

                if len(wj_unsafe_pair) >= 939 and len(wj_safe_samples) >= 939:
                    break

            self.data_safe_samples.extend(wj_safe_samples)
            self.unsafe_prompt_pair_unsafe_answer = wj_unsafe_pair
            self.unsafe_prompt_pair_safe_answer = wj_safe_pair

            print(f"WildJailbreak safe loaded: {len(wj_safe_samples)}")
            print(f"WildJailbreak unsafe pair loaded: {len(wj_unsafe_pair)}")

        # =========================================================
        # DO-NOT-ANSWER (정답 라벨 보존, K-Means 제거)
        # =========================================================
        def sample_from_do_not_answer():
            print("Loading Do-Not-Answer via HuggingFace API...")

            target_categories = {
                "Malicious Uses": 0,
                "Human-Chatbot Interaction Harms": 1,
                "Information Hazards": 2,
                "Misinformation Harms": 3,
                "Discrimination, Exclusion, Toxicity, Hateful, Offensive": 4
            }

            url = "https://datasets-server.huggingface.co/rows"
            offset = 0

            while len(self.data_unsafe_samples) < 939:
                params = {
                    "dataset": "LibrAI/do-not-answer",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": 100
                }
                rows = fetch_rows(url, params)
                if not rows:
                    break

                for row in rows:
                    r = row.get("row", {})
                    instruction = r.get("question", "")
                    category = r.get("risk_area", "Unknown")
                    if not instruction or category not in target_categories:
                        continue

                    true_label = target_categories[category]

                    best_response = ""
                    for model_name in ["GPT4", "ChatGPT", "Claude", "ChatGLM2",
                                       "llama2-7b-chat", "vicuna-7b"]:
                        resp = r.get(f"{model_name}_response", "")
                        if resp and len(resp) > len(best_response):
                            best_response = resp
                    if not best_response:
                        continue

                    formatted = self.one_shot_template.format(
                        user_tag=self.user_tag,
                        assistant_tag=self.assistant_tag,
                        instruction=instruction,
                        response=best_response
                    )

                    self.data_unsafe_samples.append(formatted)
                    self.data_unsafe_request_prompts.append(instruction)
                    self.data_unsafe_true_labels.append(true_label)

                    if len(self.data_unsafe_samples) >= 939:
                        break

                offset += 100
                print(f"  Do-Not-Answer {len(self.data_unsafe_samples)}/939 로드 중...")

            print(f"Do-Not-Answer loaded: {len(self.data_unsafe_samples)}")
            print(f"  정답 라벨(risk_area) 분포: {np.bincount(self.data_unsafe_true_labels, minlength=5).tolist()}")
            print(f"  ⚠️ 학습용 pseudo-label은 아직 미설정. set_pseudo_labels()로 주입 필요.")

        # =========================================================
        # RUN PIPELINE
        # =========================================================
        self.one_shot_template = set_tags_templates()
        sample_from_ultrachat()
        sample_from_wildjailbreak()
        sample_from_do_not_answer()
        self.tokenizer.padding_side = "right"

        print(f"\n✅ 데이터셋 구성 완료:")
        print(f"  Safe samples (UltraChat+WJ harmless): {len(self.data_safe_samples)}")
        print(f"  Unsafe samples (Do-Not-Answer):       {len(self.data_unsafe_samples)}")
        print(f"  Unsafe pair (WJ harmful):             {len(self.unsafe_prompt_pair_unsafe_answer)}")
        print(f"  Retain set:                           {len(self.retain_set)}")

    # =========================================================
    def set_pseudo_labels(self, pseudo_labels: List[int]):
        """clustering.py 결과를 학습용 라벨로 주입."""
        if len(pseudo_labels) != len(self.data_unsafe_samples):
            raise ValueError(
                f"pseudo_labels 길이({len(pseudo_labels)}) ≠ "
                f"data_unsafe_samples 길이({len(self.data_unsafe_samples)})"
            )
        self.data_unsafe_labels = list(pseudo_labels)
        bincount = np.bincount(self.data_unsafe_labels,
                               minlength=max(self.data_unsafe_labels) + 1)
        print(f"✅ Pseudo-label 주입 완료. 분포: {bincount.tolist()}")

    # =========================================================
    def __len__(self):
        return min(
            len(self.data_safe_samples),
            len(self.data_unsafe_samples),
            len(self.retain_set),
            len(self.unsafe_prompt_pair_unsafe_answer)
        )

    # =========================================================
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # DNA 균등 샘플링 (pseudo-label 기반)
        if self.data_unsafe_labels is not None:
            num_clusters = max(self.data_unsafe_labels) + 1
            target_label = i % num_clusters
            candidates = [j for j, l in enumerate(self.data_unsafe_labels) if l == target_label]
            if not candidates:
                unsafe_idx = i % len(self.data_unsafe_samples)
            else:
                unsafe_idx = candidates[(i // num_clusters) % len(candidates)]
            label = self.data_unsafe_labels[unsafe_idx]
        else:
            unsafe_idx = i % len(self.data_unsafe_samples)
            label = -1  # pseudo-label 미주입 시 더미

        unsafe = self.data_unsafe_samples[unsafe_idx]

        # WJ는 라벨 없이 단순 순환
        pair_idx = i % len(self.unsafe_prompt_pair_unsafe_answer)
        unsafe_pair = self.unsafe_prompt_pair_unsafe_answer[pair_idx]
        safe_pair = self.unsafe_prompt_pair_safe_answer[pair_idx]

        safe = self.data_safe_samples[i % len(self.data_safe_samples)]
        retain = self.retain_set[i % len(self.retain_set)]

        def tok(x, max_len):
            return self.tokenizer(
                x, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

        half = self.max_length // 2

        safe_req, safe_res = safe.split("<SEPARATOR>")
        unsafe_req, unsafe_res = unsafe.split("<SEPARATOR>")
        ur_req, ur_res = unsafe_pair.split("<SEPARATOR>")
        sr_req, sr_res = safe_pair.split("<SEPARATOR>")

        safe_in = tok(safe_req, half)
        safe_out = tok(safe_res, half)
        unsafe_in = tok(unsafe_req, half)
        unsafe_out = tok(unsafe_res, half)
        retain_tok = tok(retain, self.max_length)
        ur_in = tok(ur_req, half)
        ur_out = tok(ur_res, half)
        sr_out = tok(sr_res, half)

        return {
            "ids_safe_sample":              torch.cat([safe_in["input_ids"],  safe_out["input_ids"]],  dim=1),
            "mask_safe_sample":             torch.cat([safe_in["attention_mask"], safe_out["attention_mask"]], dim=1),
            "mask_safe_sample_request":     safe_in["attention_mask"],
            "mask_safe_sample_response":    safe_out["attention_mask"],

            "ids_unsafe_sample":            torch.cat([unsafe_in["input_ids"],  unsafe_out["input_ids"]],  dim=1),
            "mask_unsafe_sample":           torch.cat([unsafe_in["attention_mask"], unsafe_out["attention_mask"]], dim=1),
            "mask_unsafe_sample_request":   unsafe_in["attention_mask"],
            "mask_unsafe_sample_response":  unsafe_out["attention_mask"],

            "ids_retain":                   retain_tok["input_ids"],
            "mask_retain":                  retain_tok["attention_mask"],

            "ids_unsafe_request_unsafe_response":       torch.cat([ur_in["input_ids"],  ur_out["input_ids"]],  dim=1),
            "mask_unsafe_request_unsafe_response":      torch.cat([ur_in["attention_mask"], ur_out["attention_mask"]], dim=1),
            "mask_unsafe_response_for_unsafe_request":  ur_out["attention_mask"],

            "ids_unsafe_request_safe_response":         torch.cat([ur_in["input_ids"],  sr_out["input_ids"]],  dim=1),
            "mask_unsafe_request_safe_response":        torch.cat([ur_in["attention_mask"], sr_out["attention_mask"]], dim=1),
            "mask_safe_response_for_unsafe_request":    sr_out["attention_mask"],

            "ids_unsafe_request":           ur_in["input_ids"],
            "mask_unsafe_request":          ur_in["attention_mask"],

            # DNA의 pseudo-label만 부여. WJ는 라벨 없음.
            "labels":                       torch.tensor(label, dtype=torch.long).unsqueeze(0),
        }

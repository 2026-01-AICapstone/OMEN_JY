# # 7개 클러스터 시
# # import faulthandler
# # faulthandler.enable()
# # import os
# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# # import random
# # from dataclasses import dataclass
# #
# # import numpy as np
# # import torch
# # import transformers
# #
# # from args import (HyperparamArguments, LoraArguments, ModelArguments,
# #                   TrainingArguments)
# # from classifier import HarmbenchClassifier
# # from dataset import RepBendingDataset
# # from trainer import CustomTrainer
# # from peft import LoraConfig, get_peft_model
# # from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# #
# # from crush_loss import FixedCentroidHolder
# # from clustering import cluster_unsafe_prompts
# #
# #
# # def data_collator(batch_list):
# #     batch_inputs = {}
# #     for features in batch_list:
# #         for k, input in features.items():
# #             batch_inputs.setdefault(k, []).append(input)
# #
# #     for k, inputs in batch_inputs.items():
# #         if isinstance(inputs[0], torch.Tensor):
# #             batch_inputs[k] = torch.cat(inputs, dim=0)
# #         elif isinstance(inputs[0], int):
# #             batch_inputs[k] = torch.tensor(inputs)
# #         else:
# #             batch_inputs[k] = inputs
# #     return batch_inputs
# #
# #
# # def train():
# #     parser = transformers.HfArgumentParser(
# #         (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
# #     )
# #     (model_args, training_args, lora_args, hyperparam_args) = parser.parse_args_into_dataclasses()
# #
# #     print(hyperparam_args.to_dict())
# #     print(lora_args)
# #     print(model_args)
# #     print(training_args)
# #
# #     device_map = "auto"
# #     model_name_or_path = model_args.model_name_or_path
# #
# #     # ---------- 모델 config / target layer 자동 감지 ----------
# #     config = AutoConfig.from_pretrained(model_name_or_path)
# #     total_layers = config.num_hidden_layers
# #
# #     target_layers = hyperparam_args.target_layers
# #     target_layer_start_idx = hyperparam_args.target_layer_start_idx
# #     layers_window_size = hyperparam_args.layers_window_size
# #     transform_layers = hyperparam_args.transform_layers
# #     full_layers = hyperparam_args.full_layers
# #
# #     auto_start_layer = int(total_layers * 0.5)
# #     auto_end_layer = total_layers - 1
# #
# #     if target_layers == "" or target_layers == "-1":
# #         hyperparam_args.target_layers = list(range(auto_start_layer, auto_end_layer))
# #         print(f"✅ 모델 자동 감지: 총 {total_layers}개 레이어 중 "
# #               f"{auto_start_layer}번째부터 {auto_end_layer - 1}번째 레이어를 타겟으로 설정합니다.")
# #     elif layers_window_size > 0 and target_layers != "-1":
# #         hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx + layers_window_size))
# #     else:
# #         hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")]
# #
# #     if "-1" in transform_layers:
# #         lora_layers_to_transform = hyperparam_args.target_layers
# #     else:
# #         lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]
# #
# #     # ---------- 클러스터링용 레이어 (target_layers 중간) ----------
# #     target_layers_list = hyperparam_args.target_layers
# #     # cluster_layer = target_layers_list[len(target_layers_list) // 2] # 기존 코드 hidden state를 target_layers의 중간 레이어 사용
# #     cluster_layer = target_layers_list[-1] # target_layers의 마지막 레이어 사용
# #     print(f"✅ 클러스터링 레이어: {cluster_layer} (target_layers={target_layers_list})")
# #
# #     lora_config = LoraConfig(
# #         r=lora_args.lora_r,
# #         lora_alpha=lora_args.lora_alpha,
# #         target_modules=lora_args.lora_target_modules,
# #         lora_dropout=lora_args.lora_dropout,
# #         bias=lora_args.lora_bias,
# #         layers_to_transform=lora_layers_to_transform,
# #         task_type="CAUSAL_LM",
# #     )
# #
# #     drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
# #     print("lora_transform_layers", lora_config.layers_to_transform)
# #     print("drop_layers_after", drop_layers_after)
# #
# #     if drop_layers_after:
# #         config.num_hidden_layers = drop_layers_after + 1
# #
# #     # ---------- Tokenizer ----------
# #     tokenizer = AutoTokenizer.from_pretrained(
# #         model_name_or_path,
# #         model_max_length=training_args.max_seq_length,
# #         padding_side="left",
# #         use_fast=False,
# #     )
# #     tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
# #
# #     # ---------- Dataset (pseudo-label 미주입 상태로 생성) ----------
# #     train_dataset = RepBendingDataset(
# #         tokenizer, num_examples=500, mode=hyperparam_args.loss_mode,
# #         max_length=512, model_name_or_path=model_name_or_path,
# #         dataset_path=hyperparam_args.dataset_path,
# #         split=hyperparam_args.dataset_split,
# #         is_online=hyperparam_args.is_online
# #     )
# #
# #     # ---------- Base 모델 로드 (4bit) ----------
# #     bnb_config = BitsAndBytesConfig(
# #         load_in_4bit=True,
# #         bnb_4bit_quant_type="nf4",
# #         bnb_4bit_compute_dtype=torch.float16,
# #         bnb_4bit_use_double_quant=True,
# #     )
# #
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name_or_path,
# #         config=config,
# #         device_map=device_map,
# #         quantization_config=bnb_config,
# #     )
# #     training_args.model_name_or_path = model_name_or_path
# #
# #     # ============================================================
# #     # ★ 학습 시작 전 클러스터링 단계
# #     #   1. Base 모델로 DNA 프롬프트의 hidden 추출
# #     #   2. K-Means → pseudo-label 생성
# #     #   3. centroid 사전 계산 후 고정
# #     #   4. dataset.set_pseudo_labels() 주입
# #     # ============================================================
# #     print("\n" + "=" * 70)
# #     print("🌀 학습 전 클러스터링 단계")
# #     print("=" * 70)
# #
# #     NUM_CATEGORIES = 7
# #     cache_dir = os.path.join(training_args.output_dir, "cache")
# #     os.makedirs(cache_dir, exist_ok=True)
# #     hidden_cache_path = os.path.join(cache_dir, f"dna_hidden_layer{cluster_layer}.npz")
# #
# #     pseudo_labels, init_centroids = cluster_unsafe_prompts(
# #         base_model=model,
# #         tokenizer=tokenizer,
# #         raw_unsafe_prompts=train_dataset.data_unsafe_request_prompts,
# #         true_labels=train_dataset.data_unsafe_true_labels,
# #         target_layer=cluster_layer,
# #         num_clusters=NUM_CATEGORIES,
# #         cache_path=hidden_cache_path,
# #         random_state=42,
# #     )
# #     train_dataset.set_pseudo_labels(pseudo_labels)
# #
# #     # FixedCentroidHolder: 학습 내내 고정
# #     centroid_holder = FixedCentroidHolder(init_centroids.to(model.device))
# #
# #     # ---------- centroid 즉시 저장 (시각화/inference에서 사용) ----------
# #     os.makedirs(training_args.output_dir, exist_ok=True)
# #     init_centroid_path = os.path.join(training_args.output_dir, "crush_centroids_initial.pt")
# #     torch.save({
# #         "centroids": init_centroids.cpu(),
# #         "pseudo_labels": pseudo_labels,
# #         "true_labels": train_dataset.data_unsafe_true_labels,
# #         "cluster_layer": cluster_layer,
# #         "num_clusters": NUM_CATEGORIES,
# #     }, init_centroid_path)
# #     print(f"💾 초기 centroid 저장: {init_centroid_path}")
# #
# #     # ============================================================
# #     # LoRA 적용 후 학습
# #     # ============================================================
# #     model = get_peft_model(model, lora_config)
# #     print("model", model)
# #
# #     if getattr(training_args, "deepspeed", None) is not None and training_args.local_rank == 0:
# #         model.print_trainable_parameters()
# #
# #     if training_args.gradient_checkpointing:
# #         model.enable_input_require_grads()
# #
# #     print("TRAIN SIZE: ", len(train_dataset))
# #
# #     if hyperparam_args.is_online:
# #         classifier = HarmbenchClassifier()
# #     else:
# #         classifier = None
# #
# #     training_args.remove_unused_columns = False
# #
# #     trainer = CustomTrainer(
# #         model=model,
# #         tokenizer=tokenizer,
# #         args=training_args,
# #         train_dataset=train_dataset,
# #         data_collator=data_collator,
# #         max_seq_length=training_args.max_seq_length,
# #         packing=True,
# #         hyperparam_args=hyperparam_args,
# #         classifier=classifier,
# #         centroid_holder=centroid_holder,  # 고정 centroid 주입
# #     )
# #     model.config.use_cache = False
# #
# #     trainer.train()
# #
# #     # ---------- 학습 완료 후 저장 ----------
# #     trainer.save_model(training_args.output_dir)
# #     # 학습 후에도 centroid는 동일 (고정이었음). evaluate.py 호환 위해 같은 이름으로 저장.
# #     final_centroid_path = os.path.join(training_args.output_dir, "crush_centroids.pt")
# #     torch.save(centroid_holder.centroids.cpu(), final_centroid_path)
# #     print(f"\n✅ 학습 완료!")
# #     print(f"   고정 centroid: {final_centroid_path}")
# #     print(f"   진단/검증 정보: {init_centroid_path}")
# #
# #
# # if __name__ == "__main__":
# #     SEED = 42
# #
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed(SEED)
# #         torch.cuda.manual_seed_all(SEED)
# #     elif torch.backends.mps.is_available():
# #         torch.mps.manual_seed(SEED)
# #
# #     torch.manual_seed(SEED)
# #     np.random.seed(SEED)
# #
# #     train()
#
#
#
#
#
#
#
# import faulthandler
# faulthandler.enable()
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import random
# from dataclasses import dataclass
#
# import numpy as np
# import torch
# import transformers
#
# from args import (HyperparamArguments, LoraArguments, ModelArguments,
#                   TrainingArguments)
# from classifier import HarmbenchClassifier
# from dataset import RepBendingDataset
# from trainer import CustomTrainer
# from peft import LoraConfig, get_peft_model
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#
# from crush_loss import FixedCentroidHolder
# from clustering import cluster_unsafe_prompts
#
#
# def data_collator(batch_list):
#     batch_inputs = {}
#     for features in batch_list:
#         for k, input in features.items():
#             batch_inputs.setdefault(k, []).append(input)
#
#     for k, inputs in batch_inputs.items():
#         if isinstance(inputs[0], torch.Tensor):
#             batch_inputs[k] = torch.cat(inputs, dim=0)
#         elif isinstance(inputs[0], int):
#             batch_inputs[k] = torch.tensor(inputs)
#         else:
#             batch_inputs[k] = inputs
#     return batch_inputs
#
#
# def train():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
#     )
#     (model_args, training_args, lora_args, hyperparam_args) = parser.parse_args_into_dataclasses()
#
#     print(hyperparam_args.to_dict())
#     print(lora_args)
#     print(model_args)
#     print(training_args)
#
#     device_map = "auto"
#     model_name_or_path = model_args.model_name_or_path
#
#     # ---------- 모델 config / target layer 자동 감지 ----------
#     config = AutoConfig.from_pretrained(model_name_or_path)
#     total_layers = config.num_hidden_layers
#
#     target_layers = hyperparam_args.target_layers
#     target_layer_start_idx = hyperparam_args.target_layer_start_idx
#     layers_window_size = hyperparam_args.layers_window_size
#     transform_layers = hyperparam_args.transform_layers
#     full_layers = hyperparam_args.full_layers
#
#     auto_start_layer = int(total_layers * 0.5)
#     auto_end_layer = total_layers - 1
#
#     if target_layers == "" or target_layers == "-1":
#         hyperparam_args.target_layers = list(range(auto_start_layer, auto_end_layer))
#         print(f"✅ 모델 자동 감지: 총 {total_layers}개 레이어 중 "
#               f"{auto_start_layer}번째부터 {auto_end_layer - 1}번째 레이어를 타겟으로 설정합니다.")
#     elif layers_window_size > 0 and target_layers != "-1":
#         hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx + layers_window_size))
#     else:
#         hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")]
#
#     if "-1" in transform_layers:
#         lora_layers_to_transform = hyperparam_args.target_layers
#     else:
#         lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]
#
#     # ---------- 클러스터링용 레이어 (target_layers 중간) ----------
#     target_layers_list = hyperparam_args.target_layers
#     # cluster_layer = target_layers_list[len(target_layers_list) // 2] # 기존 코드 hidden state를 target_layers의 중간 레이어 사용
#     cluster_layer = target_layers_list[-1] # target_layers의 마지막 레이어 사용
#     print(f"✅ 클러스터링 레이어: {cluster_layer} (target_layers={target_layers_list})")
#
#     lora_config = LoraConfig(
#         r=lora_args.lora_r,
#         lora_alpha=lora_args.lora_alpha,
#         target_modules=lora_args.lora_target_modules,
#         lora_dropout=lora_args.lora_dropout,
#         bias=lora_args.lora_bias,
#         layers_to_transform=lora_layers_to_transform,
#         task_type="CAUSAL_LM",
#     )
#
#     drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
#     print("lora_transform_layers", lora_config.layers_to_transform)
#     print("drop_layers_after", drop_layers_after)
#
#     if drop_layers_after:
#         config.num_hidden_layers = drop_layers_after + 1
#
#     # ---------- Tokenizer ----------
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path,
#         model_max_length=training_args.max_seq_length,
#         padding_side="left",
#         use_fast=False,
#     )
#     tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
#
#     # ---------- Dataset (pseudo-label 미주입 상태로 생성) ----------
#     train_dataset = RepBendingDataset(
#         tokenizer, num_examples=500, mode=hyperparam_args.loss_mode,
#         max_length=512, model_name_or_path=model_name_or_path,
#         dataset_path=hyperparam_args.dataset_path,
#         split=hyperparam_args.dataset_split,
#         is_online=hyperparam_args.is_online
#     )
#
#     # ---------- Base 모델 로드 (4bit) ----------
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#     )
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         config=config,
#         device_map=device_map,
#         quantization_config=bnb_config,
#     )
#     training_args.model_name_or_path = model_name_or_path
#
#     # ============================================================
#     # ★ 학습 시작 전 클러스터링 단계
#     #   1. Base 모델로 DNA 프롬프트의 hidden 추출
#     #   2. K-Means → pseudo-label 생성
#     #   3. centroid 사전 계산 후 고정
#     #   4. dataset.set_pseudo_labels() 주입
#     # ============================================================
#     print("\n" + "=" * 70)
#     print("🌀 학습 전 클러스터링 단계")
#     print("=" * 70)
#
#     NUM_CATEGORIES = 5
#     cache_dir = os.path.join(training_args.output_dir, "cache")
#     os.makedirs(cache_dir, exist_ok=True)
#     hidden_cache_path = os.path.join(cache_dir, f"dna_hidden_layer{cluster_layer}.npz")
#
#     pseudo_labels, init_centroids = cluster_unsafe_prompts(
#         base_model=model,
#         tokenizer=tokenizer,
#         raw_unsafe_prompts=train_dataset.data_unsafe_request_prompts,
#         true_labels=train_dataset.data_unsafe_true_labels,
#         target_layer=cluster_layer,
#         num_clusters=NUM_CATEGORIES,
#         cache_path=hidden_cache_path,
#         random_state=42,
#     )
#     train_dataset.set_pseudo_labels(pseudo_labels)
#
#     # FixedCentroidHolder: 학습 내내 고정
#     centroid_holder = FixedCentroidHolder(init_centroids.to(model.device))
#
#     # ---------- centroid 즉시 저장 (시각화/inference에서 사용) ----------
#     os.makedirs(training_args.output_dir, exist_ok=True)
#     init_centroid_path = os.path.join(training_args.output_dir, "crush_centroids_initial.pt")
#     torch.save({
#         "centroids": init_centroids.cpu(),
#         "pseudo_labels": pseudo_labels,
#         "true_labels": train_dataset.data_unsafe_true_labels,
#         "cluster_layer": cluster_layer,
#         "num_clusters": NUM_CATEGORIES,
#     }, init_centroid_path)
#     print(f"💾 초기 centroid 저장: {init_centroid_path}")
#
#     # ============================================================
#     # LoRA 적용 후 학습
#     # ============================================================
#     model = get_peft_model(model, lora_config)
#     print("model", model)
#
#     if getattr(training_args, "deepspeed", None) is not None and training_args.local_rank == 0:
#         model.print_trainable_parameters()
#
#     if training_args.gradient_checkpointing:
#         model.enable_input_require_grads()
#
#     print("TRAIN SIZE: ", len(train_dataset))
#
#     if hyperparam_args.is_online:
#         classifier = HarmbenchClassifier()
#     else:
#         classifier = None
#
#     training_args.remove_unused_columns = False
#
#     trainer = CustomTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#         max_seq_length=training_args.max_seq_length,
#         packing=True,
#         hyperparam_args=hyperparam_args,
#         classifier=classifier,
#         centroid_holder=centroid_holder,  # 고정 centroid 주입
#     )
#     model.config.use_cache = False
#
#     trainer.train()
#
#     # ---------- 학습 완료 후 저장 ----------
#     trainer.save_model(training_args.output_dir)
#     # 학습 후에도 centroid는 동일 (고정이었음). evaluate.py 호환 위해 같은 이름으로 저장.
#     final_centroid_path = os.path.join(training_args.output_dir, "crush_centroids.pt")
#     torch.save(centroid_holder.centroids.cpu(), final_centroid_path)
#     print(f"\n✅ 학습 완료!")
#     print(f"   고정 centroid: {final_centroid_path}")
#     print(f"   진단/검증 정보: {init_centroid_path}")
#
#
# if __name__ == "__main__":
#     SEED = 42
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#         torch.cuda.manual_seed_all(SEED)
#     elif torch.backends.mps.is_available():
#         torch.mps.manual_seed(SEED)
#
#     torch.manual_seed(SEED)
#     np.random.seed(SEED)
#
#     train()


# 7개 클러스터 시
# import faulthandler
# faulthandler.enable()
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# import random
# from dataclasses import dataclass
#
# import numpy as np
# import torch
# import transformers
#
# from args import (HyperparamArguments, LoraArguments, ModelArguments,
#                   TrainingArguments)
# from classifier import HarmbenchClassifier
# from dataset import RepBendingDataset
# from trainer import CustomTrainer
# from peft import LoraConfig, get_peft_model
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#
# from crush_loss import FixedCentroidHolder
# from clustering import cluster_unsafe_prompts, extract_all_layers_hidden_and_cluster
#
#
# def data_collator(batch_list):
#     batch_inputs = {}
#     for features in batch_list:
#         for k, input in features.items():
#             batch_inputs.setdefault(k, []).append(input)
#
#     for k, inputs in batch_inputs.items():
#         if isinstance(inputs[0], torch.Tensor):
#             batch_inputs[k] = torch.cat(inputs, dim=0)
#         elif isinstance(inputs[0], int):
#             batch_inputs[k] = torch.tensor(inputs)
#         else:
#             batch_inputs[k] = inputs
#     return batch_inputs
#
#
# def train():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
#     )
#     (model_args, training_args, lora_args, hyperparam_args) = parser.parse_args_into_dataclasses()
#
#     print(hyperparam_args.to_dict())
#     print(lora_args)
#     print(model_args)
#     print(training_args)
#
#     device_map = "auto"
#     model_name_or_path = model_args.model_name_or_path
#
#     # ---------- 모델 config / target layer 자동 감지 ----------
#     config = AutoConfig.from_pretrained(model_name_or_path)
#     total_layers = config.num_hidden_layers
#
#     target_layers = hyperparam_args.target_layers
#     target_layer_start_idx = hyperparam_args.target_layer_start_idx
#     layers_window_size = hyperparam_args.layers_window_size
#     transform_layers = hyperparam_args.transform_layers
#     full_layers = hyperparam_args.full_layers
#
#     auto_start_layer = int(total_layers * 0.5)
#     auto_end_layer = total_layers - 1
#
#     if target_layers == "" or target_layers == "-1":
#         hyperparam_args.target_layers = list(range(auto_start_layer, auto_end_layer))
#         print(f"✅ 모델 자동 감지: 총 {total_layers}개 레이어 중 "
#               f"{auto_start_layer}번째부터 {auto_end_layer - 1}번째 레이어를 타겟으로 설정합니다.")
#     elif layers_window_size > 0 and target_layers != "-1":
#         hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx + layers_window_size))
#     else:
#         hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")]
#
#     if "-1" in transform_layers:
#         lora_layers_to_transform = hyperparam_args.target_layers
#     else:
#         lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]
#
#     # ---------- 클러스터링용 레이어 (target_layers 중간) ----------
#     target_layers_list = hyperparam_args.target_layers
#     # cluster_layer = target_layers_list[len(target_layers_list) // 2] # 기존 코드 hidden state를 target_layers의 중간 레이어 사용
#     cluster_layer = target_layers_list[-1] # target_layers의 마지막 레이어 사용
#     print(f"✅ 클러스터링 레이어: {cluster_layer} (target_layers={target_layers_list})")
#
#     lora_config = LoraConfig(
#         r=lora_args.lora_r,
#         lora_alpha=lora_args.lora_alpha,
#         target_modules=lora_args.lora_target_modules,
#         lora_dropout=lora_args.lora_dropout,
#         bias=lora_args.lora_bias,
#         layers_to_transform=lora_layers_to_transform,
#         task_type="CAUSAL_LM",
#     )
#
#     drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
#     print("lora_transform_layers", lora_config.layers_to_transform)
#     print("drop_layers_after", drop_layers_after)
#
#     if drop_layers_after:
#         config.num_hidden_layers = drop_layers_after + 1
#
#     # ---------- Tokenizer ----------
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path,
#         model_max_length=training_args.max_seq_length,
#         padding_side="left",
#         use_fast=False,
#     )
#     tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
#
#     # ---------- Dataset (pseudo-label 미주입 상태로 생성) ----------
#     train_dataset = RepBendingDataset(
#         tokenizer, num_examples=500, mode=hyperparam_args.loss_mode,
#         max_length=512, model_name_or_path=model_name_or_path,
#         dataset_path=hyperparam_args.dataset_path,
#         split=hyperparam_args.dataset_split,
#         is_online=hyperparam_args.is_online
#     )
#
#     # ---------- Base 모델 로드 (4bit) ----------
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#     )
#
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path,
#         config=config,
#         device_map=device_map,
#         quantization_config=bnb_config,
#     )
#     training_args.model_name_or_path = model_name_or_path
#
#     # ============================================================
#     # ★ 학습 시작 전 클러스터링 단계
#     #   1. Base 모델로 DNA 프롬프트의 hidden 추출
#     #   2. K-Means → pseudo-label 생성
#     #   3. centroid 사전 계산 후 고정
#     #   4. dataset.set_pseudo_labels() 주입
#     # ============================================================
#     print("\n" + "=" * 70)
#     print("🌀 학습 전 클러스터링 단계")
#     print("=" * 70)
#
#     NUM_CATEGORIES = 7
#     cache_dir = os.path.join(training_args.output_dir, "cache")
#     os.makedirs(cache_dir, exist_ok=True)
#     hidden_cache_path = os.path.join(cache_dir, f"dna_hidden_layer{cluster_layer}.npz")
#
#     pseudo_labels, init_centroids = cluster_unsafe_prompts(
#         base_model=model,
#         tokenizer=tokenizer,
#         raw_unsafe_prompts=train_dataset.data_unsafe_request_prompts,
#         true_labels=train_dataset.data_unsafe_true_labels,
#         target_layer=cluster_layer,
#         num_clusters=NUM_CATEGORIES,
#         cache_path=hidden_cache_path,
#         random_state=42,
#     )
#     train_dataset.set_pseudo_labels(pseudo_labels)
#
#     # FixedCentroidHolder: 학습 내내 고정
#     centroid_holder = FixedCentroidHolder(init_centroids.to(model.device))
#
#     # ---------- centroid 즉시 저장 (시각화/inference에서 사용) ----------
#     os.makedirs(training_args.output_dir, exist_ok=True)
#     init_centroid_path = os.path.join(training_args.output_dir, "crush_centroids_initial.pt")
#     torch.save({
#         "centroids": init_centroids.cpu(),
#         "pseudo_labels": pseudo_labels,
#         "true_labels": train_dataset.data_unsafe_true_labels,
#         "cluster_layer": cluster_layer,
#         "num_clusters": NUM_CATEGORIES,
#     }, init_centroid_path)
#     print(f"💾 초기 centroid 저장: {init_centroid_path}")
#
#     # ============================================================
#     # LoRA 적용 후 학습
#     # ============================================================
#     model = get_peft_model(model, lora_config)
#     print("model", model)
#
#     if getattr(training_args, "deepspeed", None) is not None and training_args.local_rank == 0:
#         model.print_trainable_parameters()
#
#     if training_args.gradient_checkpointing:
#         model.enable_input_require_grads()
#
#     print("TRAIN SIZE: ", len(train_dataset))
#
#     if hyperparam_args.is_online:
#         classifier = HarmbenchClassifier()
#     else:
#         classifier = None
#
#     training_args.remove_unused_columns = False
#
#     trainer = CustomTrainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=train_dataset,
#         data_collator=data_collator,
#         max_seq_length=training_args.max_seq_length,
#         packing=True,
#         hyperparam_args=hyperparam_args,
#         classifier=classifier,
#         centroid_holder=centroid_holder,  # 고정 centroid 주입
#     )
#     model.config.use_cache = False
#
#     trainer.train()
#
#     # ---------- 학습 완료 후 저장 ----------
#     trainer.save_model(training_args.output_dir)
#     # 학습 후에도 centroid는 동일 (고정이었음). evaluate.py 호환 위해 같은 이름으로 저장.
#     final_centroid_path = os.path.join(training_args.output_dir, "crush_centroids.pt")
#     torch.save(centroid_holder.centroids.cpu(), final_centroid_path)
#     print(f"\n✅ 학습 완료!")
#     print(f"   고정 centroid: {final_centroid_path}")
#     print(f"   진단/검증 정보: {init_centroid_path}")
#
#
# if __name__ == "__main__":
#     SEED = 42
#
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#         torch.cuda.manual_seed_all(SEED)
#     elif torch.backends.mps.is_available():
#         torch.mps.manual_seed(SEED)
#
#     torch.manual_seed(SEED)
#     np.random.seed(SEED)
#
#     train()


import faulthandler

faulthandler.enable()
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import random
from dataclasses import dataclass

import numpy as np
import torch
import transformers

from args import (HyperparamArguments, LoraArguments, ModelArguments,
                  TrainingArguments)
from classifier import HarmbenchClassifier
from dataset import RepBendingDataset
from trainer import CustomTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from crush_loss import FixedCentroidHolder
from clustering import cluster_unsafe_prompts, extract_all_layers_hidden_and_cluster


def data_collator(batch_list):
    batch_inputs = {}
    for features in batch_list:
        for k, input in features.items():
            batch_inputs.setdefault(k, []).append(input)

    for k, inputs in batch_inputs.items():
        if isinstance(inputs[0], torch.Tensor):
            batch_inputs[k] = torch.cat(inputs, dim=0)
        elif isinstance(inputs[0], int):
            batch_inputs[k] = torch.tensor(inputs)
        else:
            batch_inputs[k] = inputs
    return batch_inputs


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, HyperparamArguments)
    )
    (model_args, training_args, lora_args, hyperparam_args) = parser.parse_args_into_dataclasses()

    print(hyperparam_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    device_map = "auto"
    model_name_or_path = model_args.model_name_or_path

    # ---------- 모델 config / target layer 자동 감지 ----------
    config = AutoConfig.from_pretrained(model_name_or_path)
    total_layers = config.num_hidden_layers

    target_layers = hyperparam_args.target_layers
    target_layer_start_idx = hyperparam_args.target_layer_start_idx
    layers_window_size = hyperparam_args.layers_window_size
    transform_layers = hyperparam_args.transform_layers
    full_layers = hyperparam_args.full_layers

    auto_start_layer = int(total_layers * 0.5)
    auto_end_layer = total_layers - 1

    if target_layers == "" or target_layers == "-1":
        hyperparam_args.target_layers = list(range(auto_start_layer, auto_end_layer))
        print(f"✅ 모델 자동 감지: 총 {total_layers}개 레이어 중 "
              f"{auto_start_layer}번째부터 {auto_end_layer - 1}번째 레이어를 타겟으로 설정합니다.")
    elif layers_window_size > 0 and target_layers != "-1":
        hyperparam_args.target_layers = list(range(target_layer_start_idx, target_layer_start_idx + layers_window_size))
    else:
        hyperparam_args.target_layers = [int(layer) for layer in target_layers.split(",")]

    if "-1" in transform_layers:
        lora_layers_to_transform = hyperparam_args.target_layers
    else:
        lora_layers_to_transform = [int(layer) for layer in transform_layers.split(",")]

    # ---------- 클러스터링용 레이어 (target_layers 중간) ----------
    target_layers_list = hyperparam_args.target_layers
    # cluster_layer = target_layers_list[len(target_layers_list) // 2] # 기존 코드 hidden state를 target_layers의 중간 레이어 사용
    cluster_layer = target_layers_list[-1]  # target_layers의 마지막 레이어 사용
    print(f"✅ 클러스터링 레이어: {cluster_layer} (target_layers={target_layers_list})")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    drop_layers_after = max(hyperparam_args.target_layers) if not full_layers else None
    print("lora_transform_layers", lora_config.layers_to_transform)
    print("drop_layers_after", drop_layers_after)

    if drop_layers_after:
        config.num_hidden_layers = drop_layers_after + 1

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=training_args.max_seq_length,
        padding_side="left",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # ---------- Dataset (pseudo-label 미주입 상태로 생성) ----------
    train_dataset = RepBendingDataset(
        tokenizer, num_examples=500, mode=hyperparam_args.loss_mode,
        max_length=512, model_name_or_path=model_name_or_path,
        dataset_path=hyperparam_args.dataset_path,
        split=hyperparam_args.dataset_split,
        is_online=hyperparam_args.is_online
    )

    # ---------- Base 모델 로드 (4bit) ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    training_args.model_name_or_path = model_name_or_path

    # ============================================================
    # ★ 학습 시작 전 클러스터링 단계
    #   1. Base 모델로 DNA 프롬프트의 hidden 추출
    #   2. K-Means → pseudo-label 생성
    #   3. centroid 사전 계산 후 고정
    #   4. dataset.set_pseudo_labels() 주입
    # ============================================================
    print("\n" + "=" * 70)
    print("🌀 학습 전 클러스터링 단계")
    print("=" * 70)

    NUM_CATEGORIES = 5
    cache_dir = os.path.join(training_args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    hidden_cache_path = os.path.join(cache_dir, f"dna_hidden_layer{cluster_layer}.npz")

    pseudo_labels, init_centroids = cluster_unsafe_prompts(
        base_model=model,
        tokenizer=tokenizer,
        raw_unsafe_prompts=train_dataset.data_unsafe_request_prompts,
        true_labels=train_dataset.data_unsafe_true_labels,
        target_layer=cluster_layer,
        num_clusters=NUM_CATEGORIES,
        cache_path=hidden_cache_path,
        random_state=42,
    )
    train_dataset.set_pseudo_labels(pseudo_labels)

    # ============================================================
    # ★ 전체 target_layers 별 hidden vector + 클러스터 정보 저장
    #   - t-SNE 시각화 및 레이어별 클러스터링 분석에 사용
    #   - Base 모델(LoRA 전) 상태에서 추출하므로 LoRA 적용 전에 실행
    # ============================================================
    # 시각화용 hidden 저장 범위: target_layers 시작점 ~ 모델 마지막 레이어
    vis_layers = list(range(target_layers_list[0], total_layers + 1))

    # ── LoRA OFF (Base 모델) hidden 저장 ──────────────────────────
    print("\n💾 [LoRA OFF] 레이어별 hidden 저장 중...")
    extract_all_layers_hidden_and_cluster(
        model=model,
        tokenizer=tokenizer,
        prompts=train_dataset.data_unsafe_request_prompts,
        target_layers=vis_layers,
        pseudo_labels=pseudo_labels,
        true_labels=train_dataset.data_unsafe_true_labels,
        output_dir=training_args.output_dir,
        tag="lora_off",  # Base 모델 (LoRA 전)
    )

    # FixedCentroidHolder: 학습 내내 고정
    centroid_holder = FixedCentroidHolder(init_centroids.to(model.device))

    # ---------- centroid 즉시 저장 (시각화/inference에서 사용) ----------
    os.makedirs(training_args.output_dir, exist_ok=True)
    init_centroid_path = os.path.join(training_args.output_dir, "crush_centroids_initial.pt")
    torch.save({
        "centroids": init_centroids.cpu(),
        "pseudo_labels": pseudo_labels,
        "true_labels": train_dataset.data_unsafe_true_labels,
        "cluster_layer": cluster_layer,
        "num_clusters": NUM_CATEGORIES,
    }, init_centroid_path)
    print(f"💾 초기 centroid 저장: {init_centroid_path}")

    # ============================================================
    # LoRA 적용 후 학습
    # ============================================================
    model = get_peft_model(model, lora_config)
    print("model", model)

    if getattr(training_args, "deepspeed", None) is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    print("TRAIN SIZE: ", len(train_dataset))

    if hyperparam_args.is_online:
        classifier = HarmbenchClassifier()
    else:
        classifier = None

    training_args.remove_unused_columns = False

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        max_seq_length=training_args.max_seq_length,
        packing=True,
        hyperparam_args=hyperparam_args,
        classifier=classifier,
        centroid_holder=centroid_holder,  # 고정 centroid 주입
    )
    model.config.use_cache = False

    trainer.train()

    # ── LoRA ON (Fine-tuned 모델) hidden 저장 ─────────────────────
    print("\n💾 [LoRA ON] 레이어별 hidden 저장 중...")
    extract_all_layers_hidden_and_cluster(
        model=model,
        tokenizer=tokenizer,
        prompts=train_dataset.data_unsafe_request_prompts,
        target_layers=vis_layers,
        pseudo_labels=pseudo_labels,
        true_labels=train_dataset.data_unsafe_true_labels,
        output_dir=training_args.output_dir,
        tag="lora_on",  # Fine-tuned 모델 (LoRA 후)
    )

    # ---------- 학습 완료 후 저장 ----------
    trainer.save_model(training_args.output_dir)
    # 학습 후에도 centroid는 동일 (고정이었음). evaluate.py 호환 위해 같은 이름으로 저장.
    final_centroid_path = os.path.join(training_args.output_dir, "crush_centroids.pt")
    torch.save(centroid_holder.centroids.cpu(), final_centroid_path)
    print(f"\n✅ 학습 완료!")
    print(f"   고정 centroid: {final_centroid_path}")
    print(f"   진단/검증 정보: {init_centroid_path}")


if __name__ == "__main__":
    SEED = 42

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train()
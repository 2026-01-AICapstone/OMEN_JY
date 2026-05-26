"""
evaluate_after_plot_json_overlay.py
==================================

목적:
1) 학습 데이터(DNA)로 After LoRA latent vector 추출
2) After LoRA 기준 K-Means로 학습 데이터 클러스터/카테고리 매핑 계산
3) failed_defense_augmented_dna.jsonl 질문은 After LoRA vector만 1번 추출
4) JSONL 질문의 예측 카테고리를 계산
5) 기존 After LoRA plot 좌표를 저장하고, 그 좌표 위에 JSONL 질문들을 함께 표시
6) JSONL 질문에 대해서만 LoRA OFF / LoRA ON 답변 저장

중요:
- sklearn TSNE는 transform()이 없어서, 이미 만든 t-SNE 좌표계에 새 점을 정확히 투영할 수 없음.
- 따라서 학습 데이터의 After t-SNE 좌표는 고정 저장하고,
  JSONL 질문은 고차원 벡터 공간에서 가까운 학습 샘플들의 t-SNE 좌표 평균 위치에 overlay함.
- 이렇게 하면 학습 데이터 plot 위치는 바뀌지 않고, JSONL 질문이 어느 영역에 가까운지 확인할 수 있음.

출력:
- after_lora_train_tsne_coords.npy
- after_lora_train_tsne_plot_with_failed_json.png
- failed_json_predicted_categories.csv
- failed_json_predicted_categories.jsonl
- failed_json_lora_on_off_answers.txt
- failed_json_lora_on_off_answers.jsonl
"""

import os
import csv
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dataset import RepBendingDataset

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment


# =========================================================
# 설정
# =========================================================
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
# BASE_MODEL = "google/gemma-2-2b-it"

LORA_PATH = "./out/test"
CENTROIDS_PATH = "./out/test/crush_centroids.pt"
INIT_CENTROIDS_PATH = "./out/test/crush_centroids_initial.pt"

FAILED_JSONL_PATH = "./failed_defense_augmented_dna.jsonl"
FAILED_JSONL_QUESTION_FIELD = "question"        # "question" 또는 "original_question"
MAX_FAILED_SAMPLES = None                       # 테스트용이면 숫자 지정, 전체는 None
MAX_NEW_TOKENS = 256
NUM_TRAIN_SAMPLES_PER_CLASS = 50

# JSONL 점을 t-SNE 좌표 위에 얹을 때 사용할 이웃 수
OVERLAY_K_NEIGHBORS = 5

true_label_names = {
    0: "Malicious",
    1: "ChatbotHarm",
    2: "InfoHazard",
    3: "Misinfo",
    4: "Discrim",
}

risk_area_to_label = {
    "Malicious Uses": 0,
    "Human-Chatbot Interaction Harms": 1,
    "Information Hazards": 2,
    "Misinformation Harms": 3,
    "Discrimination, Exclusion, Toxicity, Hateful, Offensive": 4,
}


# =========================================================
# 로더
# =========================================================
def load_failed_jsonl(jsonl_path, question_field="question", max_samples=None):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL 파일을 찾을 수 없습니다: {jsonl_path}")

    rows, prompts, labels = [], [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            prompt = obj.get(question_field)
            if not prompt:
                prompt = obj.get("question") or obj.get("original_question") or obj.get("prompt")
            if not prompt:
                continue

            risk_area = obj.get("risk_area", "")
            true_label = risk_area_to_label.get(risk_area, -1)

            rows.append({
                "jsonl_index": line_idx,
                "question": prompt,
                "original_question": obj.get("original_question", ""),
                "risk_area": risk_area,
                "true_label": true_label,
            })
            prompts.append(prompt)
            labels.append(true_label)

            if max_samples is not None and len(prompts) >= max_samples:
                break

    print(f"JSONL 로드: {len(prompts)}개 ({jsonl_path}, field={question_field})")
    return rows, prompts, labels


def load_train_dataset_samples(base_model_name, num_samples_per_class=50):
    print("\n학습 데이터셋 로딩 중...")
    dummy_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dummy_tokenizer.pad_token = dummy_tokenizer.eos_token or dummy_tokenizer.unk_token

    train_dataset = RepBendingDataset(
        tokenizer=dummy_tokenizer,
        num_examples=500,
        mode="response_all",
        max_length=512,
        model_name_or_path=base_model_name,
        dataset_path="LibrAI/do-not-answer",
        split="train",
        is_online=False,
    )

    prompts, labels = [], []
    true_arr = np.array(train_dataset.data_unsafe_true_labels)

    for label_idx in range(5):
        indices = np.where(true_arr == label_idx)[0][:num_samples_per_class]
        for idx in indices:
            prompts.append(train_dataset.data_unsafe_request_prompts[idx])
            labels.append(label_idx)

    print(f"학습 평가 샘플 수: {len(prompts)}개")
    for i, cnt in enumerate(np.bincount(labels, minlength=5)):
        print(f"  [{i}] {true_label_names[i]}: {cnt}개")

    return prompts, labels


# =========================================================
# 모델/벡터
# =========================================================
def load_crush_model_and_meta(base_model_name, lora_weights_path, centroids_path, init_centroids_path):
    print("1. 모델/토크나이저 로드...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )

    print("2. LoRA 어댑터 적용...")
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()

    print("3. centroid/meta 로드...")
    centroids = None
    if os.path.exists(centroids_path):
        centroids = torch.load(centroids_path, map_location=model.device)

    cluster_layer = -1
    init_meta = None
    if os.path.exists(init_centroids_path):
        init_meta = torch.load(init_centroids_path, map_location="cpu")
        if isinstance(init_meta, dict) and "cluster_layer" in init_meta:
            cluster_layer = init_meta["cluster_layer"]
            print(f"   학습 시 사용된 cluster_layer: {cluster_layer}")
    else:
        print(f"   ⚠️ {init_centroids_path} 없음. cluster_layer=-1 사용.")

    return model, tokenizer, centroids, cluster_layer, init_meta


def extract_hidden_vectors(prompts, model, tokenizer, target_layer, title="벡터"):
    vectors = []
    model.eval()

    for i, prompt in enumerate(prompts):
        if "<SEPARATOR>" in prompt:
            prompt = prompt.split("<SEPARATOR>")[0]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer]
        vec = hidden[:, -1, :].float()
        vec = F.normalize(vec, p=2, dim=-1)
        vectors.append(vec[0].cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  {title} 추출 {i + 1}/{len(prompts)}...")

    return np.array(vectors)


# =========================================================
# 클러스터/매핑
# =========================================================
def hungarian_matching_accuracy(true_labels, pred_labels, num_classes=5):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind.tolist(), row_ind.tolist()))  # cluster_id -> true_label
    matched = sum(cm[row_ind[i], col_ind[i]] for i in range(num_classes))
    acc = matched / len(true_labels)
    return acc, mapping, cm


def fit_after_kmeans(train_vectors_after, train_labels, num_clusters=5):
    print(f"\n[After LoRA] 학습 데이터 K-Means(k={num_clusters}) 실행...")
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42, max_iter=300)
    pred = km.fit_predict(train_vectors_after)

    nmi = normalized_mutual_info_score(train_labels, pred)
    ari = adjusted_rand_score(train_labels, pred)
    acc, mapping, cm = hungarian_matching_accuracy(train_labels, pred, num_classes=num_clusters)

    print(f"  NMI                : {nmi:.4f}")
    print(f"  ARI                : {ari:.4f}")
    print(f"  Hungarian Accuracy : {acc:.4f}")
    print(f"  cluster -> label mapping: {mapping}")

    return {
        "kmeans": km,
        "pred": pred,
        "mapping": mapping,
        "cm": cm,
        "nmi": nmi,
        "ari": ari,
        "hungarian_acc": acc,
    }


def predict_failed_categories(rows, failed_vectors_after, kmeans_after, mapping,
                              save_csv="failed_json_predicted_categories.csv",
                              save_jsonl="failed_json_predicted_categories.jsonl"):
    pred_clusters = kmeans_after.predict(failed_vectors_after)
    pred_labels = np.array([int(mapping.get(int(c), -1)) for c in pred_clusters])

    out_rows = []
    for row, pred_c, pred_l in zip(rows, pred_clusters, pred_labels):
        out = dict(row)
        out["pred_cluster"] = int(pred_c)
        out["pred_label"] = int(pred_l)
        out["pred_label_name"] = true_label_names.get(int(pred_l), f"C{pred_l}")
        out_rows.append(out)

    with open(save_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "jsonl_index", "question", "original_question", "risk_area", "true_label",
            "pred_cluster", "pred_label", "pred_label_name",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(save_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nJSONL 질문 예측 카테고리 저장: {save_csv}, {save_jsonl}")
    print("[JSONL 질문 예측 카테고리 분포]")
    counts = np.bincount(pred_labels[pred_labels >= 0], minlength=5)
    for i, cnt in enumerate(counts):
        print(f"  [{i}] {true_label_names[i]:<12}: {cnt}개")

    return pred_clusters, pred_labels, out_rows


# =========================================================
# After plot 좌표 저장 + JSONL overlay
# =========================================================
def compute_or_load_after_tsne(train_vectors_after, save_coords="after_lora_train_tsne_coords.npy"):
    if os.path.exists(save_coords):
        coords = np.load(save_coords)
        if coords.shape[0] == train_vectors_after.shape[0] and coords.shape[1] == 2:
            print(f"\n기존 After LoRA t-SNE 좌표 로드: {save_coords}")
            return coords
        else:
            print(f"\n기존 좌표 shape 불일치. t-SNE 재계산: {coords.shape} vs {train_vectors_after.shape}")

    print("\nAfter LoRA 학습 데이터 t-SNE 좌표 계산...")
    perp = min(30, len(train_vectors_after) - 1)
    coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(train_vectors_after)
    np.save(save_coords, coords)
    print(f"After LoRA t-SNE 좌표 저장: {save_coords}")
    return coords


def project_failed_to_saved_tsne(train_vectors_after, train_tsne_coords, failed_vectors_after, k=5):
    """
    sklearn TSNE는 새 샘플 transform이 불가능하므로,
    고차원 After vector 공간에서 가까운 학습 샘플 k개의 t-SNE 좌표 평균으로 새 점을 overlay함.
    """
    dists = euclidean_distances(failed_vectors_after, train_vectors_after)
    nn_idx = np.argsort(dists, axis=1)[:, :k]

    # 가까울수록 큰 가중치. 0 distance 방지.
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    weights = 1.0 / (nn_dists + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)

    failed_coords = np.sum(train_tsne_coords[nn_idx] * weights[:, :, None], axis=1)
    return failed_coords


def plot_after_tsne_with_failed_overlay(train_tsne_coords, train_labels,
                                        failed_tsne_coords, failed_pred_labels,
                                        label_names,
                                        save_path="after_lora_train_tsne_plot_with_failed_json.png"):
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    plt.figure(figsize=(12, 9))
    plt.title(
        "After LoRA Latent Space + Failed JSONL Questions\n"
        "circle=train true label, X=jsonl predicted category",
        fontsize=13,
        fontweight="bold",
    )

    train_labels = np.asarray(train_labels)
    failed_pred_labels = np.asarray(failed_pred_labels)

    for label_idx in sorted(np.unique(train_labels)):
        mask = train_labels == label_idx
        plt.scatter(
            train_tsne_coords[mask, 0], train_tsne_coords[mask, 1],
            c=colors[int(label_idx) % len(colors)],
            alpha=0.35,
            s=45,
            marker="o",
            label=f"train {label_idx}:{label_names.get(int(label_idx), f'C{label_idx}')}",
        )

    for label_idx in sorted(np.unique(failed_pred_labels)):
        if label_idx < 0:
            continue
        mask = failed_pred_labels == label_idx
        plt.scatter(
            failed_tsne_coords[mask, 0], failed_tsne_coords[mask, 1],
            c=colors[int(label_idx) % len(colors)],
            alpha=0.95,
            s=100,
            marker="x",
            linewidths=2.0,
            label=f"json pred {label_idx}:{label_names.get(int(label_idx), f'C{label_idx}')}",
        )

    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"plot 저장: {save_path}")


# =========================================================
# 답변 생성: JSONL 질문만 LoRA OFF/ON
# =========================================================
def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def generate_answer_pair(model, tokenizer, prompt, max_new_tokens=256):
    with model.disable_adapter():
        off_answer = generate_answer(model, tokenizer, prompt, max_new_tokens)
    on_answer = generate_answer(model, tokenizer, prompt, max_new_tokens)
    return off_answer, on_answer


def save_failed_json_answers(model, tokenizer, predicted_rows, max_new_tokens=256,
                             save_txt="failed_json_lora_on_off_answers.txt",
                             save_jsonl="failed_json_lora_on_off_answers.jsonl"):
    print("\nJSONL 질문에 대한 LoRA OFF/ON 답변 생성 중...")

    with open(save_txt, "w", encoding="utf-8") as ft, open(save_jsonl, "w", encoding="utf-8") as fj:
        ft.write("=" * 100 + "\n")
        ft.write("FAILED JSONL QUESTIONS: LoRA OFF / LoRA ON ANSWERS\n")
        ft.write("=" * 100 + "\n\n")

        for i, row in enumerate(predicted_rows, start=1):
            prompt = row["question"]
            off_answer, on_answer = generate_answer_pair(model, tokenizer, prompt, max_new_tokens)

            out = dict(row)
            out["lora_off_answer"] = off_answer
            out["lora_on_answer"] = on_answer
            fj.write(json.dumps(out, ensure_ascii=False) + "\n")

            ft.write("=" * 100 + "\n")
            ft.write(f"[FAILED JSON #{i}] jsonl_index={row['jsonl_index']}\n")
            ft.write(f"PRED_CLUSTER : C{row['pred_cluster']}\n")
            ft.write(f"PRED_LABEL   : {row['pred_label']} / {row['pred_label_name']}\n")
            ft.write(f"RISK_AREA    : {row.get('risk_area', '')}\n\n")
            ft.write("QUESTION:\n")
            ft.write(prompt.strip() + "\n\n")
            if row.get("original_question"):
                ft.write("ORIGINAL_QUESTION:\n")
                ft.write(row["original_question"].strip() + "\n\n")
            ft.write("BASE MODEL ANSWER (LoRA OFF):\n")
            ft.write(off_answer.strip() + "\n\n")
            ft.write("LORA MODEL ANSWER (LoRA ON):\n")
            ft.write(on_answer.strip() + "\n\n")

            if i % 20 == 0:
                print(f"  답변 생성 {i}/{len(predicted_rows)}...")

    print(f"답변 저장 완료: {save_txt}, {save_jsonl}")


# =========================================================
# 메인
# =========================================================
if __name__ == "__main__":
    model, tokenizer, centroids, cluster_layer, init_meta = load_crush_model_and_meta(
        BASE_MODEL, LORA_PATH, CENTROIDS_PATH, INIT_CENTROIDS_PATH
    )

    train_prompts, train_labels = load_train_dataset_samples(
        BASE_MODEL,
        num_samples_per_class=NUM_TRAIN_SAMPLES_PER_CLASS,
    )
    failed_rows, failed_prompts, failed_true_labels = load_failed_jsonl(
        FAILED_JSONL_PATH,
        question_field=FAILED_JSONL_QUESTION_FIELD,
        max_samples=MAX_FAILED_SAMPLES,
    )

    # 핵심 변경: plot/예측용 벡터는 After LoRA만 추출함.
    print(f"\n[After LoRA] 학습 데이터 벡터 추출 (layer={cluster_layer})...")
    train_vectors_after = extract_hidden_vectors(
        train_prompts, model, tokenizer, target_layer=cluster_layer, title="학습 데이터"
    )

    print(f"\n[After LoRA] JSONL 질문 벡터 추출 (layer={cluster_layer})...")
    failed_vectors_after = extract_hidden_vectors(
        failed_prompts, model, tokenizer, target_layer=cluster_layer, title="JSONL 질문"
    )

    # After LoRA 기준으로 K-Means 학습/매핑
    rec_a = fit_after_kmeans(train_vectors_after, train_labels, num_clusters=5)

    # JSONL 질문 예측 카테고리
    failed_pred_clusters, failed_pred_labels, predicted_rows = predict_failed_categories(
        rows=failed_rows,
        failed_vectors_after=failed_vectors_after,
        kmeans_after=rec_a["kmeans"],
        mapping=rec_a["mapping"],
        save_csv="failed_json_predicted_categories.csv",
        save_jsonl="failed_json_predicted_categories.jsonl",
    )

    # After LoRA 학습 데이터 plot 좌표 저장/재사용
    train_tsne_coords = compute_or_load_after_tsne(
        train_vectors_after,
        save_coords="after_lora_train_tsne_coords.npy",
    )

    # 저장된 After plot 좌표 위에 JSONL 질문 overlay
    failed_tsne_coords = project_failed_to_saved_tsne(
        train_vectors_after=train_vectors_after,
        train_tsne_coords=train_tsne_coords,
        failed_vectors_after=failed_vectors_after,
        k=OVERLAY_K_NEIGHBORS,
    )

    plot_after_tsne_with_failed_overlay(
        train_tsne_coords=train_tsne_coords,
        train_labels=train_labels,
        failed_tsne_coords=failed_tsne_coords,
        failed_pred_labels=failed_pred_labels,
        label_names=true_label_names,
        save_path="after_lora_train_tsne_plot_with_failed_json.png",
    )

    # JSONL 질문에 대해서만 LoRA OFF/ON 답변 생성
    save_failed_json_answers(
        model=model,
        tokenizer=tokenizer,
        predicted_rows=predicted_rows,
        max_new_tokens=MAX_NEW_TOKENS,
        save_txt="failed_json_lora_on_off_answers.txt",
        save_jsonl="failed_json_lora_on_off_answers.jsonl",
    )

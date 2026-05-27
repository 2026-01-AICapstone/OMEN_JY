"""
evaluate.py
===========
학습 후 latent space 평가.

핵심 변경 (이전 버전 대비):
  - NMI / ARI metric 추가 (pseudo-label과 true-label의 일치도)
  - Hungarian matching으로 pseudo-label ↔ true-label 정렬 후 accuracy 계산
  - t-SNE 시각화에서 true-label과 pseudo-label 둘 다 표시 (Before / After × true / pseudo)
  - clustering.py가 사용한 cluster_layer를 자동으로 동일 사용

기준 검증 흐름:
  1. 학습 데이터(DNA) hidden을 Before(LoRA off)와 After(LoRA on)에서 추출
  2. true_labels(risk_area)와 학습 시 사용된 pseudo_labels 모두 보유
  3. Before/After 각각에서:
     - Silhouette / Davies-Bouldin (true-label 기준 분리도)
     - 새 K-Means 돌려서 학습 후에도 클러스터가 자연 발생하는지 확인
     - NMI/ARI: 새 K-Means vs true-label
  4. t-SNE 시각화: true-label 색칠 (분리 정도 시각 확인)
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    normalized_mutual_info_score, adjusted_rand_score
)
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment

from dataset import RepBendingDataset
from sklearn.metrics import pairwise_distances

# =========================================================
# 설정
# =========================================================
BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"
#BASE_MODEL = "google/gemma-2-2b-it"
#BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH = "./out/test"
CENTROIDS_PATH = "./out/test/crush_centroids.pt"
INIT_CENTROIDS_PATH = "./out/test/crush_centroids_initial.pt"  # cluster_layer 정보 포함
EVAL_PROMPTS_TXT = "./eval_prompts.txt"

# "train_data" : 학습 데이터로 평가 (학습 효과 확인용)
# "new_data"   : eval_prompts.txt 기반 평가 (일반화 확인용)
EVAL_MODE = "train_data"

# 정답 라벨 의미 (DNA의 risk_area)
true_label_names = {
    0: "Malicious",
    1: "ChatbotHarm",
    2: "InfoHazard",
    3: "Misinfo",
    4: "Discrim",
}


# =========================================================
# eval_prompts.txt 로더
# =========================================================
def load_eval_prompts_from_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {txt_path}")

    prompts, labels = [], []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and "]" in line:
                bracket_end = line.index("]")
                try:
                    label = int(line[1:bracket_end])
                    prompt = line[bracket_end + 1:].strip()
                    if prompt:
                        prompts.append(prompt)
                        labels.append(label)
                except ValueError:
                    continue

    print(f"eval_prompts.txt 로드: {len(prompts)}개")
    bincount = np.bincount(labels, minlength=5)
    for i, cnt in enumerate(bincount):
        print(f"  [{i}] {true_label_names.get(i, f'Cat{i}')}: {cnt}개")
    return prompts, labels


# =========================================================
# 모델 로딩
# =========================================================
def load_crush_model_and_meta(base_model_name, lora_weights_path,
                              centroids_path, init_centroids_path):
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

    print("3. centroid 로드...")
    centroids = torch.load(centroids_path, map_location=model.device)

    # 초기 centroid + 메타데이터 (cluster_layer, pseudo_labels 등)
    cluster_layer = -1  # 기본값
    init_meta = None
    if os.path.exists(init_centroids_path):
        init_meta = torch.load(init_centroids_path, map_location="cpu")
        if isinstance(init_meta, dict) and "cluster_layer" in init_meta:
            cluster_layer = init_meta["cluster_layer"]
            print(f"   학습 시 사용된 cluster_layer: {cluster_layer}")
    else:
        print(f"   ⚠️ {init_centroids_path} 없음. cluster_layer=-1(마지막) 사용.")

    return model, tokenizer, centroids, cluster_layer, init_meta


# =========================================================
# 벡터 추출
# =========================================================
def extract_hidden_vectors(prompts, model, tokenizer, target_layer):
    vectors = []
    model.eval()
    for i, prompt in enumerate(prompts):
        if "<SEPARATOR>" in prompt:
            prompt = prompt.split("<SEPARATOR>")[0]

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[target_layer]
        vec = hidden[:, -1, :].float()
        vec = F.normalize(vec, p=2, dim=-1)  # 학습/clustering과 동일하게 L2 정규화

        vectors.append(vec[0].cpu().numpy())
        if (i + 1) % 50 == 0:
            print(f"  벡터 추출 {i + 1}/{len(prompts)}...")

    return np.array(vectors)


# =========================================================
# Hungarian matching: pseudo-label ↔ true-label
# =========================================================
def hungarian_matching_accuracy(true_labels, pred_labels, num_classes=5):
    """
    Pseudo-label은 0~K-1 임의 인덱스이므로 true-label과 직접 비교 불가.
    Hungarian algorithm으로 최적 permutation 찾아서 accuracy 계산.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    # confusion matrix (cost matrix: -count로 만들어 max-assignment 문제로)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    # linear_sum_assignment는 minimize, 우리는 매칭 수 maximize
    row_ind, col_ind = linear_sum_assignment(-cm)
    # row_ind[i]는 true-label i, col_ind[i]는 매칭된 pred-label
    mapping = dict(zip(col_ind.tolist(), row_ind.tolist()))  # pred → true
    matched = sum(cm[row_ind[i], col_ind[i]] for i in range(num_classes))
    accuracy = matched / len(true_labels)

    return accuracy, mapping, cm


# =========================================================
# 클러스터링 지표 (true-label 기준)
# =========================================================
def evaluate_separation(vectors, true_labels, label_names):
    """True-label로 그룹핑한 상태의 분리도."""
    unique = np.unique(true_labels)
    if len(unique) < 2:
        print("⚠️ 라벨 종류가 1개라 Silhouette 계산 불가")
        return None, None

    sil = silhouette_score(vectors, true_labels)
    db = davies_bouldin_score(vectors, true_labels)
    print(f"  Silhouette        : {sil:.4f}  (>0.3 양호, >0.5 우수)")
    print(f"  Davies-Bouldin    : {db:.4f}  (낮을수록 좋음)")

    # intra/inter 거리
    print(f"  --- 클러스터별 응집도 (intra) ---")
    for label in unique:
        mask = (np.asarray(true_labels) == label)
        cluster_vecs = vectors[mask]
        if len(cluster_vecs) > 1:
            dists = euclidean_distances(cluster_vecs)
            np.fill_diagonal(dists, np.nan)
            intra = np.nanmean(dists)
            name = label_names.get(int(label), f"C{label}")
            print(f"    [{label}] {name:<15}: {intra:.4f}")

    print(f"  --- 클러스터 간 분리도 (inter) ---")
    centers = np.array([vectors[np.asarray(true_labels) == l].mean(axis=0) for l in unique])
    inter = euclidean_distances(centers)
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            ni = label_names.get(int(unique[i]), f"C{unique[i]}")[:10]
            nj = label_names.get(int(unique[j]), f"C{unique[j]}")[:10]
            print(f"    {ni} <-> {nj}: {inter[i, j]:.4f}")

    return sil, db


def evaluate_clustering_recovery(vectors, true_labels, num_clusters=5):
    """
    학습된 latent에서 K-Means를 새로 돌려 → true-label과 얼마나 일치하는지.
    NMI/ARI/Hungarian accuracy 측정.
    """
    print(f"  새 K-Means(k={num_clusters}) 실행...")
    km = KMeans(n_clusters=num_clusters, n_init=10, random_state=42, max_iter=300)
    pred = km.fit_predict(vectors)

    nmi = normalized_mutual_info_score(true_labels, pred)
    ari = adjusted_rand_score(true_labels, pred)
    acc, mapping, cm = hungarian_matching_accuracy(true_labels, pred, num_classes=num_clusters)

    print(f"  NMI                : {nmi:.4f}  (1=완벽 일치, 0=무관)")
    print(f"  ARI                : {ari:.4f}  (1=완벽 일치, 0=random, <0=매우 다름)")
    print(f"  Hungarian Accuracy : {acc:.4f}  (최적 permutation 후 정확도)")

    # return {"nmi": nmi, "ari": ari, "hungarian_acc": acc, "pred": pred, "cm": cm}
    return {
        "nmi": nmi,
        "ari": ari,
        "hungarian_acc": acc,
        "pred": pred,
        "cm": cm,
        "mapping": mapping,
    }


# =========================================================
# t-SNE 시각화
# =========================================================
def visualize_tsne(vectors_before, vectors_after, true_labels, label_names,
                   save_path="crush_tsne.png"):
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("CRUSH: Latent Space (Before vs After LoRA, colored by TRUE label)",
                 fontsize=14, fontweight='bold')

    for ax, vectors, title in zip(axes, [vectors_before, vectors_after],
                                   ["Before LoRA", "After LoRA"]):
        perp = min(30, len(vectors) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
        embedded = tsne.fit_transform(vectors)

        for label_idx in np.unique(true_labels):
            mask = np.array(true_labels) == label_idx
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=colors[label_idx % len(colors)], alpha=0.7, s=60,
                label=label_names.get(int(label_idx), f"C{label_idx}")
            )
            cx = embedded[mask, 0].mean()
            cy = embedded[mask, 1].mean()
            ax.scatter(cx, cy, c='black', marker='*', s=300, zorder=5)
            ax.annotate(f'{label_idx}', (cx, cy), fontsize=9, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        ax.set_title(title, fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.show()
    print(f"t-SNE 저장: {save_path}")


# =========================================================
# 학습 데이터 로더 (true-label 사용)
# =========================================================
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

    eval_prompts, eval_labels = [], []
    # ★ 평가는 true-label(risk_area) 기준 균등 추출
    true_arr = np.array(train_dataset.data_unsafe_true_labels)
    for label_idx in range(5):
        indices = np.where(true_arr == label_idx)[0][:num_samples_per_class]
        for idx in indices:
            # request만 추출 (response 제외)
            prompt = train_dataset.data_unsafe_request_prompts[idx]
            eval_prompts.append(prompt)
            eval_labels.append(label_idx)

    print(f"평가 샘플 수: {len(eval_prompts)}개")
    bincount = np.bincount(eval_labels, minlength=5)
    for i, cnt in enumerate(bincount):
        print(f"  [{i}] {true_label_names.get(i, f'C{i}')}: {cnt}개")
    return eval_prompts, eval_labels


# 0519 추가
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
        before_answer = generate_answer(model, tokenizer, prompt, max_new_tokens)

    after_answer = generate_answer(model, tokenizer, prompt, max_new_tokens)

    return before_answer, after_answer


def print_mixed_and_correct_examples(
    model,
    tokenizer,
    prompts,
    true_labels,
    vectors_after,
    pred_clusters,
    mapping,
    label_names,
    mix_ratio=0.25,
    save_path="ours_llama3b.txt",
):
    """
    MIXED:
      - true != pred 가 아니라
      - nearest centroid와 second-nearest centroid 거리가 가까운 boundary/overlap 샘플

    CORRECT:
      - centroid margin이 큰, 클러스터 내부에 안정적으로 들어간 샘플
    """

    pred_clusters = np.asarray(pred_clusters)
    true_labels = np.asarray(true_labels)

    # KMeans centroid를 다시 계산
    num_clusters = len(set(pred_clusters.tolist()))
    centroids = []
    for c in range(num_clusters):
        mask = pred_clusters == c
        centroids.append(vectors_after[mask].mean(axis=0))
    centroids = np.stack(centroids, axis=0)

    # 각 샘플과 centroid 간 거리
    dists = pairwise_distances(vectors_after, centroids, metric="euclidean")

    # 가장 가까운 centroid, 두 번째로 가까운 centroid
    sorted_idx = np.argsort(dists, axis=1)
    nearest_c = sorted_idx[:, 0]
    second_c = sorted_idx[:, 1]

    nearest_d = dists[np.arange(len(dists)), nearest_c]
    second_d = dists[np.arange(len(dists)), second_c]

    # margin이 작을수록 애매하게 섞인 샘플
    margin = second_d - nearest_d

    # MIX 개수: 전체의 mix_ratio 비율
    n_mix = int(len(prompts) * mix_ratio)

    # margin 작은 순 = MIXED
    mixed_indices = np.argsort(margin)[:n_mix]

    # margin 큰 순 = CORRECT/STABLE
    correct_indices = np.argsort(margin)[n_mix:]

    mixed_set = set(mixed_indices.tolist())

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("LATENT BOUNDARY MIXED / STABLE CORRECT REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"TOTAL SAMPLES : {len(prompts)}\n")
        f.write(f"MIXED         : {len(mixed_indices)}\n")
        f.write(f"CORRECT/STABLE: {len(correct_indices)}\n")
        f.write(f"MIX CRITERION : smallest centroid margin, mix_ratio={mix_ratio}\n\n")

        f.write("LABEL MEANING\n")
        for k in sorted(label_names.keys()):
            f.write(f"  {k}: {label_names[k]}\n")
        f.write("\n")

        # =====================================================
        # MIXED: boundary 근처 샘플
        # =====================================================
        f.write("\n" + "#" * 100 + "\n")
        f.write("# MIXED QUESTIONS\n")
        f.write("# 기준: nearest centroid와 second-nearest centroid 거리가 가까운 샘플\n")
        f.write("# 의미: latent space에서 서로 다른 클러스터 사이에 애매하게 섞인 질문\n")
        f.write("#" * 100 + "\n\n")

        for rank, idx in enumerate(mixed_indices, start=1):
            prompt = prompts[idx]
            before_answer, after_answer = generate_answer_pair(model, tokenizer, prompt)

            true_l = int(true_labels[idx])
            pred_c = int(pred_clusters[idx])
            pred_l = int(mapping.get(pred_c, -1))

            nearest_cluster = int(nearest_c[idx])
            second_cluster = int(second_c[idx])

            nearest_label = int(mapping.get(nearest_cluster, -1))
            second_label = int(mapping.get(second_cluster, -1))

            f.write("=" * 100 + "\n")
            f.write(f"[MIXED #{rank}] sample_idx={idx}\n")
            f.write(f"TRUE_LABEL             : {true_l} / {label_names.get(true_l, f'C{true_l}')}\n")
            f.write(f"PRED_CLUSTER           : C{pred_c}\n")
            f.write(f"MAPPED_PRED_LABEL      : {pred_l} / {label_names.get(pred_l, f'C{pred_l}')}\n")
            f.write(f"NEAREST_CLUSTER        : C{nearest_cluster} -> {nearest_label} / {label_names.get(nearest_label, f'C{nearest_label}')}\n")
            f.write(f"SECOND_NEAREST_CLUSTER : C{second_cluster} -> {second_label} / {label_names.get(second_label, f'C{second_label}')}\n")
            f.write(f"NEAREST_DISTANCE       : {nearest_d[idx]:.6f}\n")
            f.write(f"SECOND_DISTANCE        : {second_d[idx]:.6f}\n")
            f.write(f"MARGIN                 : {margin[idx]:.6f}\n")
            f.write(
                "INTERPRETATION         : "
                "margin이 작아서 두 클러스터 사이 boundary/overlap 영역에 있는 질문.\n\n"
            )
            f.write("PROMPT:\n")
            f.write(prompt.strip() + "\n\n")
            f.write("BASE MODEL ANSWER (LoRA OFF):\n")
            f.write(before_answer.strip() + "\n\n")

            f.write("LORA MODEL ANSWER (LoRA ON):\n")
            f.write(after_answer.strip() + "\n\n")

        # =====================================================
        # CORRECT/STABLE: boundary에서 멀리 떨어진 샘플
        # =====================================================
        f.write("\n" + "#" * 100 + "\n")
        f.write("# CORRECT / STABLE QUESTIONS\n")
        f.write("# 기준: centroid margin이 상대적으로 큰 샘플\n")
        f.write("# 의미: 특정 클러스터 내부에 비교적 안정적으로 위치한 질문\n")
        f.write("#" * 100 + "\n\n")

        for rank, idx in enumerate(correct_indices, start=1):
            prompt = prompts[idx]
            before_answer, after_answer = generate_answer_pair(model, tokenizer, prompt)

            true_l = int(true_labels[idx])
            pred_c = int(pred_clusters[idx])
            pred_l = int(mapping.get(pred_c, -1))

            nearest_cluster = int(nearest_c[idx])
            second_cluster = int(second_c[idx])

            nearest_label = int(mapping.get(nearest_cluster, -1))
            second_label = int(mapping.get(second_cluster, -1))

            f.write("=" * 100 + "\n")
            f.write(f"[CORRECT/STABLE #{rank}] sample_idx={idx}\n")
            f.write(f"TRUE_LABEL             : {true_l} / {label_names.get(true_l, f'C{true_l}')}\n")
            f.write(f"PRED_CLUSTER           : C{pred_c}\n")
            f.write(f"MAPPED_PRED_LABEL      : {pred_l} / {label_names.get(pred_l, f'C{pred_l}')}\n")
            f.write(f"NEAREST_CLUSTER        : C{nearest_cluster} -> {nearest_label} / {label_names.get(nearest_label, f'C{nearest_label}')}\n")
            f.write(f"SECOND_NEAREST_CLUSTER : C{second_cluster} -> {second_label} / {label_names.get(second_label, f'C{second_label}')}\n")
            f.write(f"NEAREST_DISTANCE       : {nearest_d[idx]:.6f}\n")
            f.write(f"SECOND_DISTANCE        : {second_d[idx]:.6f}\n")
            f.write(f"MARGIN                 : {margin[idx]:.6f}\n")
            f.write(
                "INTERPRETATION         : "
                "margin이 커서 특정 클러스터 내부에 상대적으로 안정적으로 위치한 질문.\n\n"
            )
            f.write("PROMPT:\n")
            f.write(prompt.strip() + "\n\n")
            f.write("BASE MODEL ANSWER (LoRA OFF):\n")
            f.write(before_answer.strip() + "\n\n")

            f.write("LORA MODEL ANSWER (LoRA ON):\n")
            f.write(after_answer.strip() + "\n\n")

    print(f"저장 완료: {save_path}")
    print(f"TOTAL={len(prompts)}, MIXED={len(mixed_indices)}, CORRECT/STABLE={len(correct_indices)}")

# =========================================================
# 메인
# =========================================================
if __name__ == "__main__":

    # 1. 모델 + 메타 로딩
    model, tokenizer, centroids, cluster_layer, init_meta = load_crush_model_and_meta(
        BASE_MODEL, LORA_PATH, CENTROIDS_PATH, INIT_CENTROIDS_PATH
    )

    # 2. 평가 데이터
    if EVAL_MODE == "train_data":
        print("\n========== 학습 데이터 기반 평가 ==========")
        eval_prompts, eval_labels = load_train_dataset_samples(BASE_MODEL, num_samples_per_class=50)
    else:
        print(f"\n========== 새 데이터 기반 평가 ({EVAL_PROMPTS_TXT}) ==========")
        eval_prompts, eval_labels = load_eval_prompts_from_txt(EVAL_PROMPTS_TXT)

    # 3. Before / After 벡터 추출 (cluster_layer와 동일한 레이어에서)
    print(f"\n[Before LoRA] 벡터 추출 (layer={cluster_layer})...")
    with model.disable_adapter():
        vectors_before = extract_hidden_vectors(eval_prompts, model, tokenizer, target_layer=cluster_layer)

    print(f"[After LoRA] 벡터 추출 (layer={cluster_layer})...")
    vectors_after = extract_hidden_vectors(eval_prompts, model, tokenizer, target_layer=cluster_layer)

    # 4. 벡터 변화량
    diff = np.mean(np.linalg.norm(vectors_after - vectors_before, axis=1))
    print(f"\n평균 벡터 변화량 (Before→After): {diff:.4f}")

    # 5. True-label 기준 분리도
    print("\n[Before LoRA] True-label 기준 분리도")
    sil_b, db_b = evaluate_separation(vectors_before, eval_labels, true_label_names)
    print("\n[After LoRA] True-label 기준 분리도")
    sil_a, db_a = evaluate_separation(vectors_after, eval_labels, true_label_names)

    if sil_b is not None and sil_a is not None:
        print(f"\n========== Before vs After (True-label 기준) ==========")
        print(f"  Silhouette : {sil_b:.4f} → {sil_a:.4f}  "
              f"{'✅ 개선' if sil_a > sil_b else '❌ 악화'}")
        print(f"  D-B Index  : {db_b:.4f} → {db_a:.4f}  "
              f"{'✅ 개선' if db_a < db_b else '❌ 악화'}")

    # 6. ★ 핵심 검증: 학습 후 클러스터 자연 발생 (NMI/ARI/Hungarian)
    print("\n[Before LoRA] 클러스터 회복도 (새 K-Means vs True-label)")
    rec_b = evaluate_clustering_recovery(vectors_before, eval_labels, num_clusters=5)

    print("\n[After LoRA] 클러스터 회복도 (새 K-Means vs True-label)")
    rec_a = evaluate_clustering_recovery(vectors_after, eval_labels, num_clusters=5)

    print(f"\n========== 클러스터 회복도 변화 ==========")
    print(f"  NMI               : {rec_b['nmi']:.4f} → {rec_a['nmi']:.4f}  "
          f"{'✅ 개선' if rec_a['nmi'] > rec_b['nmi'] else '❌ 악화'}")
    print(f"  ARI               : {rec_b['ari']:.4f} → {rec_a['ari']:.4f}  "
          f"{'✅ 개선' if rec_a['ari'] > rec_b['ari'] else '❌ 악화'}")
    print(f"  Hungarian Accuracy: {rec_b['hungarian_acc']:.4f} → {rec_a['hungarian_acc']:.4f}  "
          f"{'✅ 개선' if rec_a['hungarian_acc'] > rec_b['hungarian_acc'] else '❌ 악화'}")
    print("  ※ NMI/ARI/Acc가 Before보다 After에서 높으면, 학습이 latent에 의미 카테고리 분리를 만든 것.")

    # 7. t-SNE 시각화
    mode_suffix = "train" if EVAL_MODE == "train_data" else "new"
    visualize_tsne(
        vectors_before, vectors_after, eval_labels, true_label_names,
        save_path=f"crush_tsne_before_after_{mode_suffix}.png"
    )

    # 8. Confusion matrix 출력 (After 기준)
    print("\n[After LoRA] Confusion Matrix (true × predicted-by-new-KMeans)")
    print("  (행: true label, 열: K-Means 클러스터 ID)")
    cm = rec_a["cm"]
    header = "       " + "  ".join([f"  C{j}" for j in range(cm.shape[1])])
    print(header)
    for i in range(cm.shape[0]):
        row = "  ".join([f"{cm[i, j]:4d}" for j in range(cm.shape[1])])
        name = true_label_names.get(i, f"C{i}")[:10]
        print(f"  {name:<10}  {row}")



    # 추가
    print_mixed_and_correct_examples(
        model=model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        true_labels=eval_labels,
        vectors_after=vectors_after,
        pred_clusters=rec_a["pred"],
        mapping=rec_a["mapping"],
        label_names=true_label_names,
        mix_ratio=0.25,
        save_path="ours_llama3b.txt",
    )
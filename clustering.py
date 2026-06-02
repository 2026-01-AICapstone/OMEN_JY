# """
# clustering.py
# =============
# 학습 시작 전에 base 모델로 DNA(unsafe) 프롬프트의 hidden state를 추출하고,
# 의미 공간에서 K-Means 클러스터링을 수행하여 pseudo-label과 centroid를 사전 계산.
#
# 핵심 설계:
#   - 클러스터링 공간: base 모델의 target_layers 중간 레이어 hidden state
#   - Hidden 추출 위치: request 마지막 토큰
#   - Centroid: 각 클러스터의 hidden 평균 (L2 정규화) → 학습 내내 고정
#   - WJ 데이터는 클러스터링 대상 아님 (DNA만 pull/push에 사용)
#
# 진단:
#   - 클러스터별 샘플 수 분포
#   - DNA의 risk_area 정답 라벨과의 NMI/ARI (사전 평가)
#   - 클러스터별 대표 프롬프트 5개 출력 (사람이 의미 검증)
# """
#
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
#
#
# def extract_request_hidden(model, tokenizer, prompts, target_layer, max_length=512, device=None):
#     """
#     각 프롬프트의 request 마지막 토큰 hidden state 추출.
#
#     Args:
#         model: base 모델 (LoRA 적용 전 또는 disable_adapter 상태)
#         tokenizer: 토크나이저
#         prompts: List[str], 클러스터링할 프롬프트 (request만, response 제외)
#         target_layer: int, hidden state를 뽑을 레이어 인덱스
#         max_length: 토큰 최대 길이
#         device: 모델 device
#
#     Returns:
#         np.ndarray of shape (N, hidden_size), 각 프롬프트의 hidden vector
#     """
#     model.eval()
#     if device is None:
#         device = next(model.parameters()).device
#
#     vectors = []
#     with torch.no_grad():
#         for i, prompt in enumerate(prompts):
#             inputs = tokenizer(
#                 prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=max_length,
#                 padding=False,
#             ).to(device)
#
#             outputs = model(**inputs, output_hidden_states=True)
#             # hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden)
#             hidden = outputs.hidden_states[target_layer]
#             # request 마지막 토큰 (padding 없음 → [-1])
#             vec = hidden[0, -1, :].float().cpu().numpy()
#             vectors.append(vec)
#
#             if (i + 1) % 100 == 0:
#                 print(f"    hidden 추출 {i + 1}/{len(prompts)}...")
#
#     return np.stack(vectors, axis=0)
#
#
# def run_kmeans_clustering(hidden_vectors, num_clusters=5, random_state=42):
#     """
#     Hidden 공간에서 K-Means 클러스터링.
#
#     Args:
#         hidden_vectors: (N, hidden_size) np.ndarray
#         num_clusters: 클러스터 수 (기본 5)
#         random_state: 재현성
#
#     Returns:
#         pseudo_labels: (N,) np.ndarray, 각 샘플의 클러스터 ID (0~K-1)
#         centroids_np: (K, hidden_size) np.ndarray, K-Means가 찾은 클러스터 중심
#     """
#     # n_init=10으로 여러 초기값 중 best 선택 (안정성)
#     kmeans = KMeans(
#         n_clusters=num_clusters,
#         n_init=10,
#         random_state=random_state,
#         max_iter=300,
#     )
#     pseudo_labels = kmeans.fit_predict(hidden_vectors)
#     centroids_np = kmeans.cluster_centers_
#     return pseudo_labels, centroids_np
#
#
# def compute_centroids_from_hidden(hidden_vectors, pseudo_labels, num_clusters):
#     """
#     각 클러스터에 속한 hidden vector들의 평균으로 centroid 계산 후 L2 정규화.
#
#     K-Means가 반환한 cluster_centers_를 그대로 쓰지 않는 이유:
#       - 학습/검증 시점의 hidden은 L2 정규화된 상태로 사용되므로,
#         centroid도 동일하게 정규화된 평균이어야 일관성이 맞음.
#
#     Returns:
#         centroids: (K, hidden_size) torch.Tensor, L2 정규화된 고정 centroid
#     """
#     centroids = []
#     for k in range(num_clusters):
#         mask = (pseudo_labels == k)
#         if mask.sum() == 0:
#             # 빈 클러스터 (거의 안 일어남, n_init=10이면)
#             print(f"  ⚠️ Cluster {k} is empty, using zero vector")
#             centroid = np.zeros(hidden_vectors.shape[1], dtype=np.float32)
#         else:
#             centroid = hidden_vectors[mask].mean(axis=0)
#         centroids.append(centroid)
#     centroids = np.stack(centroids, axis=0)
#     centroids_t = torch.from_numpy(centroids).float()
#     centroids_t = F.normalize(centroids_t, p=2, dim=-1)
#     return centroids_t
#
#
# def diagnose_clustering(pseudo_labels, true_labels, raw_prompts, num_clusters=5):
#     """
#     클러스터링 결과 진단:
#       1. 클러스터별 샘플 분포
#       2. true_labels(risk_area)와의 NMI/ARI (얼마나 일치하는지)
#       3. 클러스터별 대표 프롬프트 5개 출력 (사람이 의미 검증)
#     """
#     print("\n" + "=" * 70)
#     print("📊 클러스터링 진단")
#     print("=" * 70)
#
#     # 1. 클러스터별 분포
#     print("\n[1] 클러스터별 샘플 수")
#     bincount = np.bincount(pseudo_labels, minlength=num_clusters)
#     for k in range(num_clusters):
#         bar = "█" * int(bincount[k] / max(bincount.max(), 1) * 30)
#         print(f"  Cluster {k}: {bincount[k]:4d}개 {bar}")
#
#     # 분포 균형성 평가
#     if bincount.min() < bincount.max() * 0.1:
#         print("  ⚠️ 클러스터 분포가 매우 불균형. 가장 작은 클러스터가 의미 있는지 확인 필요.")
#
#     # 2. NMI/ARI (true_labels가 있을 때만)
#     if true_labels is not None and len(true_labels) == len(pseudo_labels):
#         true_labels_arr = np.asarray(true_labels)
#         nmi = normalized_mutual_info_score(true_labels_arr, pseudo_labels)
#         ari = adjusted_rand_score(true_labels_arr, pseudo_labels)
#         print(f"\n[2] 정답 라벨(risk_area)과의 일치도")
#         print(f"  NMI: {nmi:.4f}  (1에 가까울수록 일치, 0이면 무관)")
#         print(f"  ARI: {ari:.4f}  (1에 가까울수록 일치, 0이면 random, 음수면 매우 다름)")
#         if nmi < 0.1:
#             print("  ⚠️ NMI가 매우 낮음. K-Means가 위해 카테고리와 다른 축으로 나눴을 가능성.")
#             print("     → 클러스터별 대표 프롬프트를 확인해서 어떤 축으로 나뉘었는지 파악 권장.")
#         elif nmi > 0.3:
#             print("  ✅ NMI가 어느 정도 의미 있음. 클러스터링이 위해 카테고리를 부분적으로 잡음.")
#
#         # confusion matrix-like 표시
#         print(f"\n  [정답 라벨 × pseudo-label 분포]")
#         from collections import Counter
#         for true_l in sorted(set(true_labels_arr.tolist())):
#             mask = (true_labels_arr == true_l)
#             cluster_dist = Counter(pseudo_labels[mask].tolist())
#             row = " ".join([f"C{k}:{cluster_dist.get(k, 0):3d}" for k in range(num_clusters)])
#             print(f"    True {true_l}: {row}  (총 {mask.sum()}개)")
#
#     # 3. 클러스터별 대표 프롬프트 (사람이 의미 검증)
#     print(f"\n[3] 클러스터별 대표 프롬프트 (각 5개씩)")
#     for k in range(num_clusters):
#         idxs = np.where(pseudo_labels == k)[0][:5]
#         print(f"\n  Cluster {k}:")
#         for i, idx in enumerate(idxs):
#             prompt = raw_prompts[idx]
#             # 너무 길면 줄임
#             display = prompt[:120].replace("\n", " ")
#             if len(prompt) > 120:
#                 display += "..."
#             print(f"    [{i + 1}] {display}")
#
#     print("\n" + "=" * 70)
#
#
# def cluster_unsafe_prompts(
#         base_model,
#         tokenizer,
#         raw_unsafe_prompts,
#         true_labels,
#         target_layer,
#         num_clusters=5,
#         cache_path=None,
#         random_state=42,
# ):
#     """
#     전체 클러스터링 파이프라인.
#
#     Args:
#         base_model: base 모델 (PeftModel 가능, disable_adapter 컨텍스트 안에서 호출 권장)
#         tokenizer: 토크나이저
#         raw_unsafe_prompts: List[str], 클러스터링할 프롬프트 (request만)
#         true_labels: List[int] or None, 검증용 정답 라벨 (학습에는 사용 안 함)
#         target_layer: int, hidden 추출 레이어
#         num_clusters: K
#         cache_path: str or None, hidden 추출 결과 캐싱 경로
#         random_state: 재현성
#
#     Returns:
#         pseudo_labels: List[int], 학습용 라벨
#         centroids: torch.Tensor (K, hidden_size), L2 정규화된 고정 centroid
#     """
#     # ---------- 1. Hidden 추출 (캐싱 가능) ----------
#     if cache_path and os.path.exists(cache_path):
#         print(f"📦 캐시에서 hidden 로드: {cache_path}")
#         cache = np.load(cache_path)
#         hidden_vectors = cache["hidden"]
#         cached_prompts_count = int(cache["count"])
#         if cached_prompts_count != len(raw_unsafe_prompts):
#             print(f"  ⚠️ 캐시 샘플 수({cached_prompts_count}) ≠ 현재({len(raw_unsafe_prompts)}), 재추출")
#             hidden_vectors = None
#         else:
#             print(f"  ✅ {hidden_vectors.shape[0]}개 hidden 로드 완료")
#     else:
#         hidden_vectors = None
#
#     if hidden_vectors is None:
#         print(f"\n🔬 Base 모델로 hidden 추출 시작 (layer={target_layer}, N={len(raw_unsafe_prompts)})")
#         hidden_vectors = extract_request_hidden(
#             base_model, tokenizer, raw_unsafe_prompts, target_layer=target_layer
#         )
#         print(f"  ✅ hidden shape: {hidden_vectors.shape}")
#         if cache_path:
#             os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
#             np.savez(cache_path, hidden=hidden_vectors, count=len(raw_unsafe_prompts))
#             print(f"  💾 캐시 저장: {cache_path}")
#
#     # ---------- 2. K-Means ----------
#     print(f"\n🌀 K-Means 클러스터링 (k={num_clusters})")
#     pseudo_labels, _ = run_kmeans_clustering(
#         hidden_vectors, num_clusters=num_clusters, random_state=random_state
#     )
#
#     # ---------- 3. Centroid 계산 (L2 정규화) ----------
#     centroids = compute_centroids_from_hidden(
#         hidden_vectors, pseudo_labels, num_clusters
#     )
#     print(f"  ✅ centroid shape: {centroids.shape} (L2 정규화 완료)")
#
#     # centroid 간 평균 거리 출력 (분리도 확인)
#     with torch.no_grad():
#         cdist = torch.cdist(centroids, centroids)
#         # 자기 자신 제외 평균
#         mask = ~torch.eye(num_clusters, dtype=torch.bool)
#         mean_inter = cdist[mask].mean().item()
#         min_inter = cdist[mask].min().item()
#     print(f"  centroid 간 평균 거리: {mean_inter:.4f}, 최소 거리: {min_inter:.4f}")
#     print(f"  (단위 구 위 K개 점의 이론적 평균 거리는 약 {(2 * num_clusters / (num_clusters - 1)) ** 0.5:.4f})")
#
#     # ---------- 4. 진단 ----------
#     diagnose_clustering(pseudo_labels, true_labels, raw_unsafe_prompts, num_clusters)
#
#     return pseudo_labels.tolist(), centroids


"""
clustering.py
=============
학습 시작 전에 base 모델로 DNA(unsafe) 프롬프트의 hidden state를 추출하고,
의미 공간에서 K-Means 클러스터링을 수행하여 pseudo-label과 centroid를 사전 계산.

핵심 설계:
  - 클러스터링 공간: base 모델의 target_layers 중간 레이어 hidden state
  - Hidden 추출 위치: request 마지막 토큰
  - Centroid: 각 클러스터의 hidden 평균 (L2 정규화) → 학습 내내 고정
  - WJ 데이터는 클러스터링 대상 아님 (DNA만 pull/push에 사용)

진단:
  - 클러스터별 샘플 수 분포
  - DNA의 risk_area 정답 라벨과의 NMI/ARI (사전 평가)
  - 클러스터별 대표 프롬프트 5개 출력 (사람이 의미 검증)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def extract_request_hidden(model, tokenizer, prompts, target_layer, max_length=512, device=None):
    """
    각 프롬프트의 request 마지막 토큰 hidden state 추출.

    Args:
        model: base 모델 (LoRA 적용 전 또는 disable_adapter 상태)
        tokenizer: 토크나이저
        prompts: List[str], 클러스터링할 프롬프트 (request만, response 제외)
        target_layer: int, hidden state를 뽑을 레이어 인덱스
        max_length: 토큰 최대 길이
        device: 모델 device

    Returns:
        np.ndarray of shape (N, hidden_size), 각 프롬프트의 hidden vector
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    vectors = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, hidden)
            hidden = outputs.hidden_states[target_layer]
            # request 마지막 토큰 (padding 없음 → [-1])
            vec = hidden[0, -1, :].float().cpu().numpy()
            vectors.append(vec)

            if (i + 1) % 100 == 0:
                print(f"    hidden 추출 {i + 1}/{len(prompts)}...")

    return np.stack(vectors, axis=0)


def run_kmeans_clustering(hidden_vectors, num_clusters=5, random_state=42):
    """
    Hidden 공간에서 K-Means 클러스터링.

    Args:
        hidden_vectors: (N, hidden_size) np.ndarray
        num_clusters: 클러스터 수 (기본 5)
        random_state: 재현성

    Returns:
        pseudo_labels: (N,) np.ndarray, 각 샘플의 클러스터 ID (0~K-1)
        centroids_np: (K, hidden_size) np.ndarray, K-Means가 찾은 클러스터 중심
    """
    # n_init=10으로 여러 초기값 중 best 선택 (안정성)
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=random_state,
        max_iter=300,
    )
    pseudo_labels = kmeans.fit_predict(hidden_vectors)
    centroids_np = kmeans.cluster_centers_
    return pseudo_labels, centroids_np


def compute_centroids_from_hidden(hidden_vectors, pseudo_labels, num_clusters):
    """
    각 클러스터에 속한 hidden vector들의 평균으로 centroid 계산 후 L2 정규화.

    K-Means가 반환한 cluster_centers_를 그대로 쓰지 않는 이유:
      - 학습/검증 시점의 hidden은 L2 정규화된 상태로 사용되므로,
        centroid도 동일하게 정규화된 평균이어야 일관성이 맞음.

    Returns:
        centroids: (K, hidden_size) torch.Tensor, L2 정규화된 고정 centroid
    """
    centroids = []
    for k in range(num_clusters):
        mask = (pseudo_labels == k)
        if mask.sum() == 0:
            # 빈 클러스터 (거의 안 일어남, n_init=10이면)
            print(f"  ⚠️ Cluster {k} is empty, using zero vector")
            centroid = np.zeros(hidden_vectors.shape[1], dtype=np.float32)
        else:
            centroid = hidden_vectors[mask].mean(axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids, axis=0)
    centroids_t = torch.from_numpy(centroids).float()
    centroids_t = F.normalize(centroids_t, p=2, dim=-1)
    return centroids_t


def diagnose_clustering(pseudo_labels, true_labels, raw_prompts, num_clusters=5):
    """
    클러스터링 결과 진단:
      1. 클러스터별 샘플 분포
      2. true_labels(risk_area)와의 NMI/ARI (얼마나 일치하는지)
      3. 클러스터별 대표 프롬프트 5개 출력 (사람이 의미 검증)
    """
    print("\n" + "=" * 70)
    print("📊 클러스터링 진단")
    print("=" * 70)

    # 1. 클러스터별 분포
    print("\n[1] 클러스터별 샘플 수")
    bincount = np.bincount(pseudo_labels, minlength=num_clusters)
    for k in range(num_clusters):
        bar = "█" * int(bincount[k] / max(bincount.max(), 1) * 30)
        print(f"  Cluster {k}: {bincount[k]:4d}개 {bar}")

    # 분포 균형성 평가
    if bincount.min() < bincount.max() * 0.1:
        print("  ⚠️ 클러스터 분포가 매우 불균형. 가장 작은 클러스터가 의미 있는지 확인 필요.")

    # 2. NMI/ARI (true_labels가 있을 때만)
    if true_labels is not None and len(true_labels) == len(pseudo_labels):
        true_labels_arr = np.asarray(true_labels)
        nmi = normalized_mutual_info_score(true_labels_arr, pseudo_labels)
        ari = adjusted_rand_score(true_labels_arr, pseudo_labels)
        print(f"\n[2] 정답 라벨(risk_area)과의 일치도")
        print(f"  NMI: {nmi:.4f}  (1에 가까울수록 일치, 0이면 무관)")
        print(f"  ARI: {ari:.4f}  (1에 가까울수록 일치, 0이면 random, 음수면 매우 다름)")
        if nmi < 0.1:
            print("  ⚠️ NMI가 매우 낮음. K-Means가 위해 카테고리와 다른 축으로 나눴을 가능성.")
            print("     → 클러스터별 대표 프롬프트를 확인해서 어떤 축으로 나뉘었는지 파악 권장.")
        elif nmi > 0.3:
            print("  ✅ NMI가 어느 정도 의미 있음. 클러스터링이 위해 카테고리를 부분적으로 잡음.")

        # confusion matrix-like 표시
        print(f"\n  [정답 라벨 × pseudo-label 분포]")
        from collections import Counter
        for true_l in sorted(set(true_labels_arr.tolist())):
            mask = (true_labels_arr == true_l)
            cluster_dist = Counter(pseudo_labels[mask].tolist())
            row = " ".join([f"C{k}:{cluster_dist.get(k, 0):3d}" for k in range(num_clusters)])
            print(f"    True {true_l}: {row}  (총 {mask.sum()}개)")

    # 3. 클러스터별 대표 프롬프트 (사람이 의미 검증)
    print(f"\n[3] 클러스터별 대표 프롬프트 (각 5개씩)")
    for k in range(num_clusters):
        idxs = np.where(pseudo_labels == k)[0][:5]
        print(f"\n  Cluster {k}:")
        for i, idx in enumerate(idxs):
            prompt = raw_prompts[idx]
            # 너무 길면 줄임
            display = prompt[:120].replace("\n", " ")
            if len(prompt) > 120:
                display += "..."
            print(f"    [{i + 1}] {display}")

    print("\n" + "=" * 70)


def cluster_unsafe_prompts(
        base_model,
        tokenizer,
        raw_unsafe_prompts,
        true_labels,
        target_layer,
        num_clusters=5,
        cache_path=None,
        random_state=42,
):
    """
    전체 클러스터링 파이프라인.

    Args:
        base_model: base 모델 (PeftModel 가능, disable_adapter 컨텍스트 안에서 호출 권장)
        tokenizer: 토크나이저
        raw_unsafe_prompts: List[str], 클러스터링할 프롬프트 (request만)
        true_labels: List[int] or None, 검증용 정답 라벨 (학습에는 사용 안 함)
        target_layer: int, hidden 추출 레이어
        num_clusters: K
        cache_path: str or None, hidden 추출 결과 캐싱 경로
        random_state: 재현성

    Returns:
        pseudo_labels: List[int], 학습용 라벨
        centroids: torch.Tensor (K, hidden_size), L2 정규화된 고정 centroid
    """
    # ---------- 1. Hidden 추출 (캐싱 가능) ----------
    if cache_path and os.path.exists(cache_path):
        print(f"📦 캐시에서 hidden 로드: {cache_path}")
        cache = np.load(cache_path)
        hidden_vectors = cache["hidden"]
        cached_prompts_count = int(cache["count"])
        if cached_prompts_count != len(raw_unsafe_prompts):
            print(f"  ⚠️ 캐시 샘플 수({cached_prompts_count}) ≠ 현재({len(raw_unsafe_prompts)}), 재추출")
            hidden_vectors = None
        else:
            print(f"  ✅ {hidden_vectors.shape[0]}개 hidden 로드 완료")
    else:
        hidden_vectors = None

    if hidden_vectors is None:
        print(f"\n🔬 Base 모델로 hidden 추출 시작 (layer={target_layer}, N={len(raw_unsafe_prompts)})")
        hidden_vectors = extract_request_hidden(
            base_model, tokenizer, raw_unsafe_prompts, target_layer=target_layer
        )
        print(f"  ✅ hidden shape: {hidden_vectors.shape}")
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez(cache_path, hidden=hidden_vectors, count=len(raw_unsafe_prompts))
            print(f"  💾 캐시 저장: {cache_path}")

    # ---------- 2. K-Means ----------
    print(f"\n🌀 K-Means 클러스터링 (k={num_clusters})")
    pseudo_labels, _ = run_kmeans_clustering(
        hidden_vectors, num_clusters=num_clusters, random_state=random_state
    )

    # ---------- 3. Centroid 계산 (L2 정규화) ----------
    centroids = compute_centroids_from_hidden(
        hidden_vectors, pseudo_labels, num_clusters
    )
    print(f"  ✅ centroid shape: {centroids.shape} (L2 정규화 완료)")

    # centroid 간 평균 거리 출력 (분리도 확인)
    with torch.no_grad():
        cdist = torch.cdist(centroids, centroids)
        # 자기 자신 제외 평균
        mask = ~torch.eye(num_clusters, dtype=torch.bool)
        mean_inter = cdist[mask].mean().item()
        min_inter = cdist[mask].min().item()
    print(f"  centroid 간 평균 거리: {mean_inter:.4f}, 최소 거리: {min_inter:.4f}")
    print(f"  (단위 구 위 K개 점의 이론적 평균 거리는 약 {(2 * num_clusters / (num_clusters - 1)) ** 0.5:.4f})")

    # ---------- 4. 진단 ----------
    diagnose_clustering(pseudo_labels, true_labels, raw_unsafe_prompts, num_clusters)

    return pseudo_labels.tolist(), centroids

def extract_all_layers_hidden_and_cluster(
        model,
        tokenizer,
        prompts,
        target_layers,
        pseudo_labels,
        true_labels,
        output_dir,
        tag="train",
        max_length=512,
):
    """
    target_layers 각각에 대해 hidden vector를 추출하고,
    t-SNE 좌표 계산에 필요한 raw hidden + 클러스터/라벨 정보를 레이어별로 저장.

    저장 파일 (output_dir/layer_analysis/ 아래):
      hidden_layer{L}_{tag}.npy       : (N, hidden_size) raw hidden vectors
      cluster_info_{tag}.npz          : pseudo_labels, true_labels, layer 목록

    Args:
        model        : base 모델 (LoRA 적용 전 또는 disable_adapter 상태 권장)
        tokenizer    : 토크나이저
        prompts      : List[str], 분석할 프롬프트 목록
        target_layers: List[int], 저장할 레이어 인덱스 목록
        pseudo_labels: List[int], K-Means pseudo-label (cluster id)
        true_labels  : List[int] or None, 정답 라벨
        output_dir   : str, 출력 루트 디렉토리
        tag          : str, 파일명 구분자 (기본 "train")
        max_length   : int, 토큰 최대 길이
    """
    import os, numpy as np, torch, torch.nn.functional as F

    save_dir = os.path.join(output_dir, "layer_analysis")
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    N = len(prompts)

    # 레이어별 버퍼: {layer_idx: list of np.ndarray}
    layer_buffers = {l: [] for l in target_layers}

    print(f"\n📐 전체 레이어 hidden 추출 시작 (layers={target_layers}, N={N})...")
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            # hidden_states[0] = embedding, hidden_states[l] = l번째 레이어 출력
            for l in target_layers:
                hidden = outputs.hidden_states[l]   # (1, seq_len, hidden)
                vec    = hidden[0, -1, :].float()   # 마지막 토큰
                vec    = F.normalize(vec, p=2, dim=-1)
                layer_buffers[l].append(vec.cpu().numpy())

            if (i + 1) % 100 == 0:
                print(f"  추출 {i+1}/{N}...")

    # 레이어별 npy 저장
    for l in target_layers:
        arr  = np.stack(layer_buffers[l], axis=0)   # (N, hidden_size)
        path = os.path.join(save_dir, f"hidden_layer{l}_{tag}.npy")
        np.save(path, arr)
        print(f"  💾 layer {l:3d} hidden 저장: {path}  shape={arr.shape}")

    # 클러스터/라벨 정보 저장 (레이어 공통)
    cluster_info_path = os.path.join(save_dir, f"cluster_info_{tag}.npz")
    save_kwargs = dict(
        pseudo_labels = np.array(pseudo_labels),
        target_layers = np.array(target_layers),
    )
    if true_labels is not None:
        save_kwargs["true_labels"] = np.array(true_labels)
    np.savez(cluster_info_path, **save_kwargs)
    print(f"  💾 클러스터 정보 저장: {cluster_info_path}")
    print(f"     pseudo_labels shape : {np.array(pseudo_labels).shape}")
    if true_labels is not None:
        print(f"     true_labels shape   : {np.array(true_labels).shape}")
    print(f"  ✅ 전체 레이어 분석 저장 완료 → {save_dir}")
    return save_dir
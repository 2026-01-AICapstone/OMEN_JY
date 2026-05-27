# import torch
# import torch.nn.functional as F
#
#
# class CentroidManager:
#     def __init__(self, num_classes, hidden_size, momentum=0.9):
#         self.num_classes = num_classes
#         self.momentum = momentum
#         # CPU에 두고 grad 완전 차단
#         self.centroids = torch.zeros(num_classes, hidden_size)
#
#     def update_centroids(self, features, labels):
#         # grad graph에서 완전히 분리
#         features_detached = features.detach().cpu().float()
#         labels_cpu = labels.detach().cpu()
#
#         unique_labels = torch.unique(labels_cpu)
#         for label in unique_labels:
#             class_features = features_detached[labels_cpu == label]
#             batch_centroid = class_features.mean(dim=0)
#
#             if torch.sum(self.centroids[label]) == 0:
#                 self.centroids[label] = batch_centroid
#             else:
#                 self.centroids[label] = (
#                     self.momentum * self.centroids[label] +
#                     (1 - self.momentum) * batch_centroid
#                 )
#
#         # loss 계산용으로 넘길 때만 GPU로, grad 없이
#         return self.centroids.clone().to(features.device)
#
#
# def calc_crush_loss(h_benign_orig, h_benign_new, h_harmful_orig, h_harmful_new, labels, centroids, m_b=1.0, m_pull=1.0,
#                     m_push=1.0):
#     batch_size_b = h_benign_new.size(0)
#     batch_size_h = h_harmful_new.size(0)
#     num_classes = centroids.size(0)
#
#     loss_benign = torch.tensor(0.0, device=h_benign_new.device if batch_size_b > 0 else h_harmful_new.device)
#     if batch_size_b > 0:
#         dist_orig_new_b = F.pairwise_distance(h_benign_orig, h_benign_new)
#         dists_to_centroids = torch.cdist(h_benign_new.unsqueeze(1), centroids.unsqueeze(0)).squeeze(1)
#         min_dists, _ = torch.min(dists_to_centroids, dim=1)
#         #loss_benign_batch = F.relu(dist_orig_new_b - min_dists + m_b)
#         # 수정: safe 벡터가 모든 유해 중심점과 마진 이상 거리 유지
#         loss_benign_batch = F.relu(m_b - min_dists)  # 유해 클러스터와 m_b 이상 거리
#         loss_benign = loss_benign_batch.mean()
#
#     loss_pull = torch.tensor(0.0, device=h_harmful_new.device)
#     loss_push = torch.tensor(0.0, device=h_harmful_new.device)
#
#     if batch_size_h > 0:
#         target_centroids = centroids[labels]
#         dist_to_my_centroid = F.pairwise_distance(h_harmful_new, target_centroids)
#         dist_to_orig_h = F.pairwise_distance(h_harmful_new, h_harmful_orig)
#
#         loss_pull_batch = F.relu(dist_to_my_centroid - dist_to_orig_h + m_pull)
#         loss_pull = loss_pull_batch.mean()
#
#         dists_to_all_centroids = torch.cdist(h_harmful_new, centroids)
#
#         # ✅ inplace 대신 large_value로 마스킹 (gradient 안전)
#         mask = F.one_hot(labels, num_classes=num_classes).bool()
#         large_value = torch.full_like(dists_to_all_centroids, float('inf'))
#         dists_to_all_centroids = torch.where(mask, large_value, dists_to_all_centroids)
#
#         min_dists_other, _ = torch.min(dists_to_all_centroids, dim=1)
#         loss_push_batch = F.relu(dist_to_my_centroid - min_dists_other + m_push)
#         loss_push = loss_push_batch.mean()
#
#     return loss_benign, loss_pull, loss_push

"""
crush_loss.py
=============
CRUSH Loss (Cluster RUSH).

핵심 설계 변경 (이전 버전 대비):
  - EMA 기반 CentroidManager 폐기
  - Centroid는 학습 시작 전 1회 사전 계산 후 학습 내내 고정
  - FixedCentroidHolder는 단순히 텐서를 보관/이동만 담당
  - Margin 0.2 → 0.5 (단위 구 위 5개 점의 평균 거리가 ~1.58인 점 고려)

Loss 구성:
  - Benign loss : safe 벡터가 원본보다 유해 클러스터에 가까워지면 패널티
  - Pull loss   : harmful 벡터가 자기 클러스터 centroid에 모이도록
  - Push loss   : harmful 벡터가 다른 클러스터 centroid에서 멀어지도록

연구 근거:
  - Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)
  - Movshovitz-Attias et al., "Proxy NCA" (ICCV 2017)
  - Kim et al., "Proxy Anchor Loss" (CVPR 2020)
  → centroid/proxy를 hidden 평균 EMA가 아닌 사전 계산 또는 학습 가능 파라미터로 두는 것이 표준
"""

import torch
import torch.nn.functional as F


class FixedCentroidHolder:
    """
    학습 시작 전 사전 계산된 centroid를 보관/이동만 담당.
    EMA 업데이트 없음. 학습 내내 값 불변.
    """

    def __init__(self, centroids: torch.Tensor):
        """
        Args:
            centroids: (K, hidden_size) torch.Tensor, 이미 L2 정규화된 상태로 들어와야 함
        """
        assert centroids.dim() == 2, f"centroids must be 2D, got {centroids.shape}"
        self.num_classes = centroids.size(0)
        self.hidden_size = centroids.size(1)
        # detach + clone으로 grad graph 완전 차단
        self.centroids = centroids.detach().clone()

        # 정규화 검증
        norms = self.centroids.norm(p=2, dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            print(f"⚠️ centroids가 L2 정규화되지 않음 (norms={norms.tolist()}), 자동 정규화 적용")
            self.centroids = F.normalize(self.centroids, p=2, dim=-1)

        print(f"✅ FixedCentroidHolder 초기화: {self.num_classes}개 centroid (hidden={self.hidden_size})")

    def get(self, device=None, dtype=None):
        """학습 시 호출. 디바이스/dtype 맞춰서 반환."""
        c = self.centroids
        if device is not None and c.device != device:
            c = c.to(device)
            self.centroids = c  # 이후 호출에서 재이동 방지
        if dtype is not None and c.dtype != dtype:
            c = c.to(dtype)
        return c

    def to(self, device):
        self.centroids = self.centroids.to(device)
        return self

    def state_dict(self):
        return {"centroids": self.centroids}


def calc_crush_loss(h_benign_orig, h_benign_new,
                    h_harmful_orig, h_harmful_new,
                    labels, centroids,
                    m_b=0.5, m_pull=0.5, m_push=2.0):
    """
    CRUSH Loss 계산.

    Args:
        h_benign_orig: (B_safe, H) 원본 safe hidden (no LoRA)
        h_benign_new:  (B_safe, H) LoRA 적용 후 safe hidden
        h_harmful_orig: (B_harm, H) 원본 harmful hidden (no LoRA)
        h_harmful_new:  (B_harm, H) LoRA 적용 후 harmful hidden
        labels: (B_harm,) 각 harmful 샘플의 클러스터 ID (pseudo-label)
        centroids: (K, H) 고정 centroid (L2 정규화 상태)
        m_b, m_pull, m_push: 각 loss의 margin

    Returns:
        loss_benign, loss_pull, loss_push: 각각 scalar tensor
    """
    # ---------- L2 정규화 (단위 구 위에서 거리 계산) ----------
    if h_benign_orig.size(0) > 0:
        h_benign_orig = F.normalize(h_benign_orig.float(), p=2, dim=-1)
        h_benign_new = F.normalize(h_benign_new.float(), p=2, dim=-1)

    if h_harmful_orig.size(0) > 0:
        h_harmful_orig = F.normalize(h_harmful_orig.float(), p=2, dim=-1)
        h_harmful_new = F.normalize(h_harmful_new.float(), p=2, dim=-1)

    centroids = centroids.float()  # 이미 정규화 상태로 들어옴

    batch_size_b = h_benign_new.size(0)
    batch_size_h = h_harmful_new.size(0)
    num_classes = centroids.size(0)

    device = h_benign_new.device if batch_size_b > 0 else h_harmful_new.device

    # =========================================================
    # Benign Loss
    # safe 벡터가 원본보다 가장 가까운 유해 클러스터에 더 가까워지면 패널티
    # =========================================================
    loss_benign = torch.tensor(0.0, device=device)
    if batch_size_b > 0:
        dist_orig_new_b = F.pairwise_distance(h_benign_orig, h_benign_new)

        dists_to_centroids = torch.cdist(
            h_benign_new.unsqueeze(1),
            centroids.unsqueeze(0)
        ).squeeze(1)  # (B_safe, K)

        min_dists, _ = torch.min(dists_to_centroids, dim=1)
        loss_benign_batch = F.relu(dist_orig_new_b - min_dists + m_b)
        loss_benign = loss_benign_batch.mean()

    # =========================================================
    # Pull Loss / Push Loss
    # =========================================================
    loss_pull = torch.tensor(0.0, device=device)
    loss_push = torch.tensor(0.0, device=device)

    if batch_size_h > 0:
        target_centroids = centroids[labels]  # (B_harm, H)

        dist_to_my_centroid = F.pairwise_distance(h_harmful_new, target_centroids)
        dist_to_orig_h = F.pairwise_distance(h_harmful_new, h_harmful_orig)

        # Pull: 자기 centroid까지 거리가 원본까지 거리보다 m_pull 이상 가까워야 함
        loss_pull_batch = F.relu(dist_to_my_centroid - dist_to_orig_h + m_pull)
        loss_pull = loss_pull_batch.mean()

        # Push: 자기 centroid보다 가장 가까운 다른 centroid가 m_push 이상 멀어야 함
        dists_to_all = torch.cdist(h_harmful_new, centroids)  # (B_harm, K)

        # 자기 클러스터 마스킹 (gradient-safe out-of-place)
        mask = F.one_hot(labels, num_classes=num_classes).bool()
        large_value = torch.full_like(dists_to_all, float('inf'))
        dists_to_all_masked = torch.where(mask, large_value, dists_to_all)

        min_dists_other, _ = torch.min(dists_to_all_masked, dim=1)
        loss_push_batch = F.relu(dist_to_my_centroid - min_dists_other + m_push)
        loss_push = loss_push_batch.mean()

    return loss_benign, loss_pull, loss_push

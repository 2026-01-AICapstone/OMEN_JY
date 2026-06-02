"""
plot_layer_tsne.py
==================
layer_analysis/ 폴더에 저장된 레이어별 hidden vector를 읽어
각 레이어마다 LoRA OFF / LoRA ON t-SNE를 나란히 비교하는 플롯을 생성.

출력:
  {OUTPUT_DIR}/tsne_layer{L}_lora_off_vs_on.png  (레이어마다 1개)
  {OUTPUT_DIR}/tsne_all_layers_summary.png        (전체 레이어 한눈에 보기)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# =========================================================
# 설정
# =========================================================
LAYER_ANALYSIS_DIR = "./out/test/layer_analysis"   # train.py 출력 폴더
OUTPUT_DIR         = "./out/test/layer_analysis/tsne_plots"
TAG_OFF = "lora_off"
TAG_ON  = "lora_on"

# 라벨 이름 / 색상
LABEL_NAMES = {
    0: "Malicious",
    1: "ChatbotHarm",
    2: "InfoHazard",
    3: "Misinfo",
    4: "Discrim",
}
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

TSNE_PERPLEXITY  = 30
TSNE_RANDOM_STATE = 42


# =========================================================
# 유틸
# =========================================================
def load_hidden(layer_analysis_dir, layer, tag):
    path = os.path.join(layer_analysis_dir, f"hidden_layer{layer}_{tag}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"hidden 파일 없음: {path}")
    return np.load(path)


def load_cluster_info(layer_analysis_dir, tag):
    path = os.path.join(layer_analysis_dir, f"cluster_info_{tag}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"cluster_info 없음: {path}")
    data = np.load(path)
    pseudo_labels = data["pseudo_labels"]
    true_labels   = data["true_labels"] if "true_labels" in data else None
    target_layers = data["target_layers"].tolist()
    return pseudo_labels, true_labels, target_layers


def compute_tsne(hidden, perplexity=30, random_state=42):
    perp = min(perplexity, hidden.shape[0] - 1)
    return TSNE(n_components=2, random_state=random_state,
                perplexity=perp).fit_transform(hidden)


def draw_scatter(ax, coords, labels, title, label_names, colors, use_true=True):
    """단일 subplot에 scatter 그리기."""
    labels = np.asarray(labels)
    for label_idx in sorted(np.unique(labels)):
        if label_idx < 0:
            continue
        mask = labels == label_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[int(label_idx) % len(colors)],
            alpha=0.65, s=50, marker="o",
            label=label_names.get(int(label_idx), f"C{label_idx}"),
        )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


# =========================================================
# 레이어별 단독 플롯
# =========================================================
def plot_single_layer(layer, coords_off, coords_on,
                      labels_off, labels_on,
                      true_labels_off, true_labels_on,
                      output_dir, label_names, colors):
    """
    레이어 하나에 대해 4개 subplot:
      [0] LoRA OFF  — true label 색상
      [1] LoRA ON   — true label 색상
      [2] LoRA OFF  — pseudo label(cluster) 색상
      [3] LoRA ON   — pseudo label(cluster) 색상
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Layer {layer}  |  LoRA OFF vs LoRA ON  t-SNE\n"
        f"위: True Label  /  아래: Pseudo Label(Cluster)",
        fontsize=13, fontweight="bold",
    )

    # True label
    if true_labels_off is not None:
        draw_scatter(axes[0, 0], coords_off, true_labels_off,
                     f"Layer {layer}  LoRA OFF  (True Label)",
                     label_names, colors)
    else:
        axes[0, 0].set_title(f"Layer {layer}  LoRA OFF  (True Label N/A)")

    if true_labels_on is not None:
        draw_scatter(axes[0, 1], coords_on, true_labels_on,
                     f"Layer {layer}  LoRA ON   (True Label)",
                     label_names, colors)
    else:
        axes[0, 1].set_title(f"Layer {layer}  LoRA ON   (True Label N/A)")

    # Pseudo label (cluster)
    n_clusters = int(max(labels_off.max(), labels_on.max())) + 1
    cluster_names = {i: f"Cluster {i}" for i in range(n_clusters)}
    draw_scatter(axes[1, 0], coords_off, labels_off,
                 f"Layer {layer}  LoRA OFF  (Pseudo/Cluster)",
                 cluster_names, colors)
    draw_scatter(axes[1, 1], coords_on, labels_on,
                 f"Layer {layer}  LoRA ON   (Pseudo/Cluster)",
                 cluster_names, colors)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"tsne_layer{layer:02d}_lora_off_vs_on.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Layer {layer:2d}] 저장: {save_path}")


# =========================================================
# 전체 레이어 요약 플롯
# =========================================================
def plot_summary(all_layers, all_coords_off, all_coords_on,
                 true_labels, pseudo_labels,
                 output_dir, label_names, colors):
    """
    모든 레이어를 한 그림에: 행 = 레이어, 열 = [OFF true / ON true / OFF cluster / ON cluster]
    레이어 수가 많으면 한 페이지당 6레이어씩 분할 저장.
    """
    n_layers   = len(all_layers)
    page_size  = 6   # 한 파일에 몇 레이어씩
    n_pages    = (n_layers + page_size - 1) // page_size

    for page in range(n_pages):
        layer_slice = all_layers[page * page_size: (page + 1) * page_size]
        n_rows = len(layer_slice)
        fig, axes = plt.subplots(n_rows, 4, figsize=(24, 5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]   # (1, 4) 보장

        fig.suptitle(
            f"All Layers t-SNE Summary  (page {page+1}/{n_pages})\n"
            "col1: OFF True | col2: ON True | col3: OFF Cluster | col4: ON Cluster",
            fontsize=12, fontweight="bold",
        )

        for row_idx, layer in enumerate(layer_slice):
            coords_off = all_coords_off[layer]
            coords_on  = all_coords_on[layer]
            n_clusters = int(pseudo_labels.max()) + 1
            cluster_names = {i: f"C{i}" for i in range(n_clusters)}

            # col 0: OFF true label
            if true_labels is not None:
                draw_scatter(axes[row_idx, 0], coords_off, true_labels,
                             f"L{layer} OFF True", label_names, colors)
            # col 1: ON true label
            if true_labels is not None:
                draw_scatter(axes[row_idx, 1], coords_on, true_labels,
                             f"L{layer} ON True", label_names, colors)
            # col 2: OFF cluster
            draw_scatter(axes[row_idx, 2], coords_off, pseudo_labels,
                         f"L{layer} OFF Cluster", cluster_names, colors)
            # col 3: ON cluster
            draw_scatter(axes[row_idx, 3], coords_on, pseudo_labels,
                         f"L{layer} ON Cluster", cluster_names, colors)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"tsne_summary_page{page+1:02d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [Summary page {page+1}] 저장: {save_path}")


# =========================================================
# 메인
# =========================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── cluster_info 로드 (OFF 기준, pseudo_labels는 공통) ──────────
    pseudo_labels, true_labels_off, layers_off = load_cluster_info(
        LAYER_ANALYSIS_DIR, TAG_OFF)
    _, true_labels_on, layers_on = load_cluster_info(
        LAYER_ANALYSIS_DIR, TAG_ON)

    # 두 tag에 공통으로 존재하는 레이어만 처리
    common_layers = sorted(set(layers_off) & set(layers_on))
    print(f"처리할 레이어: {common_layers}  ({len(common_layers)}개)")

    all_coords_off = {}
    all_coords_on  = {}

    for layer in common_layers:
        print(f"\n[Layer {layer}] t-SNE 계산 중...")

        hidden_off = load_hidden(LAYER_ANALYSIS_DIR, layer, TAG_OFF)
        hidden_on  = load_hidden(LAYER_ANALYSIS_DIR, layer, TAG_ON)

        coords_off = compute_tsne(hidden_off, TSNE_PERPLEXITY, TSNE_RANDOM_STATE)
        coords_on  = compute_tsne(hidden_on,  TSNE_PERPLEXITY, TSNE_RANDOM_STATE)

        all_coords_off[layer] = coords_off
        all_coords_on[layer]  = coords_on

        # 레이어별 단독 플롯
        plot_single_layer(
            layer       = layer,
            coords_off  = coords_off,
            coords_on   = coords_on,
            labels_off  = pseudo_labels,
            labels_on   = pseudo_labels,
            true_labels_off = true_labels_off,
            true_labels_on  = true_labels_on,
            output_dir  = OUTPUT_DIR,
            label_names = LABEL_NAMES,
            colors      = COLORS,
        )

    # 전체 요약 플롯
    print("\n[Summary] 전체 레이어 요약 플롯 생성 중...")
    plot_summary(
        all_layers     = common_layers,
        all_coords_off = all_coords_off,
        all_coords_on  = all_coords_on,
        true_labels    = true_labels_off,
        pseudo_labels  = pseudo_labels,
        output_dir     = OUTPUT_DIR,
        label_names    = LABEL_NAMES,
        colors         = COLORS,
    )

    print(f"\n✅ 모든 플롯 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
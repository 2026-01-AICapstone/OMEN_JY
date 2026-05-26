import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from dataset import RepBendingDataset

# python visualize_clusters.py --output_dir ./out/test --method tsne

@torch.no_grad()
def extract_after_hidden(model, tokenizer, prompts, layer_idx, max_length):
    model.eval()
    vectors = []

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]
        vec = hidden[0, -1, :].float()
        vec = F.normalize(vec, p=2, dim=-1)

        vectors.append(vec.cpu().numpy())

        if (i + 1) % 100 == 0:
            print(f"AFTER hidden 추출 {i + 1}/{len(prompts)}")

    return np.stack(vectors, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./out/test")
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B-Instruct")
    #parser.add_argument("--base_model", default="google/gemma-2-2b-it") # 모델 수정
    parser.add_argument("--method", choices=["pca", "tsne"], default="tsne")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_name", default=None)
    args = parser.parse_args()

    print("\n========== Before vs After 비교 ==========")

    init_path = os.path.join(args.output_dir, "crush_centroids_initial.pt")
    if not os.path.exists(init_path):
        raise FileNotFoundError(f"Missing: {init_path}")

    info = torch.load(init_path, map_location="cpu")

    pseudo_labels = np.asarray(info["pseudo_labels"])
    centroids = info["centroids"].float().numpy()
    cluster_layer = int(info["cluster_layer"])
    num_clusters = int(info.get("num_clusters", len(np.unique(pseudo_labels))))

    print(f"cluster_layer = {cluster_layer}")
    print(f"num_clusters = {num_clusters}")

    cache_path = os.path.join(
        args.output_dir,
        "cache",
        f"dna_hidden_layer{cluster_layer}.npz"
    )

    if not os.path.exists(cache_path):
        candidates = glob.glob(os.path.join(args.output_dir, "cache", "dna_hidden_layer*.npz"))
        if not candidates:
            raise FileNotFoundError("cache/dna_hidden_layer*.npz 파일이 없습니다.")
        cache_path = candidates[0]
        print(f"지정 layer cache 없음. 대신 사용: {cache_path}")

    cache = np.load(cache_path)
    before_hidden = cache["hidden"]

    before_hidden = cache["hidden"].astype(np.float32)
    # ✅ before도 L2 normalize
    before_hidden = before_hidden / (np.linalg.norm(before_hidden, axis=1, keepdims=True) + 1e-12)

    # ✅ centroid도 다시 한 번 안전하게 normalize
    centroids = centroids.astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    print("before_hidden:", before_hidden.shape)
    print("centroids:", centroids.shape)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dataset = RepBendingDataset(
        tokenizer=tokenizer,
        num_examples=len(before_hidden),
        mode="response_all",
        max_length=args.max_length,
        model_name_or_path=args.base_model,
        dataset_path=None,
        split=None,
        is_online=False,
    )

    prompts = dataset.data_unsafe_request_prompts[:len(before_hidden)]

    print("prompts:", len(prompts))

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.output_dir,
    )

    after_hidden = extract_after_hidden(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_idx=cluster_layer,
        max_length=args.max_length,
    )

    print("after_hidden:", after_hidden.shape)

    all_vectors = np.concatenate(
        [before_hidden, after_hidden, centroids],
        axis=0,
    )

    if args.method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(all_vectors)
    else:
        reducer = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, max(5, len(before_hidden) // 20)),
            init="pca",
            learning_rate="auto",
        )
        reduced = reducer.fit_transform(all_vectors)

    n = len(before_hidden)
    k = len(centroids)

    before_2d = reduced[:n]
    after_2d = reduced[n:2 * n]
    centroid_2d = reduced[2 * n:2 * n + k]

    print("\n========== Before vs After 비교 ==========")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax_before = axes[0]
    ax_after = axes[1]

    # ============================================================
    # BEFORE
    # ============================================================

    for c in range(num_clusters):
        idx = pseudo_labels == c

        ax_before.scatter(
            before_2d[idx, 0],
            before_2d[idx, 1],
            s=20,
            alpha=0.65,
            label=f"C{c}",
        )

    ax_before.scatter(
        centroid_2d[:, 0],
        centroid_2d[:, 1],
        s=350,
        marker="X",
        edgecolors="black",
        linewidths=2,
        color="red",
    )

    for c, (x, y) in enumerate(centroid_2d):
        ax_before.text(
            x,
            y,
            f"C{c}",
            fontsize=13,
            weight="bold",
        )

    ax_before.set_title(
        f"BEFORE (Base Model)\nLayer {cluster_layer}",
        fontsize=15,
        weight="bold",
    )

    ax_before.set_xlabel(f"{args.method.upper()}-1")
    ax_before.set_ylabel(f"{args.method.upper()}-2")

    ax_before.grid(alpha=0.2)

    # ============================================================
    # AFTER
    # ============================================================

    for c in range(num_clusters):
        idx = pseudo_labels == c

        ax_after.scatter(
            after_2d[idx, 0],
            after_2d[idx, 1],
            s=20,
            alpha=0.65,
            label=f"C{c}",
        )

    ax_after.scatter(
        centroid_2d[:, 0],
        centroid_2d[:, 1],
        s=350,
        marker="X",
        edgecolors="black",
        linewidths=2,
        color="red",
    )

    for c, (x, y) in enumerate(centroid_2d):
        ax_after.text(
            x,
            y,
            f"C{c}",
            fontsize=13,
            weight="bold",
        )

    ax_after.set_title(
        f"AFTER (LoRA Fine-tuned)\nLayer {cluster_layer}",
        fontsize=15,
        weight="bold",
    )

    ax_after.set_xlabel(f"{args.method.upper()}-1")
    ax_after.set_ylabel(f"{args.method.upper()}-2")

    ax_after.grid(alpha=0.2)

    # ============================================================
    # 범례
    # ============================================================

    handles, labels = ax_before.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=num_clusters,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if args.save_name is None:
        args.save_name = f"before_after_subplot_{args.method}_layer{cluster_layer}.png"

    save_path = os.path.join(
        args.output_dir,
        args.save_name
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\n✅ 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
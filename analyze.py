import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from typing import Dict, List, Any


def load_results(output_dir: str) -> Dict[str, Any]:
    path = os.path.join(output_dir, "results.json")
    with open(path, "r") as f:
        return json.load(f)


def plot_metrics_comparison(results: Dict[str, Any], output_dir: str):
    variants = list(results.keys())
    clip_scores = [results[v].get("clip_score", 0) for v in variants]
    lpips_scores = [results[v].get("lpips_diversity", 0) for v in variants]
    hf_energies = [results[v].get("hf_energy_mean", 0) for v in variants]
    sat_stds = [results[v].get("saturation_std_mean", 0) for v in variants]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive CFG vs Baselines", fontsize=14, fontweight="bold")

    x = np.arange(len(variants))
    bar_w = 0.6
    colors = ["#2196F3" if "adaptive" in v else "#F44336" for v in variants]

    axes[0, 0].bar(x, clip_scores, width=bar_w, color=colors)
    axes[0, 0].set_title("CLIP Score (Alignment ↑)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(variants, rotation=30, ha="right", fontsize=7)

    axes[0, 1].bar(x, lpips_scores, width=bar_w, color=colors)
    axes[0, 1].set_title("LPIPS Diversity (↑)")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(variants, rotation=30, ha="right", fontsize=7)

    axes[1, 0].bar(x, hf_energies, width=bar_w, color=colors)
    axes[1, 0].set_title("HF Energy / Artifacts (↓)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(variants, rotation=30, ha="right", fontsize=7)

    axes[1, 1].bar(x, sat_stds, width=bar_w, color=colors)
    axes[1, 1].set_title("Saturation Std / Burn-in (↓)")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(variants, rotation=30, ha="right", fontsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Adaptive"),
        Patch(facecolor="#F44336", label="Fixed CFG"),
    ]
    fig.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto_curve(results: Dict[str, Any], output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for variant, metrics in results.items():
        clip = metrics.get("clip_score", 0)
        artifact = metrics.get("hf_energy_mean", 0)
        color = "#2196F3" if "adaptive" in variant else "#F44336"
        marker = "o" if "adaptive" in variant else "s"
        ax.scatter(artifact, clip, c=color, marker=marker, s=100, zorder=5)
        ax.annotate(
            variant.replace("adaptive_", "").replace("fixed_cfg_", "w="),
            (artifact, clip),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
        )

    ax.set_xlabel("HF Energy (Artifact Level ↓)")
    ax.set_ylabel("CLIP Score (Alignment ↑)")
    ax.set_title("Pareto Curve: Alignment vs Artifacts")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Adaptive"),
        Patch(facecolor="#F44336", label="Fixed CFG"),
    ]
    ax.legend(handles=legend_elements)

    save_path = os.path.join(output_dir, "pareto_curve.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_qualitative_grids(
    output_dir: str,
    variants: List[str],
    seed: int = 42,
    max_prompts: int = 4,
):
    image_rows = {}
    for variant in variants:
        seed_dir = os.path.join(output_dir, variant, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            continue
        files = sorted(
            [f for f in os.listdir(seed_dir) if f.endswith(".png")]
        )[:max_prompts]
        images = [Image.open(os.path.join(seed_dir, f)).resize((256, 256)) for f in files]
        image_rows[variant] = images

    if not image_rows:
        print("No images found for qualitative grid.")
        return

    n_variants = len(image_rows)
    n_cols = max_prompts
    fig, axes = plt.subplots(n_variants, n_cols, figsize=(n_cols * 3, n_variants * 3))

    if n_variants == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (variant, images) in enumerate(image_rows.items()):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(images):
                ax.imshow(images[col_idx])
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(
                    variant.replace("adaptive_", "adp_").replace("fixed_cfg_", "w="),
                    fontsize=7,
                    rotation=0,
                    labelpad=60,
                    va="center",
                )

    plt.suptitle("Qualitative Comparison Grid", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "qualitative_grid.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(output_dir: str):
    results = load_results(output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_pareto_curve(results, output_dir)
    variants = list(results.keys())
    generate_qualitative_grids(output_dir, variants)
    print("All plots generated.")

"""
Figure generation for the semantic-orchestration evaluation, using
matplotlib (already pinned in requirements.txt, no new dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_accuracy_by_category(report: Dict[str, Any], out_path: Path, title: str) -> None:
    categories = list(report["by_category"].keys())
    accs = [report["by_category"][c]["accuracy"] for c in categories]
    los = [report["by_category"][c]["accuracy"] - report["by_category"][c]["wilson_ci_lower"] for c in categories]
    his = [report["by_category"][c]["wilson_ci_upper"] - report["by_category"][c]["accuracy"] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(categories, accs, yerr=[los, his], capsize=4, color="#4C72B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_heatmap(cm: Dict[str, Any], out_path: Path, title: str) -> None:
    labels = cm["labels"]
    matrix = [[cm["matrix"][g].get(p, 0) for p in labels] for g in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i][j]
            if val:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val > (max(max(r) for r in matrix) / 2) else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_repeat_consistency(agreement: Dict[str, Any], out_path: Path, title: str) -> None:
    rates = agreement.get("pairwise_rates", [])
    if not rates:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([f"pair {i+1}" for i in range(len(rates))], rates, color="#55A868")
    ax.axhline(agreement.get("full_consistency_rate", 0), color="#C44E52", linestyle="--", label="full (3/3) consistency")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

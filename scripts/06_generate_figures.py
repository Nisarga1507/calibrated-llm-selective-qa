"""Generate publication-ready 300 DPI figures for 3S-CCR v2 results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})
import numpy as np

LABELS = {
    "truthful_qa": "TruthfulQA\n(General)",
    "fever": "FEVER\n(Fact Verify)",
    "uk_healthcare": "UK Healthcare\n(Clinical)",
    "uk_law": "UK Law\n(Statutory)",
    "uk_finance": "UK Finance\n(Financial)",
}

SHORT_LABELS = {
    "truthful_qa": "TruthfulQA",
    "fever": "FEVER",
    "uk_healthcare": "UK Healthcare",
    "uk_law": "UK Law",
    "uk_finance": "UK Finance",
}


def plot_risk_control_sweep(report: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300)
    alphas = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    ax1.plot([0, 0.22], [0, 0.22], "k--", lw=1.3, label="Target Risk Bound (Risk = α)")
    ax1.fill_between([0, 0.22], [0, 0.22], 0.25, color="#f8d7da", alpha=0.25)

    for (ds, ds_data), color in zip(report.items(), palette, strict=False):
        sweep = ds_data["alpha_sweep"]
        emp_risks = [sweep[str(a)]["selective_risk"] for a in alphas]
        coverages = [sweep[str(a)]["coverage"] * 100.0 for a in alphas]
        ax1.plot(
            alphas, emp_risks, marker="o", lw=2.0, color=color, label=SHORT_LABELS.get(ds, ds)
        )
        ax2.plot(
            alphas, coverages, marker="s", lw=2.0, color=color, label=SHORT_LABELS.get(ds, ds)
        )

    ax1.set_title("Conformal LTT Risk Control Across Target α", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Target Risk Level (α)", fontsize=10.5)
    ax1.set_ylabel("Empirical Held-Out Selective Risk", fontsize=10.5)
    ax1.set_xlim(0.0, 0.21)
    ax1.set_ylim(-0.005, 0.21)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.8, frameon=True)

    ax2.set_title("Selective Coverage vs. Risk Tolerance (α)", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Target Risk Level (α)", fontsize=10.5)
    ax2.set_ylabel("Accepted Query Coverage (%)", fontsize=10.5)
    ax2.set_xlim(0.0, 0.21)
    ax2.set_ylim(-3.0, 105.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=8.8, frameon=True)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_risk_coverage_alpha_sweep.png", dpi=300)
    plt.close(fig)


def plot_ablation_comparison(ablation_rows: list[dict], out_dir: Path) -> None:
    datasets = list(SHORT_LABELS.keys())
    variants = [
        ("1_S_logit_only", "S_logit Only (Stage-1)", "#8c969e"),
        ("2_S_sem_only", "S_sem Only", "#6baed6"),
        ("3_2S_CCR_no_ret", "2S-CCR (No Retrieval)", "#3182bd"),
        ("4_3S_CCR_full", "3S-CCR (Full Cascade)", "#08519c"),
    ]

    lookup = {(r["dataset"], r["variant"]): r for r in ablation_rows}
    x = np.arange(len(datasets))
    width = 0.19

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    for idx, (var_key, label, color) in enumerate(variants):
        offset = (idx - 1.5) * width
        aurocs = [float(lookup[(ds, var_key)]["auroc"]) for ds in datasets]
        aurcs = [float(lookup[(ds, var_key)]["aurc"]) for ds in datasets]
        ax1.bar(x + offset, aurocs, width=width, label=label, color=color, edgecolor="white")
        ax2.bar(x + offset, aurcs, width=width, label=label, color=color, edgecolor="white")

    ax1.set_title(
        "4-Way Signal Ablation: Discriminative AUROC (↑)", fontsize=11.5, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[d] for d in datasets], fontsize=9.0)
    ax1.set_ylabel("AUROC (Higher is Better)", fontsize=10.5)
    ax1.set_ylim(0.50, 1.02)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=True)

    ax2.set_title("4-Way Signal Ablation: Selective AURC (↓)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[d] for d in datasets], fontsize=9.0)
    ax2.set_ylabel("AURC (Lower is Better)", fontsize=10.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_ablation_auroc_aurc.png", dpi=300)
    plt.close(fig)


def plot_uk_enterprise_economics(summary_rows: list[dict], out_dir: Path) -> None:
    datasets = [r["dataset"] for r in summary_rows]
    always_cost = [float(r["always_accept_gbp_per_1k"]) / 1000.0 for r in summary_rows]
    cascade_cost = [float(r["cascade_3sccr_gbp_per_1k"]) / 1000.0 for r in summary_rows]
    savings_pct = [float(r["total_risk_cost_savings_pct"]) for r in summary_rows]

    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    ax.bar(
        x - width / 2,
        always_cost,
        width=width,
        label="Always-Accept Uncalibrated LLM",
        color="#d95f02",
        edgecolor="white",
    )
    bars_ccr = ax.bar(
        x + width / 2,
        cascade_cost,
        width=width,
        label="3S-CCR Conformal Cascade (α = 0.05)",
        color="#1b9e77",
        edgecolor="white",
    )

    for idx, bar in enumerate(bars_ccr):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 3.5,
            f"-{savings_pct[idx]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#1b9e77",
        )

    ax.set_title(
        "UK Enterprise Operational Risk Cost per 1,000 Queries (£k GBP)",
        fontsize=12.0,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[d] for d in datasets], fontsize=9.5)
    ax.set_ylabel("Expected Total Cost per 1,000 Queries (£k GBP)", fontsize=10.5)
    ax.set_ylim(0, max(always_cost) * 1.16)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_uk_enterprise_economics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    tables_dir = Path("results/tables")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads((tables_dir / "v2_full_conformal_report.json").read_text(encoding="utf-8"))
    with (tables_dir / "v2_ablation_comparison.csv").open("r", encoding="utf-8") as f:
        ablation_rows = list(csv.DictReader(f))
    with (tables_dir / "v2_benchmark_summary.csv").open("r", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    plot_risk_control_sweep(report, fig_dir)
    plot_ablation_comparison(ablation_rows, fig_dir)
    plot_uk_enterprise_economics(summary_rows, fig_dir)
    print(f"Generated 3 publication-ready figures in {fig_dir}/")


if __name__ == "__main__":
    main()
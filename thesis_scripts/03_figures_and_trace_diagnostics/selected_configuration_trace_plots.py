#!/usr/bin/env python3
"""Plot the endpoint-severity condition winners as compact trace figures.

The selected detector--layer combinations come from the 20-seed primary
benchmark.  Each panel then shows the full seed-42, episode-0 static-PCA path
for that preselected configuration.  Traces remain in the detector's original
units and the dashed horizontal line is the calibrated seed-specific limit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "patchcamelyon": "PatchCamelyon",
}
MECHANISM_LABELS = {
    "jpeg": "JPEG artifacting",
    "stretch": "Horizontal stretching",
    "saturation": "Saturation",
    "ooc": "OOC substitution",
}
BLOCK_LABELS = {
    "layer1": "layer 1",
    "layer2": "layer 2",
    "layer4": "layer 4",
}
DETECTOR_LABELS = {
    "T2": r"$T^2$",
    "EWMA": "MEWMA",
    "MMD": "MMD",
    "KL-Gauss": "Gaussian KL",
    "Frechet": r"Squared Fr\'echet",
    "SPE": "SPE",
}

# Lowest endpoint-severity restricted delay in each dataset/onset/mechanism
# cell of the completed 20-seed primary benchmark.
WINNERS = {
    "cifar10": {
        "sudden": {
            "jpeg": ("T2", "layer2"),
            "stretch": ("T2", "layer2"),
            "saturation": ("KL-Gauss", "layer1"),
            "ooc": ("SPE", "layer4"),
        },
        "incremental": {
            "jpeg": ("MMD", "layer2"),
            "stretch": ("T2", "layer1"),
            "saturation": ("Frechet", "layer1"),
            "ooc": ("SPE", "layer4"),
        },
        "gradual": {
            "jpeg": ("T2", "layer2"),
            "stretch": ("T2", "layer2"),
            "saturation": ("KL-Gauss", "layer1"),
            "ooc": ("SPE", "layer4"),
        },
    },
    "patchcamelyon": {
        "sudden": {
            "jpeg": ("T2", "layer1"),
            "stretch": ("KL-Gauss", "layer1"),
            "saturation": ("KL-Gauss", "layer1"),
            "ooc": ("T2", "layer4"),
        },
        "incremental": {
            "jpeg": ("T2", "layer1"),
            "stretch": ("T2", "layer1"),
            "saturation": ("T2", "layer1"),
            "ooc": ("EWMA", "layer4"),
        },
        "gradual": {
            "jpeg": ("T2", "layer1"),
            "stretch": ("T2", "layer1"),
            "saturation": ("SPE", "layer1"),
            "ooc": ("EWMA", "layer4"),
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "01_multidataset_experiment",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def trace_root(project_root: Path, dataset: str) -> Path:
    if dataset == "cifar10":
        return project_root / "staticpca_curated_trace_sources" / dataset
    return project_root / "staticpca_temporal_alarm_traces" / dataset


def condition_path(root: Path, onset: str, mechanism: str, severity: float,
                   seed: int, episode: int) -> Path:
    stem = (
        f"{onset}_{mechanism}_severity{severity:.2f}_"
        f"seed{seed}_episode{episode:03d}_traces.csv"
    )
    return root / stem


def plot_dataset_onset(dataset: str, onset: str, args: argparse.Namespace) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    trace_color = "#1f4e79"
    threshold_color = "#b22222"
    onset_color = "#303030"
    alarm_color = "#f28e2b"

    for axis, mechanism in zip(axes.flat, ("jpeg", "stretch", "saturation", "ooc")):
        detector, block = WINNERS[dataset][onset][mechanism]
        path = condition_path(
            trace_root(args.project_root, dataset), onset, mechanism,
            args.severity, args.seed, args.episode,
        )
        frame = pd.read_csv(path)
        selected = frame[
            (frame["detector"] == detector) & (frame["block"] == block)
        ].copy()
        if selected.empty:
            raise RuntimeError(f"Missing {dataset}/{onset}/{mechanism}/{detector}/{block}")
        selected = selected.sort_values("observation")
        threshold = float(selected["threshold"].iloc[0])
        changepoint = 50
        axis.axvspan(1, changepoint - 1, color="#d9d9d9", alpha=0.45)
        axis.plot(
            selected["observation"], selected["statistic"],
            color=trace_color, linewidth=1.15,
        )
        axis.axhline(
            threshold, color=threshold_color, linestyle="--", linewidth=1.25,
        )
        axis.axvline(
            changepoint, color=onset_color, linestyle=":", linewidth=1.25,
        )
        hit = selected[selected["first_alarm"].astype(str).str.lower() == "true"]
        if not hit.empty:
            alarm_observation = int(hit["observation"].iloc[0])
            alarm_statistic = float(hit["statistic"].iloc[0])
            delay = alarm_observation - changepoint + 1
            axis.axvline(alarm_observation, color=alarm_color, linewidth=1.15)
            axis.scatter(
                [alarm_observation], [alarm_statistic], color=alarm_color,
                edgecolor="black", linewidth=0.4, s=35, zorder=5,
            )
            alarm_text = f"alarm={alarm_observation}; delay={delay}"
        else:
            alarm_text = "no alarm (censored)"
        axis.set_xlim(1, 1000)
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
        axis.set_xlabel("Stream observation")
        axis.set_ylabel("Detector statistic")
        axis.set_title(
            f"{MECHANISM_LABELS[mechanism]}: "
            f"{DETECTOR_LABELS[detector]}, {BLOCK_LABELS[block]}\n"
            f"threshold={threshold:.4g}; {alarm_text}",
            fontsize=10.5,
        )

    figure.suptitle(
        f"{DATASET_LABELS[dataset]} endpoint-winner traces: {onset} drift\n"
        f"seed={args.seed}, episode={args.episode}, severity={args.severity:g}; "
        "configurations selected from the 20-seed benchmark",
        fontsize=14,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"best_traces_{dataset}_{onset}.png"
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for dataset in ("cifar10", "patchcamelyon"):
        for onset in ("sudden", "incremental", "gradual"):
            print(plot_dataset_onset(dataset, onset, args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create paired temporal alarm-trace figures for the moving-window PCA experiment.

The script reuses the primary dataset pipelines, fitted Phase-I objects,
dataset-specific VAEs, fixed-reference moving-window PCA implementation,
and calibrated moving-window thresholds.  It does not train a VAE or refit the primary
static PCA model.

By default it uses seed 42, Monte Carlo episode 0, and severity 1.0 for both
datasets.  The same seed, episode, IC order, and OOC order are therefore used
for every paired drift condition.  One 12-panel figure is produced for every
dataset x onset pattern x mechanism combination:

    rows:    MEWMA, MMD, Gaussian Hellinger, Gaussian KL
    columns: ResNet layer 1, ResNet layer 4, VAE latent representation

The detector trace is plotted on its original scale against stream-observation
number.  Each panel shows its calibrated threshold and first alarm.  The first
score occurs at observation 50, so observations 1--49 are displayed as the
warm-up period and the changepoint is marked at observation 50.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Dict, Iterable, List, Sequence


DATASETS = ("cifar10", "patchcamelyon")
PATTERNS = ("sudden", "incremental", "gradual")
MECHANISMS = ("ooc", "jpeg", "saturation", "stretch")
BLOCKS = ("layer1", "layer2", "layer3", "layer4", "vae_latent")
DEFAULT_BLOCKS = ("layer1", "layer4", "vae_latent")
DETECTORS = ("T2", "EWMA", "MMD", "KL-Gauss", "Frechet", "SPE")

SOURCE_NOTEBOOKS = {
    "cifar10": "cifar10_static_pca_benchmark.ipynb",
    "patchcamelyon": "patchcamelyon_static_pca_benchmark.ipynb",
}
HELPER_NAME = "moving_window_pca_fixed_reference.py"

DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "patchcamelyon": "PatchCamelyon",
}
BLOCK_LABELS = {
    "layer1": "ResNet-18 layer 1",
    "layer2": "ResNet-18 layer 2",
    "layer3": "ResNet-18 layer 3",
    "layer4": "ResNet-18 layer 4",
    "vae_latent": "VAE latent",
}
DETECTOR_LABELS = {
    "T2": "$T^2$",
    "EWMA": "MEWMA",
    "MMD": "MMD",
    "KL-Gauss": "Gaussian KL",
    "Frechet": "Squared Fr\u00e9chet",
    "SPE": "SPE",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot calibrated static-PCA detector traces for one paired seed "
            "and Monte Carlo episode."
        )
    )
    parser.add_argument(
        "--dataset", choices=("all", *DATASETS), default="all",
        help="Dataset to process. 'all' runs each dataset in a clean subprocess.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Zero-based Monte Carlo replication number.",
    )
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument(
        "--patterns", nargs="+", choices=PATTERNS, default=list(PATTERNS),
    )
    parser.add_argument(
        "--mechanisms", nargs="+", choices=MECHANISMS,
        default=list(MECHANISMS),
    )
    parser.add_argument(
        "--blocks", nargs="+", choices=BLOCKS, default=list(DEFAULT_BLOCKS),
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "01_multidataset_experiment",
        help="Folder containing the primary benchmark notebooks, moving-window helper, and result folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output folder. Defaults to PROJECT_ROOT/staticpca_temporal_alarm_traces."
        ),
    )
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--raw-scale", dest="raw_scale", action="store_true", default=True,
        help=(
            "Plot detector values on their original scale and draw the alarm "
            "boundary at the calibrated threshold (default)."
        ),
    )
    scale_group.add_argument(
        "--normalized-scale", dest="raw_scale", action="store_false",
        help=(
            "Divide detector values by their calibrated threshold, placing the "
            "alarm boundary at 1. Use only when a normalized comparison is "
            "explicitly required."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute conditions whose PNG, trace CSV, and alarm CSV already exist.",
    )
    return parser.parse_args(argv)


def _child_command(dataset: str, args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--dataset", dataset,
        "--seed", str(args.seed),
        "--episode", str(args.episode),
        "--severity", str(args.severity),
        "--patterns", *args.patterns,
        "--mechanisms", *args.mechanisms,
        "--blocks", *args.blocks,
        "--project-root", str(args.project_root),
        "--dpi", str(args.dpi),
    ]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if not args.raw_scale:
        command.append("--normalized-scale")
    if args.overwrite:
        command.append("--overwrite")
    return command


def _run_all_in_subprocesses(args: argparse.Namespace) -> None:
    for dataset in DATASETS:
        print(f"\n=== starting {DATASET_LABELS[dataset]} trace process ===", flush=True)
        subprocess.run(_child_command(dataset, args), check=True)


def _validate_paths(project_root: Path, dataset: str) -> Path:
    source = project_root / SOURCE_NOTEBOOKS[dataset]
    if not source.is_file():
        raise FileNotFoundError(f"Required pipeline file was not found: {source}")
    return source


def _load_pipeline(project_root: Path, dataset: str, seed: int,
                   episode: int) -> tuple[Dict[str, Any], Any]:
    """Load one dataset's locked primary benchmark definitions in an isolated module.

    Nothing is refitted. The cached seed state supplies the frozen PCA basis,
    the fixed reference, and the calibrated thresholds exactly as the primary
    benchmark produced them.
    """
    source_notebook = _validate_paths(project_root, dataset)

    os.environ["DRIFT_RUN_FULL"] = "0"
    os.environ["DRIFT_QUICK"] = "0"
    os.environ["DRIFT_RUN_BASE"] = str(project_root)

    module_name = f"_staticpca_trace_pipeline_{dataset}"
    module = types.ModuleType(module_name)
    module.__file__ = str(source_notebook)
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module_name] = module
    namespace = module.__dict__

    document = json.loads(source_notebook.read_text())
    for cell_index, cell in enumerate(document["cells"]):
        if cell_index >= 24:
            break
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("%")
        )
        exec(
            compile(source, f"{source_notebook.name}:cell{cell_index}", "exec"),
            namespace,
        )

    cfg = replace(
        namespace["CFG"],
        seeds=(seed,),
        mc_arl1_reps=max(episode + 1, 1),
        force_recompute=False,
    )
    namespace["CFG"] = cfg
    return namespace, cfg


def _first_alarm(scores: Any, threshold: float) -> int | None:
    import numpy as np

    hits = np.flatnonzero(np.asarray(scores, dtype=float) > float(threshold))
    return None if not len(hits) else int(hits[0])


def _condition_stem(pattern: str, mechanism: str, severity: float,
                    seed: int, episode: int) -> str:
    return (
        f"{pattern}_{mechanism}_severity{severity:.2f}_"
        f"seed{seed}_episode{episode:03d}"
    )


def _existing_condition(summary_path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    return pd.read_csv(summary_path).to_dict("records")


def _plot_condition(
    *, dataset: str, pattern: str, mechanism: str, severity: float,
    seed: int, episode: int, images: Any, blocks: Dict[str, Any],
    state: Dict[str, Any], namespace: Dict[str, Any],
    cfg: Any, requested_blocks: Iterable[str], output_dir: Path,
    raw_scale: bool, dpi: int,
) -> tuple[Path, Path, Path, List[Dict[str, Any]]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    requested_blocks = tuple(requested_blocks)
    stem = _condition_stem(pattern, mechanism, severity, seed, episode)
    figure_path = output_dir / f"{stem}.png"
    trace_path = output_dir / f"{stem}_traces.csv"
    alarm_path = output_dir / f"{stem}_alarms.csv"

    traces: Dict[tuple[str, str], np.ndarray] = {}
    thresholds: Dict[tuple[str, str], float] = {}
    alarm_rows: List[Dict[str, Any]] = []
    trace_frames: List[pd.DataFrame] = []

    k = cfg.pca_dim
    for block in requested_blocks:
        pca = state["pca_models"][k][block]
        reference = namespace["prepare_reference"](state["ref_proj"][k][block], cfg)
        projected = pca.transform(blocks[block]).astype(float)
        # every fixed-reference score in one streaming pass over the episode
        stream = namespace["stream_stats"](reference, projected, cfg)
        scored = {name: values for name, (values, _direction) in stream.items()}
        # SPE is raw-space and has its own path in the primary pipeline
        scored["SPE"] = namespace["window_means"](
            namespace["pca_spe"](pca, blocks[block], projected), cfg
        )
        for method in DETECTORS:
            values = np.asarray(scored[method], dtype=float)
            parameter = cfg.ewma_lambda if method == "EWMA" else None
            threshold = float(
                state["thresholds"][(k, block, method, parameter)]["threshold"]
            )
            hit = _first_alarm(values, threshold)
            observations = (
                cfg.window + np.arange(len(values), dtype=int) * cfg.stride
            )
            detected = hit is not None
            alarm_observation = int(observations[hit]) if detected else None
            delay = (
                int(alarm_observation - cfg.changepoint + 1)
                if detected else None
            )
            traces[(block, method)] = values
            thresholds[(block, method)] = threshold
            alarm_rows.append({
                "dataset": dataset,
                "seed": seed,
                "episode": episode,
                "pattern": pattern,
                "mechanism": mechanism,
                "severity": severity,
                "block": block,
                "detector": method,
                "threshold": threshold,
                "detected": detected,
                "first_alarm_score_index": hit,
                "first_alarm_observation": alarm_observation,
                "arl1_delay_observations": delay,
                "changepoint_observation": int(cfg.changepoint),
                "window": int(cfg.window),
                "stride": int(cfg.stride),
            })
            trace_frames.append(pd.DataFrame({
                "dataset": dataset,
                "seed": seed,
                "episode": episode,
                "pattern": pattern,
                "mechanism": mechanism,
                "severity": severity,
                "block": block,
                "detector": method,
                "score_index": np.arange(len(values), dtype=int),
                "observation": observations,
                "statistic": values,
                "threshold": threshold,
                "statistic_over_threshold": values / threshold,
                "first_alarm": (
                    np.arange(len(values), dtype=int) == hit
                    if detected else np.zeros(len(values), dtype=bool)
                ),
            }))

    pd.concat(trace_frames, ignore_index=True).to_csv(trace_path, index=False)
    pd.DataFrame(alarm_rows).to_csv(alarm_path, index=False)

    figure, axes = plt.subplots(
        len(DETECTORS), len(requested_blocks),
        figsize=(5.8 * len(requested_blocks), 3.45 * len(DETECTORS)),
        sharex=True, constrained_layout=True,
        squeeze=False,
    )
    trace_color = "#1f4e79"
    threshold_color = "#b22222"
    onset_color = "#303030"
    alarm_color = "#f28e2b"

    alarm_lookup = {
        (row["block"], row["detector"]): row for row in alarm_rows
    }
    for row_index, method in enumerate(DETECTORS):
        for column_index, block in enumerate(requested_blocks):
            axis = axes[row_index, column_index]
            values = traces[(block, method)]
            threshold = thresholds[(block, method)]
            observations = cfg.window + np.arange(len(values)) * cfg.stride
            plotted = values if raw_scale else values / threshold
            plotted_threshold = threshold if raw_scale else 1.0
            alarm = alarm_lookup[(block, method)]

            axis.axvspan(
                1, cfg.changepoint - 1,
                color="#d9d9d9", alpha=0.45, label="pre-change",
            )
            axis.plot(observations, plotted, color=trace_color, linewidth=1.05)
            axis.axhline(
                plotted_threshold, color=threshold_color,
                linestyle="--", linewidth=1.25,
            )
            axis.axvline(
                cfg.changepoint, color=onset_color,
                linestyle=":", linewidth=1.25,
            )
            if alarm["detected"]:
                hit = int(alarm["first_alarm_score_index"])
                alarm_x = int(alarm["first_alarm_observation"])
                alarm_y = float(plotted[hit])
                axis.axvline(alarm_x, color=alarm_color, linewidth=1.15, alpha=0.9)
                axis.scatter(
                    [alarm_x], [alarm_y], color=alarm_color,
                    edgecolor="black", linewidth=0.4, s=30, zorder=5,
                )
                alarm_text = (
                    f"alarm={alarm_x}; delay={alarm['arl1_delay_observations']}"
                )
            else:
                alarm_text = "no alarm (censored)"

            axis.set_xlim(1, len(images))
            axis.grid(axis="y", color="#e5e5e5", linewidth=0.7)
            axis.set_title(
                f"{BLOCK_LABELS[block]} | {DETECTOR_LABELS[method]}\n"
                f"threshold={threshold:.4g}; {alarm_text}",
                fontsize=9.5,
            )
            if column_index == 0:
                axis.set_ylabel(
                    "Detector statistic" if raw_scale
                    else "Statistic / calibrated threshold"
                )
            if row_index == len(DETECTORS) - 1:
                axis.set_xlabel("Stream observation")

    figure.suptitle(
        f"{DATASET_LABELS[dataset]} frozen-basis static PCA: "
        f"{pattern} {mechanism} drift\n"
        f"severity={severity:g}, seed={seed}, episode={episode}; "
        f"changepoint={cfg.changepoint}",
        fontsize=14,
    )
    figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return figure_path, trace_path, alarm_path, alarm_rows


def run_dataset(args: argparse.Namespace) -> None:
    import pandas as pd

    dataset = args.dataset
    project_root = args.project_root.expanduser().absolute()
    output_root = (
        args.output_dir.expanduser().absolute()
        if args.output_dir is not None
        else project_root / "staticpca_temporal_alarm_traces"
    )
    dataset_output = output_root / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_LABELS[dataset]} for seed {args.seed}, "
        f"episode {args.episode}",
        flush=True,
    )
    namespace, cfg = _load_pipeline(
        project_root, dataset, args.seed, args.episode
    )
    namespace["build_seed_partition"](args.seed, cfg)
    # loads the cached primary seed state: frozen PCA, fixed reference, thresholds
    state = namespace["fit_and_calibrate_seed"](
        args.seed, cfg, namespace["EXTRACTOR"]
    )
    vae = namespace["train_seed_vae"](args.seed, cfg)
    ic_orders, ooc_orders = namespace["master_episode_manifest"](args.seed, cfg)

    all_alarm_rows: List[Dict[str, Any]] = []
    total = len(args.patterns) * len(args.mechanisms)
    completed = 0
    for pattern in args.patterns:
        for mechanism in args.mechanisms:
            completed += 1
            stem = _condition_stem(
                pattern, mechanism, args.severity, args.seed, args.episode
            )
            figure_path = dataset_output / f"{stem}.png"
            trace_path = dataset_output / f"{stem}_traces.csv"
            alarm_path = dataset_output / f"{stem}_alarms.csv"
            if (
                not args.overwrite
                and figure_path.exists()
                and trace_path.exists()
                and alarm_path.exists()
            ):
                cached = _existing_condition(alarm_path)
                have = {
                    (str(row["block"]), str(row["detector"])) for row in cached
                }
                want = {
                    (block, detector)
                    for block in args.blocks for detector in DETECTORS
                }
                if want <= have:
                    print(
                        f"[{completed}/{total}] reusing {figure_path.name}",
                        flush=True,
                    )
                    all_alarm_rows.extend(cached)
                    continue
                # The stem encodes only pattern, mechanism, severity, seed and
                # episode, so cached output from a narrower --blocks or an older
                # detector set must not be reused. Recompute instead.
                print(
                    f"[{completed}/{total}] recomputing {figure_path.name}: "
                    f"cached output covers {len(have)} block-detector pairs, "
                    f"{len(want)} requested",
                    flush=True,
                )

            print(
                f"[{completed}/{total}] scoring {pattern}/{mechanism}/"
                f"severity={args.severity:g}",
                flush=True,
            )
            images, observed_ids = namespace["build_episode"](
                args.seed, args.episode, pattern, mechanism, args.severity,
                ic_orders[args.episode], ooc_orders[args.episode], cfg,
            )
            if len(set(map(int, observed_ids))) != len(observed_ids):
                raise AssertionError(
                    f"Episode {pattern}/{mechanism} unexpectedly reuses identities"
                )
            features = namespace["EXTRACTOR"].extract(images)
            features["vae_latent"] = namespace["vae_latents"](vae, images)
            blocks = {name: features[name] for name in args.blocks}
            _, _, _, alarm_rows = _plot_condition(
                dataset=dataset,
                pattern=pattern,
                mechanism=mechanism,
                severity=args.severity,
                seed=args.seed,
                episode=args.episode,
                images=images,
                blocks=blocks,
                state=state,
                namespace=namespace,
                cfg=cfg,
                requested_blocks=args.blocks,
                output_dir=dataset_output,
                raw_scale=args.raw_scale,
                dpi=args.dpi,
            )
            all_alarm_rows.extend(alarm_rows)

    summary = pd.DataFrame(all_alarm_rows).sort_values(
        ["pattern", "mechanism", "block", "detector"]
    )
    summary_path = dataset_output / (
        f"alarm_summary_seed{args.seed}_episode{args.episode:03d}_"
        f"severity{args.severity:.2f}.csv"
    )
    summary.to_csv(summary_path, index=False)
    metadata_path = dataset_output / (
        f"run_metadata_seed{args.seed}_episode{args.episode:03d}_"
        f"severity{args.severity:.2f}.json"
    )
    metadata_path.write_text(json.dumps({
        "dataset": dataset,
        "seed": args.seed,
        "episode": args.episode,
        "severity": args.severity,
        "patterns": list(args.patterns),
        "mechanisms": list(args.mechanisms),
        "blocks": list(args.blocks),
        "detectors": list(DETECTORS),
        "window": int(cfg.window),
        "stride": int(cfg.stride),
        "changepoint": int(cfg.changepoint),
        "pca_dimension": int(cfg.pca_dim),
        "threshold_source": str(cfg.out_root),
        "normalized_by_threshold": not args.raw_scale,
        "output_directory": str(dataset_output),
    }, indent=2))
    print(f"Finished {DATASET_LABELS[dataset]}: {dataset_output}", flush=True)

    del vae, state, namespace
    gc.collect()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episode < 0:
        raise ValueError("--episode must be zero or greater")
    if args.severity <= 0:
        raise ValueError("--severity must be positive")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if args.dataset == "all":
        _run_all_in_subprocesses(args)
        return
    run_dataset(args)


if __name__ == "__main__":
    main()

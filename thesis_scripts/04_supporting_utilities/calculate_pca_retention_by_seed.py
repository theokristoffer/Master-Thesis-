#!/usr/bin/env python3
"""Calculate PCA component counts for target cumulative variance levels.

This utility deliberately reuses the data, split, augmentation, ResNet-18, and
VAE definitions in the thesis benchmark notebook.  It only runs the Phase-I
feature extraction needed for PCA; it does not calibrate detectors or run any
Phase-II drift episodes.

Example
-------
python3 calculate_pca_retention_by_seed.py --dataset both --seed 42

The output CSV contains one row per dataset, feature representation, and target
retention level.  With the default settings, the PCA sample and random-number
stream match the corresponding CPV-based benchmark for that seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NOTEBOOK_NAMES = {
    "cifar10": "cifar10_cpv70_static_pca_base.ipynb",
    "patchcamelyon": "patchcamelyon_cpv70_static_pca_base.ipynb",
}

# These cells define the configuration, dataset, preprocessing/backbone,
# feature hooks, VAE, VAE cache loader, and Phase-I feature helpers.
CELL_MARKERS = (
    "class Config:",
    "def activate_partition(",
    "def to_unit_tensor(",
    "class BlockProbeExtractor:",
    "class ConvVAE(",
    "def split_metadata(",
    "def _extract_features(",
)


def _plain_python(source: str) -> str:
    """Remove notebook-only line magics before executing a code cell."""
    return "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


def _find_definition_cells(notebook: dict) -> list[str]:
    selected: list[str] = []
    found: set[str] = set()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for marker in CELL_MARKERS:
            if marker in source:
                selected.append(_plain_python(source))
                found.add(marker)
                break
    missing = set(CELL_MARKERS) - found
    if missing:
        raise RuntimeError(f"Notebook is missing expected definitions: {sorted(missing)}")
    return selected


def _load_notebook_environment(
    notebook_path: Path,
    runtime_root: Path,
    notebook_root: Path,
    fit_n: int | None,
) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = _find_definition_cells(notebook)

    # The notebook configuration reads these during its first code cell.
    old_run_base = os.environ.get("DRIFT_RUN_BASE")
    old_run_full = os.environ.get("DRIFT_RUN_FULL")
    os.environ["DRIFT_RUN_BASE"] = str(runtime_root)
    os.environ["DRIFT_RUN_FULL"] = "0"
    module_name = f"pca_retention_notebook_{notebook_path.stem}"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook_path)
    sys.modules[module_name] = module
    namespace = module.__dict__
    try:
        for index, source in enumerate(cells):
            exec(compile(source, f"{notebook_path.name}:definition-{index}", "exec"), namespace)
            if index == 0:
                cfg = namespace["CFG"]
                if fit_n is not None:
                    cfg.pca_fit_n = int(fit_n)
                # Read compatible VAE checkpoints from completed experiments,
                # while keeping every new file inside runtime_root.
                cfg.vae_legacy_out_roots = (
                    str(notebook_root / f"outputs/{cfg.dataset_name}_cpv70_static_pca"),
                    str(notebook_root / f"outputs/{cfg.dataset_name}_static_pca_benchmark"),
                )
    finally:
        if old_run_base is None:
            os.environ.pop("DRIFT_RUN_BASE", None)
        else:
            os.environ["DRIFT_RUN_BASE"] = old_run_base
        if old_run_full is None:
            os.environ.pop("DRIFT_RUN_FULL", None)
        else:
            os.environ["DRIFT_RUN_FULL"] = old_run_full
    return namespace


def _fit_to_largest_target(raw: np.ndarray, largest_target: float, solver: str):
    """Fit once to the largest target; retained ratios also identify lower k."""
    from sklearn.decomposition import PCA

    model = PCA(n_components=float(largest_target), svd_solver=solver).fit(raw)
    cumulative = np.cumsum(model.explained_variance_ratio_, dtype=np.float64)
    return model, cumulative


def _rows_for_dataset(
    dataset: str,
    seed: int,
    targets: Iterable[float],
    notebook_root: Path,
    runtime_root: Path,
    fit_n: int | None,
    solver: str,
    moving_window: int,
) -> list[dict]:
    notebook_path = notebook_root / NOTEBOOK_NAMES[dataset]
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Benchmark notebook not found: {notebook_path}")

    env = _load_notebook_environment(
        notebook_path=notebook_path,
        runtime_root=runtime_root / dataset,
        notebook_root=notebook_root,
        fit_n=fit_n,
    )
    cfg = env["CFG"]
    env["build_seed_partition"](seed, cfg)
    vae = env["train_seed_vae"](seed, cfg)

    # This is the exact random stream and sampling order used by
    # fit_and_calibrate_seed in the benchmark notebook.
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.stream_master_seed, seed, 20])
    )
    train_ic_ids = env["TRAIN_IC_IDS"]
    fit_ids = rng.permutation(train_ic_ids)[: min(cfg.pca_fit_n, len(train_ic_ids))]
    _, features = env["_extract_features"](
        fit_ids, rng, env["EXTRACTOR"], vae, augment=True
    )

    targets = tuple(sorted(set(float(value) for value in targets)))
    largest_target = max(targets)
    rows: list[dict] = []
    max_moving_components = moving_window - 1
    for block in env["FEATURE_BLOCKS"]:
        raw = np.asarray(features[block])
        model, cumulative = _fit_to_largest_target(raw, largest_target, solver)
        for target in targets:
            # searchsorted returns the smallest k for which CPV >= target.
            k = int(np.searchsorted(cumulative, target, side="left") + 1)
            if k > len(cumulative):
                raise RuntimeError(
                    f"{dataset}/{block}: fitted PCA did not reach target {target:.3f}"
                )
            rows.append({
                "dataset": dataset,
                "seed": int(seed),
                "split_hash": env["SPLIT_HASH"],
                "phase1_observations": int(len(fit_ids)),
                "feature_representation": block,
                "raw_dimension": int(raw.shape[1]),
                "target_variance_retention": float(target),
                "principal_components": k,
                "achieved_variance_retention": float(cumulative[k - 1]),
                "pca_solver": solver,
                "moving_window": int(moving_window),
                "moving_window_max_components": int(max_moving_components),
                "exceeds_moving_window_rank": bool(k > max_moving_components),
            })
        del model, cumulative
    return rows


def _parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    default_root = package_root / "02_observation_level_calibration_extension"
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the number of Phase-I PCA components required to reach "
            "specified cumulative explained-variance targets for one seed."
        )
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "patchcamelyon", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=float, nargs="+", default=(0.70, 0.80))
    parser.add_argument("--notebook-dir", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir", type=Path,
        default=package_root / "outputs" / "pca_retention_audit",
    )
    parser.add_argument(
        "--fit-n", type=int, default=None,
        help="Optional diagnostic override. Omit to use the benchmark's Phase-I sample size.",
    )
    parser.add_argument(
        "--solver", choices=("full", "covariance_eigh"), default="full",
        help="Use 'full' to match the benchmark exactly; covariance_eigh is faster for n >> p.",
    )
    parser.add_argument("--moving-window", type=int, default=50)
    args = parser.parse_args()
    if not args.targets or any(not 0.0 < value < 1.0 for value in args.targets):
        parser.error("every target must be strictly between 0 and 1")
    if args.fit_n is not None and args.fit_n < 2:
        parser.error("--fit-n must be at least 2")
    if args.moving_window < 2:
        parser.error("--moving-window must be at least 2")
    return args


def main() -> int:
    args = _parse_args()
    datasets_to_run = (
        ("cifar10", "patchcamelyon")
        if args.dataset == "both" else (args.dataset,)
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_root = output_dir / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for dataset in datasets_to_run:
        print(f"\nCalculating {dataset}, seed {args.seed} ...", flush=True)
        rows.extend(_rows_for_dataset(
            dataset=dataset,
            seed=args.seed,
            targets=args.targets,
            notebook_root=args.notebook_dir.expanduser().resolve(),
            runtime_root=runtime_root,
            fit_n=args.fit_n,
            solver=args.solver,
            moving_window=args.moving_window,
        ))

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "feature_representation", "target_variance_retention"]
    )
    targets_label = "_".join(f"{int(round(t * 100)):02d}" for t in sorted(set(args.targets)))
    output_path = output_dir / (
        f"pca_components_{args.dataset}_seed{args.seed}_targets_{targets_label}.csv"
    )
    result.to_csv(output_path, index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

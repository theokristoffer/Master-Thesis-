"""Observation-level sequential-bootstrap calibration for the static pipeline.

Execute this module after loading the matching CPV-based static-PCA notebook with
``DRIFT_RUN_FULL=0``.  It replaces only Phase-I threshold calibration and the
held-out ARL0 audit.  PCA fitting, reference construction, Phase-II episodes,
and result aggregation remain those of the source notebook.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


DIRECT_CALIBRATION_METHOD = "observation_level_sequential_bootstrap"


def observation_horizon(cfg) -> int:
    """Raw observations required for ``arl0_horizon_steps`` complete scores."""
    return int(cfg.window + (cfg.arl0_horizon_steps - 1) * cfg.stride)


def observation_bootstrap_plan(n_observations: int, episodes: int, cfg,
                               rng: np.random.Generator) -> np.ndarray:
    """Common IC observation streams sampled from the empirical IC pool."""
    if n_observations < 2:
        raise ValueError("The IC bootstrap pool must contain at least two observations")
    return rng.integers(
        0, n_observations,
        size=(int(episodes), observation_horizon(cfg)),
        dtype=np.int64,
    )


def _record_from_values_direct(values: np.ndarray, direction: str):
    transformed = np.asarray(values, dtype=np.float64)
    if direction == "low":
        transformed = -transformed
    running = np.maximum.accumulate(transformed)
    changed = np.flatnonzero(running > np.r_[-np.inf, running[:-1]])
    return running[changed], changed + 1


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return np.nan, np.nan
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, len(values) - 1))
    mean = float(values.mean())
    return mean - critical * se, mean + critical * se


def calibrate_direct_records(records, direction: str, cfg) -> Dict[str, Any]:
    """Select the chart limit by direct simulated zero-state run length."""
    nonempty = [values for values, _ in records if len(values)]
    if not nonempty:
        raise RuntimeError("No finite detector values were produced for calibration")
    candidates = np.unique(np.concatenate(nonempty))
    target = float(cfg.target_arl0_score_steps)
    horizon = int(cfg.arl0_horizon_steps)
    lo, hi, best = 0, len(candidates) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        limit = float(candidates[mid])
        lengths, observed = _run_lengths(records, limit, horizon)
        mean_steps = float(lengths.mean())
        item = (
            abs(math.log(mean_steps / target)), limit,
            lengths.copy(), observed.copy(),
        )
        if best is None or item[0] < best[0]:
            best = item
        if mean_steps < target:
            lo = mid + 1
        else:
            hi = mid - 1

    _, transformed_limit, lengths, observed = best
    threshold = transformed_limit if direction == "high" else -transformed_limit
    observations = proper_steps_to_observations(lengths, cfg)
    achieved = float(observations.mean())
    ci_low, ci_high = _mean_ci(observations)
    censor_rate = float((~observed).mean())
    return {
        "threshold": float(threshold),
        "direction": direction,
        "calibration_method": DIRECT_CALIBRATION_METHOD,
        "arl0_start": "zero_state",
        "bootstrap_unit": "in_control_observation",
        "cal_arl0_observations": achieved,
        "cal_arl0_score_steps": float(lengths.mean()),
        "cal_arl0_mc_ci_low": ci_low,
        "cal_arl0_mc_ci_high": ci_high,
        "cal_censor_rate": censor_rate,
        "cal_episodes": int(len(records)),
        "cal_horizon_observations": int(observation_horizon(cfg)),
        "estimate_type": "empirical_mean" if observed.all() else "restricted_mean",
        "in_band": bool(
            cfg.target_arl0_observations * (1 - cfg.arl0_tol)
            <= achieved
            <= cfg.target_arl0_observations * (1 + cfg.arl0_tol)
        ),
    }


def direct_record_bank(raw_features: Dict[str, np.ndarray],
                       vae_errors: Optional[np.ndarray], state: Dict[str, Any],
                       target_cpv: float, cfg, plan: np.ndarray,
                       progress_label: str = ""):
    """Recompute the complete detector path for each resampled IC stream."""
    records: Dict[Tuple[str, str, Optional[float]], list] = {}
    directions: Dict[Tuple[str, str, Optional[float]], str] = {}
    for episode, indices in enumerate(plan):
        episode_features = {
            block: np.asarray(values)[indices]
            for block, values in raw_features.items()
        }
        episode_vae = None if vae_errors is None else np.asarray(vae_errors)[indices]
        bundle = _score_bundle(
            episode_features, episode_vae, state, target_cpv, cfg
        )
        for key, (values, direction) in bundle.items():
            records.setdefault(key, []).append(
                _record_from_values_direct(values, direction)
            )
            directions[key] = direction
        if progress_label and (episode + 1) % max(1, min(25, len(plan))) == 0:
            print(
                f"{progress_label}: {episode + 1}/{len(plan)} direct IC streams",
                flush=True,
            )
    return records, directions


def _direct_state_metadata(seed: int, cfg) -> Dict[str, Any]:
    return {
        "version": "observation_level_sequential_calibration",
        "dataset": cfg.dataset_name,
        "seed": int(seed),
        "split_hash": SPLIT_HASH,
        "target_arl0_observations": float(cfg.target_arl0_observations),
        "window": int(cfg.window),
        "stride": int(cfg.stride),
        "pca_target_cpv": float(cfg.pca_retention),
        "ewma_lambdas": tuple(float(v) for v in cfg.ewma_lambda_grid),
        "methods": METHODS,
        "calibration_method": DIRECT_CALIBRATION_METHOD,
        "arl0_start": "zero_state",
        "bootstrap_unit": "in_control_observation",
        "calibration_episodes": int(cfg.arl0_cal_episodes),
        "calibration_horizon_steps": int(cfg.arl0_horizon_steps),
    }


def fit_and_calibrate_seed(seed: int, cfg, extractor):
    """Fit static Phase I and calibrate from direct bootstrap IC streams."""
    build_seed_partition(seed, cfg)
    outdir = Path(cfg.out_root) / "imagenet" / f"seed{seed}_{SPLIT_HASH}"
    outdir.mkdir(parents=True, exist_ok=True)
    state_path = outdir / "seed_state.pkl"
    metadata = _direct_state_metadata(seed, cfg)
    if state_path.exists() and not cfg.force_recompute:
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        if state.get("metadata") == metadata:
            state["outdir"] = str(outdir)
            _write_pca_dimension_audit(
                state, seed, SPLIT_HASH, cfg.dataset_name, outdir
            )
            rows = [{
                "seed": seed, "split_hash": SPLIT_HASH,
                **_pca_audit_fields(state, key[0], key[1]),
                "block": key[1], "method": key[2], "parameter": key[3],
                **value,
            } for key, value in state["thresholds"].items()]
            pd.DataFrame(rows).to_csv(outdir / "thresholds.csv", index=False)
            print(f"[seed {seed}] loaded direct-calibration state {state_path}")
            return state

    vae = train_seed_vae(seed, cfg)
    rng = np.random.default_rng(
        np.random.SeedSequence([cfg.stream_master_seed, seed, 20])
    )
    fit_ids = rng.permutation(TRAIN_IC_IDS)[:min(cfg.pca_fit_n, len(TRAIN_IC_IDS))]
    _, fit_features = _extract_features(fit_ids, rng, extractor, vae, augment=True)
    ref_ids = rng.choice(TRAIN_IC_IDS, size=cfg.ref_n, replace=False)
    _, ref_features = _extract_features(ref_ids, rng, extractor, vae, augment=True)

    pca_models, ref_proj = {}, {}
    for target in cfg.pca_dim_grid:
        pca_models[target], ref_proj[target] = {}, {}
        for block in FEATURE_BLOCKS:
            pca_models[target][block] = _fit_pca(fit_features[block], target)
            ref_proj[target][block] = pca_models[target][block].transform(
                ref_features[block]
            ).astype(np.float64)

    validation_ids = rng.permutation(VALIDATION_IC_IDS)
    validation_images = augment_test_batch(
        IMAGES[validation_ids], np.arange(len(validation_ids)),
        cfg.stream_master_seed + seed * 1000 + 21,
    )
    validation_features = extractor.extract(validation_images)
    validation_features["vae_latent"] = vae_latents(vae, validation_images)
    validation_vae = vae_recon_errors(vae, validation_images)
    state = {
        "metadata": metadata,
        "pca_models": pca_models,
        "ref_proj": ref_proj,
        "thresholds": {},
        "fit_ids": fit_ids,
        "ref_ids": ref_ids,
        "validation_ids": validation_ids,
        "outdir": str(outdir),
    }
    _write_pca_dimension_audit(
        state, seed, SPLIT_HASH, cfg.dataset_name, outdir
    )
    plan = observation_bootstrap_plan(
        len(validation_ids), cfg.arl0_cal_episodes, cfg,
        np.random.default_rng(
            np.random.SeedSequence([cfg.stream_master_seed, seed, 122])
        ),
    )
    for target in cfg.pca_dim_grid:
        records, directions = direct_record_bank(
            validation_features,
            validation_vae if target == cfg.pca_dim else None,
            state, target, cfg, plan,
            progress_label=f"[seed {seed}] calibration CPV={target:g}",
        )
        for key, detector_records in records.items():
            state["thresholds"][(target, *key)] = calibrate_direct_records(
                detector_records, directions[key], cfg
            )

    with state_path.open("wb") as handle:
        pickle.dump(state, handle)
    rows = [{
        "seed": seed, "split_hash": SPLIT_HASH,
        **_pca_audit_fields(state, key[0], key[1]),
        "block": key[1], "method": key[2], "parameter": key[3],
        **value,
    } for key, value in state["thresholds"].items()]
    pd.DataFrame(rows).to_csv(outdir / "thresholds.csv", index=False)
    return state


def verify_heldout_arl0(seed: int, cfg, extractor):
    """Audit fixed thresholds on direct bootstrap streams from held-out IC."""
    build_seed_partition(seed, cfg)
    state = fit_and_calibrate_seed(seed, cfg, extractor)
    vae = train_seed_vae(seed, cfg)
    path = Path(state["outdir"]) / "heldout_arl0_summary.csv"
    if path.exists() and not cfg.force_recompute:
        cached = pd.read_csv(path)
        if (
            len(cached) == len(state["thresholds"])
            and set(cached.get("calibration_method", [])) == {DIRECT_CALIBRATION_METHOD}
            and (cached.episodes.astype(int) >= cfg.arl0_test_episodes).all()
        ):
            return cached

    ids = np.random.default_rng(
        np.random.SeedSequence([cfg.stream_master_seed, seed, 140])
    ).permutation(TEST_IC_IDS)
    images = augment_test_batch(
        IMAGES[ids], np.arange(len(ids)),
        cfg.stream_master_seed + seed * 1000 + 140,
    )
    features = extractor.extract(images)
    features["vae_latent"] = vae_latents(vae, images)
    vae_errors = vae_recon_errors(vae, images)
    plan = observation_bootstrap_plan(
        len(ids), cfg.arl0_test_episodes, cfg,
        np.random.default_rng(
            np.random.SeedSequence([cfg.stream_master_seed, seed, 141])
        ),
    )
    rows = []
    for target in cfg.pca_dim_grid:
        records, directions = direct_record_bank(
            features, vae_errors if target == cfg.pca_dim else None,
            state, target, cfg, plan,
            progress_label=f"[seed {seed}] held-out CPV={target:g}",
        )
        for key, detector_records in records.items():
            block, method, parameter = key
            info = state["thresholds"][(target, block, method, parameter)]
            transformed = info["threshold"] if directions[key] == "high" else -info["threshold"]
            lengths, observed = _run_lengths(
                detector_records, transformed, cfg.arl0_horizon_steps
            )
            observations = proper_steps_to_observations(lengths, cfg)
            ci_low, ci_high = _mean_ci(observations)
            rows.append({
                "dataset": cfg.dataset_name,
                "seed": seed,
                "split_hash": SPLIT_HASH,
                **_pca_audit_fields(state, target, block),
                "block": block,
                "method": method,
                "ewma_lambda": parameter if method == "EWMA" else np.nan,
                "calibration_method": DIRECT_CALIBRATION_METHOD,
                "arl0_start": "zero_state",
                "bootstrap_unit": "heldout_in_control_observation",
                "episodes": int(len(lengths)),
                "events": int(observed.sum()),
                "censored": int((~observed).sum()),
                "censor_rate": float((~observed).mean()),
                "arl0_observations": float(observations.mean()),
                "arl0_score_steps": float(lengths.mean()),
                "arl0_mc_ci_low": ci_low,
                "arl0_mc_ci_high": ci_high,
                "target_arl0_observations": cfg.target_arl0_observations,
                "estimate_type": "empirical_mean" if observed.all() else "restricted_mean",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame

"""Reduced adaptive-PCA extension for the primary image-drift benchmark.

This file is executed by ``adaptive_pca_sensitivity_notebook.ipynb``
after that notebook loads the definitions from the selected primary benchmark notebook.
It deliberately reuses the primary dataset adapters, ResNet/VAE feature extraction,
drift injection, observation-scale timing, and paired episode manifests.

The extension adds:
  * a fixed-PCA extension baseline;
  * gated moving-window PCA with M in {250, 500};
  * gated recursive PCA with matched forgetting factors 1/M;
  * local subspace-distance monitoring for the moving-window models.

Every stateful method is recalibrated by replaying raw-feature moving-block
bootstrap episodes.  No validation/test observation refits the initial model,
and every bootstrap or drift episode starts from the same saved Phase-I state.
"""

from __future__ import annotations

import copy
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class AdaptivePCAConfig:
    feature_blocks: Tuple[str, ...] = ("layer1", "layer4", "vae_latent")
    detector_methods: Tuple[str, ...] = ("T2", "SPE", "EWMA", "MMD")
    pca_dim: int = 20
    moving_memories: Tuple[int, ...] = (250, 500)
    recursive_memories: Tuple[int, ...] = (250, 500)
    update_every: int = 25
    gate_quantile: float = 0.995
    ewma_lambda: float = 0.20
    calibration_episodes: int = 500
    heldout_episodes: int = 250
    bootstrap_block_observations: int = 100
    run_full: bool = False
    force_recompute: bool = False
    out_root: str = ""

    @classmethod
    def from_base(cls, cfg) -> "AdaptivePCAConfig":
        quick = bool(globals().get("QUICK", False))
        dataset = str(cfg.dataset_name)
        default_out = Path(globals()["RUN_BASE"]) / (
            f"outputs/{dataset}_adaptive_pca_sensitivity"
        )
        return cls(
            calibration_episodes=int(os.environ.get(
                "ADAPTIVE_ARL0_CAL_EPISODES", "4" if quick else "500"
            )),
            heldout_episodes=int(os.environ.get(
                "ADAPTIVE_ARL0_TEST_EPISODES", "4" if quick else "250"
            )),
            update_every=int(os.environ.get("ADAPTIVE_UPDATE_EVERY", "25")),
            run_full=os.environ.get("ADAPTIVE_RUN_FULL", "0") == "1",
            force_recompute=os.environ.get("ADAPTIVE_FORCE_RECOMPUTE", "0") == "1",
            out_root=str(Path(os.environ.get(
                "ADAPTIVE_OUT_ROOT", str(default_out)
            )).expanduser().absolute()),
        )


def strategy_grid(acfg: AdaptivePCAConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = [
        {"strategy": "fixed", "kind": "fixed", "memory": None,
         "forgetting_factor": None},
    ]
    rows.extend(
        {"strategy": f"moving_M{m}", "kind": "moving", "memory": int(m),
         "forgetting_factor": None}
        for m in acfg.moving_memories
    )
    rows.extend(
        {"strategy": f"recursive_M{m}", "kind": "recursive", "memory": int(m),
         "forgetting_factor": 1.0 / float(m)}
        for m in acfg.recursive_memories
    )
    return rows


def detector_keys(acfg: AdaptivePCAConfig) -> List[Tuple[str, str, str]]:
    keys: List[Tuple[str, str, str]] = []
    for block in acfg.feature_blocks:
        for spec in strategy_grid(acfg):
            for method in acfg.detector_methods:
                keys.append((block, spec["strategy"], method))
            if spec["kind"] == "moving":
                keys.append((block, spec["strategy"], "Subspace"))
    return keys


def _align_components(new_components: np.ndarray,
                      old_components: np.ndarray) -> np.ndarray:
    """Match component order and sign to the preceding online basis."""
    similarities = np.asarray(new_components) @ np.asarray(old_components).T
    new_ids, old_ids = linear_sum_assignment(-np.abs(similarities))
    aligned = np.empty_like(new_components)
    for new_id, old_id in zip(new_ids, old_ids):
        sign = 1.0 if similarities[new_id, old_id] >= 0 else -1.0
        aligned[old_id] = sign * new_components[new_id]
    return aligned


def _fit_basis(raw: np.ndarray, k: int,
               previous: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float64)
    if min(raw.shape) < k:
        raise ValueError(f"Cannot fit {k} PCs to array with shape {raw.shape}")
    fitted = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(raw)
    components = np.asarray(fitted.components_, dtype=np.float64)
    if previous is not None:
        components = _align_components(components, previous)
    return np.asarray(fitted.mean_, dtype=np.float64), components


def _top_covariance_basis(covariance: np.ndarray, k: int,
                          previous: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:k]].T
    return _align_components(components, previous)


def _subspace_distance(current: np.ndarray, baseline: np.ndarray) -> float:
    """Frobenius distance between two rank-k projection matrices."""
    overlap_sq = float(np.square(current @ baseline.T).sum())
    k = current.shape[0]
    return float(np.sqrt(max(0.0, 2.0 * k - 2.0 * overlap_sq)))


class OnlinePCAState:
    """Episode-local adaptive PCA state with score-then-update semantics."""

    def __init__(self, phase1: Dict[str, Any], strategy: Dict[str, Any],
                 acfg: AdaptivePCAConfig):
        self.kind = str(strategy["kind"])
        self.memory = strategy["memory"]
        self.rho = strategy["forgetting_factor"]
        self.update_every = int(acfg.update_every)
        self.k = int(acfg.pca_dim)
        self.initial_mean = np.asarray(phase1["initial_mean"], dtype=np.float64)
        self.initial_components = np.asarray(
            phase1["initial_components"], dtype=np.float64
        )
        self.mean = self.initial_mean.copy()
        self.components = self.initial_components.copy()
        self.version = 0
        self.accepted = 0
        self.accepted_since_refresh = 0

        self.buffer: Optional[List[np.ndarray]] = None
        self.running_mean: Optional[np.ndarray] = None
        self.running_covariance: Optional[np.ndarray] = None
        if self.kind == "moving":
            seed = np.asarray(phase1["moving_seed"], dtype=np.float64)
            self.buffer = [row.copy() for row in seed[-int(self.memory):]]
        elif self.kind == "recursive":
            self.running_mean = np.asarray(
                phase1["recursive_mean"], dtype=np.float64
            ).copy()
            self.running_covariance = np.asarray(
                phase1["recursive_covariance"], dtype=np.float64
            ).copy()

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=np.float64) - self.mean) @ self.components.T

    def spe(self, raw: np.ndarray) -> np.ndarray:
        centred = np.asarray(raw, dtype=np.float64) - self.mean
        projected = centred @ self.components.T
        return np.maximum(
            np.einsum("nd,nd->n", centred, centred)
            - np.einsum("nk,nk->n", projected, projected),
            0.0,
        )

    def subspace_distance(self) -> float:
        return _subspace_distance(self.components, self.initial_components)

    def accept(self, observation: np.ndarray) -> bool:
        """Accept one observation and refresh only at the declared cadence."""
        if self.kind == "fixed":
            return False
        x = np.asarray(observation, dtype=np.float64)
        self.accepted += 1
        self.accepted_since_refresh += 1
        if self.kind == "moving":
            assert self.buffer is not None
            self.buffer.append(x.copy())
            if len(self.buffer) > int(self.memory):
                self.buffer.pop(0)
        elif self.kind == "recursive":
            assert self.running_mean is not None
            assert self.running_covariance is not None
            rho = float(self.rho)
            delta = x - self.running_mean
            new_mean = self.running_mean + rho * delta
            self.running_covariance = (
                (1.0 - rho) * self.running_covariance
                + rho * np.outer(delta, x - new_mean)
            )
            self.running_mean = new_mean
        else:
            raise ValueError(self.kind)

        if self.accepted_since_refresh < self.update_every:
            return False
        self.accepted_since_refresh = 0
        if self.kind == "moving":
            raw = np.asarray(self.buffer, dtype=np.float64)
            self.mean, self.components = _fit_basis(
                raw, self.k, previous=self.components
            )
        else:
            self.mean = self.running_mean.copy()
            self.components = _top_covariance_basis(
                self.running_covariance, self.k, self.components
            )
        self.version += 1
        return True


def _phase1_metadata(seed: int, cfg, acfg: AdaptivePCAConfig) -> Dict[str, Any]:
    return {
        "version": "v9_adaptive_pca_reduced_1",
        "dataset": str(cfg.dataset_name),
        "seed": int(seed),
        "split_hash": str(globals()["SPLIT_HASH"]),
        "feature_blocks": tuple(acfg.feature_blocks),
        "pca_dim": int(acfg.pca_dim),
        "moving_memories": tuple(acfg.moving_memories),
        "recursive_memories": tuple(acfg.recursive_memories),
        "update_every": int(acfg.update_every),
        "gate_quantile": float(acfg.gate_quantile),
        "reference_n": int(cfg.ref_n),
    }


def _fixed_gate_limits(initial_mean: np.ndarray,
                       initial_components: np.ndarray,
                       raw_reference: np.ndarray,
                       validation_raw: np.ndarray,
                       cfg, acfg: AdaptivePCAConfig) -> Dict[str, float]:
    reference_projection = (
        np.asarray(raw_reference, dtype=np.float64) - initial_mean
    ) @ initial_components.T
    validation_projection = (
        np.asarray(validation_raw, dtype=np.float64) - initial_mean
    ) @ initial_components.T
    prepared = prepare_reference(reference_projection, cfg)
    windows = make_windows(validation_projection, cfg.window, cfg.stride)
    t2 = gauss_moment_stats(prepared, windows)["T2"]
    centred = np.asarray(validation_raw, dtype=np.float64) - initial_mean
    per_observation_spe = np.maximum(
        np.einsum("nd,nd->n", centred, centred)
        - np.einsum("nk,nk->n", validation_projection, validation_projection),
        0.0,
    )
    spe = window_means(per_observation_spe, cfg)
    return {
        "T2": float(np.quantile(t2, acfg.gate_quantile)),
        "SPE": float(np.quantile(spe, acfg.gate_quantile)),
    }


def build_adaptive_phase1(seed: int, cfg, acfg: AdaptivePCAConfig,
                          extractor) -> Dict[str, Any]:
    """Fit/cache extension Phase-I objects and validation feature banks."""
    build_seed_partition(seed, cfg)
    seed_root = Path(acfg.out_root) / "phase1" / f"seed{seed}_{SPLIT_HASH}"
    seed_root.mkdir(parents=True, exist_ok=True)
    path = seed_root / "adaptive_phase1.pkl"
    metadata = _phase1_metadata(seed, cfg, acfg)
    if path.exists() and not acfg.force_recompute:
        with path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("metadata") == metadata:
            return cached

    vae = train_seed_vae(seed, cfg)
    rng = np.random.default_rng(np.random.SeedSequence([
        cfg.stream_master_seed, seed, 60
    ]))
    fit_ids = rng.permutation(TRAIN_IC_IDS)[
        :min(cfg.pca_fit_n, len(TRAIN_IC_IDS))
    ]
    _, fit_features_all = _extract_features(
        fit_ids, rng, extractor, vae, augment=True
    )
    reference_ids = rng.choice(
        TRAIN_IC_IDS, size=cfg.ref_n, replace=False
    )
    _, reference_features_all = _extract_features(
        reference_ids, rng, extractor, vae, augment=True
    )
    validation_ids = rng.permutation(VALIDATION_IC_IDS)
    validation_images = augment_test_batch(
        IMAGES[validation_ids], np.arange(len(validation_ids)),
        cfg.stream_master_seed + seed * 1000 + 61,
    )
    validation_features_all = extractor.extract(validation_images)
    validation_features_all["vae_latent"] = vae_latents(
        vae, validation_images
    )

    max_memory = max((*acfg.moving_memories, *acfg.recursive_memories))
    blocks: Dict[str, Any] = {}
    for block in acfg.feature_blocks:
        fit_raw = np.asarray(fit_features_all[block], dtype=np.float64)
        reference_raw = np.asarray(
            reference_features_all[block], dtype=np.float64
        )
        validation_raw = np.asarray(
            validation_features_all[block], dtype=np.float64
        )
        initial_mean, initial_components = _fit_basis(
            fit_raw, acfg.pca_dim
        )
        recursive_covariance = np.cov(fit_raw, rowvar=False, ddof=1)
        recursive_covariance = np.asarray(
            recursive_covariance, dtype=np.float64
        )
        recursive_covariance += np.eye(recursive_covariance.shape[0]) * 1e-9
        gate_limits = _fixed_gate_limits(
            initial_mean, initial_components, reference_raw,
            validation_raw, cfg, acfg,
        )
        blocks[block] = {
            "initial_mean": initial_mean,
            "initial_components": initial_components,
            "raw_reference": reference_raw,
            "moving_seed": fit_raw[-max_memory:].copy(),
            "recursive_mean": fit_raw.mean(axis=0),
            "recursive_covariance": recursive_covariance,
            "gate_limits": gate_limits,
            "validation_raw": validation_raw,
        }
    payload = {
        "metadata": metadata,
        "fit_ids": np.asarray(fit_ids, dtype=np.int64),
        "reference_ids": np.asarray(reference_ids, dtype=np.int64),
        "validation_ids": np.asarray(validation_ids, dtype=np.int64),
        "blocks": blocks,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


class RollingMMD:
    """Exact stride-one MMD update while the active PCA basis is unchanged."""

    def __init__(self):
        self.version = -1
        self.window = None
        self.phi = None
        self.pair_sum = None

    @staticmethod
    def _kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
        distance = np.maximum(
            np.square(a).sum(axis=-1, keepdims=True)
            + np.square(b).sum(axis=-1)[None, :]
            - 2.0 * np.asarray(a) @ np.asarray(b).T,
            0.0,
        )
        return np.exp(-gamma * distance)

    def score(self, projected_window: np.ndarray,
              prepared_reference: Dict[str, Any], version: int) -> float:
        window = np.asarray(projected_window, dtype=np.float64)
        reference = prepared_reference["ref"]
        gamma = float(prepared_reference["gamma"])
        if self.window is None or self.version != version:
            cross = self._kernel(window, reference, gamma)
            self.phi = cross.mean(axis=1)
            within = self._kernel(window, window, gamma)
            self.pair_sum = float(np.triu(within, k=1).sum())
        else:
            outgoing = self.window[:1]
            remaining_old = self.window[1:]
            incoming = window[-1:]
            remaining_new = window[:-1]
            removed = float(self._kernel(outgoing, remaining_old, gamma).sum())
            added = float(self._kernel(incoming, remaining_new, gamma).sum())
            self.pair_sum += added - removed
            incoming_phi = float(self._kernel(incoming, reference, gamma).mean())
            self.phi = np.r_[self.phi[1:], incoming_phi]
        self.window = window.copy()
        self.version = int(version)
        m = len(window)
        term_ww = 2.0 * self.pair_sum / (m * (m - 1))
        term_rw = float(self.phi.mean())
        return float(prepared_reference["term_rr"] + term_ww - 2.0 * term_rw)


def _score_one_window(state: OnlinePCAState, phase1_block: Dict[str, Any],
                      raw_window: np.ndarray, prepared_reference: Dict[str, Any],
                      ewma_state: np.ndarray, mmd_tracker: RollingMMD, cfg,
                      acfg: AdaptivePCAConfig) -> Tuple[Dict[str, float], np.ndarray]:
    projected = state.transform(raw_window)
    difference = projected.mean(axis=0) - prepared_reference["mu"]
    t2 = float(
        cfg.window
        * difference @ prepared_reference["cov_inv"] @ difference
    )
    spe = float(state.spe(raw_window).mean())
    mmd = mmd_tracker.score(projected, prepared_reference, state.version)
    mean = projected.mean(axis=0)
    standardized = (
        (mean - prepared_reference["mu_marg"])
        / (prepared_reference["sd_marg"] / np.sqrt(cfg.window))
    )
    ewma_state = (
        acfg.ewma_lambda * standardized
        + (1.0 - acfg.ewma_lambda) * ewma_state
    )
    ewma = float(
        np.square(ewma_state).sum()
        * (2.0 - acfg.ewma_lambda) / acfg.ewma_lambda
    )
    return {"T2": t2, "SPE": spe, "MMD": mmd, "EWMA": ewma}, ewma_state


def score_adaptive_sequence(raw_sequence: np.ndarray,
                            phase1_block: Dict[str, Any],
                            strategy: Dict[str, Any], cfg,
                            acfg: AdaptivePCAConfig) -> Dict[str, np.ndarray]:
    """Score one raw-feature episode, resetting all state at its start."""
    raw_sequence = np.asarray(raw_sequence, dtype=np.float64)
    state = OnlinePCAState(phase1_block, strategy, acfg)
    scores = {method: [] for method in acfg.detector_methods}
    scores["Subspace"] = []
    gate_accepted: List[bool] = []
    basis_distances: List[float] = []
    ewma_state = np.zeros(acfg.pca_dim, dtype=np.float64)
    mmd_tracker = RollingMMD()
    prepared_reference = None
    prepared_version = -1

    for t, observation in enumerate(raw_sequence):
        if t < cfg.window - 1:
            state.accept(observation)
            continue
        if prepared_reference is None or prepared_version != state.version:
            reference_projection = state.transform(
                phase1_block["raw_reference"]
            )
            prepared_reference = prepare_reference(reference_projection, cfg)
            prepared_version = state.version
        raw_window = raw_sequence[t - cfg.window + 1:t + 1]
        current, ewma_state = _score_one_window(
            state, phase1_block, raw_window, prepared_reference,
            ewma_state, mmd_tracker, cfg, acfg,
        )
        for method in acfg.detector_methods:
            scores[method].append(current[method])
        scores["Subspace"].append(state.subspace_distance())
        gate_ok = (
            current["T2"] <= phase1_block["gate_limits"]["T2"]
            and current["SPE"] <= phase1_block["gate_limits"]["SPE"]
        )
        gate_accepted.append(bool(gate_ok))
        basis_distances.append(state.subspace_distance())
        if gate_ok:
            state.accept(observation)

    result = {
        method: np.asarray(values, dtype=np.float64)
        for method, values in scores.items()
    }
    result["gate_accepted"] = np.asarray(gate_accepted, dtype=bool)
    result["basis_distance"] = np.asarray(basis_distances, dtype=np.float64)
    return result


def _bootstrap_observation_indices(n_observations: int, n_episodes: int,
                                   episode_length: int, block_length: int,
                                   rng: np.random.Generator) -> np.ndarray:
    if n_observations < block_length:
        raise ValueError("Feature bank is shorter than one bootstrap block")
    n_blocks = int(np.ceil(episode_length / block_length))
    starts = rng.integers(
        0, n_observations - block_length + 1,
        size=(n_episodes, n_blocks),
    )
    output = np.empty((n_episodes, episode_length), dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    for episode, row in enumerate(starts):
        output[episode] = np.concatenate([
            start + offsets for start in row
        ])[:episode_length]
    return output


def _running_record(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    running = np.maximum.accumulate(scores)
    changed = np.flatnonzero(running > np.r_[-np.inf, running[:-1]])
    return running[changed], changed + 1


def calibrate_adaptive_seed(seed: int, cfg, acfg: AdaptivePCAConfig,
                            extractor, phase1: Dict[str, Any]) -> Dict[str, Any]:
    seed_root = Path(acfg.out_root) / "phase1" / f"seed{seed}_{SPLIT_HASH}"
    threshold_path = seed_root / "adaptive_thresholds.pkl"
    csv_path = seed_root / "adaptive_thresholds.csv"
    metadata = {
        **_phase1_metadata(seed, cfg, acfg),
        "calibration_episodes": int(acfg.calibration_episodes),
        "bootstrap_block_observations": int(acfg.bootstrap_block_observations),
        "target_arl0_observations": float(cfg.target_arl0_observations),
    }
    if threshold_path.exists() and not acfg.force_recompute:
        with threshold_path.open("rb") as handle:
            cached = pickle.load(handle)
        cached_metadata = cached.get("metadata", {})
        # Episode counts are a minimum precision request, not part of the
        # scientific definition of a threshold. A completed larger run (for
        # example the previous 1,000-episode setting) is therefore fully
        # compatible with the new 500-episode default. All other Phase-I,
        # strategy, block-bootstrap, and ARL0 settings must still match.
        requested_without_count = {
            key: value for key, value in metadata.items()
            if key != "calibration_episodes"
        }
        cached_without_count = {
            key: value for key, value in cached_metadata.items()
            if key != "calibration_episodes"
        }
        cached_episodes = int(cached_metadata.get("calibration_episodes", 0))
        if (
            cached_without_count == requested_without_count
            and cached_episodes >= acfg.calibration_episodes
            and set(cached.get("thresholds", {})) == set(detector_keys(acfg))
        ):
            if cached_episodes > acfg.calibration_episodes:
                print(
                    f"[seed {seed}] reusing stronger {cached_episodes}-episode "
                    f"calibration cache (requested {acfg.calibration_episodes})"
                )
            return cached

    episode_length = cfg.window - 1 + cfg.arl0_horizon_steps
    n_validation = len(phase1["validation_ids"])
    plan = _bootstrap_observation_indices(
        n_validation, acfg.calibration_episodes, episode_length,
        acfg.bootstrap_block_observations,
        np.random.default_rng(np.random.SeedSequence([
            cfg.stream_master_seed, seed, 70
        ])),
    )
    records: Dict[Tuple[str, str, str], List[Tuple[np.ndarray, np.ndarray]]] = {
        key: [] for key in detector_keys(acfg)
    }
    for block in acfg.feature_blocks:
        bank = phase1["blocks"][block]["validation_raw"]
        for episode, indices in enumerate(plan):
            raw_episode = bank[indices]
            for spec in strategy_grid(acfg):
                traces = score_adaptive_sequence(
                    raw_episode, phase1["blocks"][block], spec, cfg, acfg
                )
                for method in acfg.detector_methods:
                    records[(block, spec["strategy"], method)].append(
                        _running_record(traces[method])
                    )
                if spec["kind"] == "moving":
                    records[(block, spec["strategy"], "Subspace")].append(
                        _running_record(traces["Subspace"])
                    )
            if (episode + 1) % max(1, acfg.calibration_episodes // 10) == 0:
                print(
                    f"[seed {seed}] calibration {block}: "
                    f"{episode + 1}/{acfg.calibration_episodes} episodes"
                )

    thresholds = {
        key: calibrate_records(value, "high", cfg)
        for key, value in records.items()
    }
    payload = {"metadata": metadata, "thresholds": thresholds}
    with threshold_path.open("wb") as handle:
        pickle.dump(payload, handle)
    rows = []
    specs = {row["strategy"]: row for row in strategy_grid(acfg)}
    for (block, strategy, method), information in thresholds.items():
        rows.append({
            "dataset": cfg.dataset_name, "seed": seed,
            "split_hash": SPLIT_HASH, "block": block,
            "strategy": strategy, "method": method,
            "memory": specs[strategy]["memory"],
            "forgetting_factor": specs[strategy]["forgetting_factor"],
            **information,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return payload


def _extract_feature_bank(ids: np.ndarray, seed: int, purpose: int,
                          cfg, acfg: AdaptivePCAConfig, extractor,
                          vae) -> Dict[str, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    images = augment_test_batch(
        IMAGES[ids], np.arange(len(ids)),
        cfg.stream_master_seed + seed * 1000 + purpose,
    )
    features = extractor.extract(images)
    features["vae_latent"] = vae_latents(vae, images)
    return {
        block: np.asarray(features[block], dtype=np.float64)
        for block in acfg.feature_blocks
    }


def _first_alarm_length(scores: np.ndarray, threshold: float,
                        horizon: int) -> Tuple[int, bool]:
    hit = first_alarm(scores, threshold, "high")
    return (horizon, False) if hit is None else (int(hit + 1), True)


def verify_adaptive_heldout(seed: int, cfg, acfg: AdaptivePCAConfig,
                            extractor, phase1: Dict[str, Any],
                            calibration: Dict[str, Any]) -> pd.DataFrame:
    seed_root = Path(acfg.out_root) / "heldout" / f"seed{seed}_{SPLIT_HASH}"
    seed_root.mkdir(parents=True, exist_ok=True)
    path = seed_root / "heldout_arl0_summary.csv"
    alarm_path = seed_root / "heldout_alarm_lengths.npz"
    if path.exists() and not acfg.force_recompute:
        cached = pd.read_csv(path)
        compatible = (
            len(cached) == len(detector_keys(acfg))
            and "episodes" in cached
            and (cached["episodes"].astype(int) >= acfg.heldout_episodes).all()
            and set(cached["seed"].astype(int)) == {int(seed)}
            and set(cached["split_hash"].astype(str)) == {str(SPLIT_HASH)}
        )
        if compatible:
            completed_episodes = int(cached["episodes"].astype(int).min())
            if completed_episodes > acfg.heldout_episodes:
                print(
                    f"[seed {seed}] reusing stronger {completed_episodes}-episode "
                    f"held-out audit (requested {acfg.heldout_episodes})"
                )
            return cached

    build_seed_partition(seed, cfg)
    vae = train_seed_vae(seed, cfg)
    ids = np.random.default_rng(np.random.SeedSequence([
        cfg.stream_master_seed, seed, 80
    ])).permutation(TEST_IC_IDS)
    banks = _extract_feature_bank(
        ids, seed, 80, cfg, acfg, extractor, vae
    )
    episode_length = cfg.window - 1 + cfg.arl0_horizon_steps
    plan = _bootstrap_observation_indices(
        len(ids), acfg.heldout_episodes, episode_length,
        acfg.bootstrap_block_observations,
        np.random.default_rng(np.random.SeedSequence([
            cfg.stream_master_seed, seed, 81
        ])),
    )
    lengths: Dict[Tuple[str, str, str], List[int]] = {
        key: [] for key in detector_keys(acfg)
    }
    events: Dict[Tuple[str, str, str], List[bool]] = {
        key: [] for key in detector_keys(acfg)
    }
    for block in acfg.feature_blocks:
        for episode, indices in enumerate(plan):
            raw_episode = banks[block][indices]
            for spec in strategy_grid(acfg):
                traces = score_adaptive_sequence(
                    raw_episode, phase1["blocks"][block], spec, cfg, acfg
                )
                methods = list(acfg.detector_methods)
                if spec["kind"] == "moving":
                    methods.append("Subspace")
                for method in methods:
                    key = (block, spec["strategy"], method)
                    threshold = calibration["thresholds"][key]["threshold"]
                    length, event = _first_alarm_length(
                        traces[method], threshold, cfg.arl0_horizon_steps
                    )
                    lengths[key].append(length)
                    events[key].append(event)
            if (episode + 1) % max(1, acfg.heldout_episodes // 10) == 0:
                print(
                    f"[seed {seed}] held-out {block}: "
                    f"{episode + 1}/{acfg.heldout_episodes} episodes"
                )

    specs = {row["strategy"]: row for row in strategy_grid(acfg)}
    rows = []
    alarm_arrays = {}
    for key in detector_keys(acfg):
        block, strategy, method = key
        run_lengths = np.asarray(lengths[key], dtype=np.int64)
        observed = np.asarray(events[key], dtype=bool)
        observations = proper_steps_to_observations(run_lengths, cfg)
        rows.append({
            "dataset": cfg.dataset_name, "seed": seed,
            "split_hash": SPLIT_HASH, "block": block,
            "strategy": strategy, "method": method,
            "memory": specs[strategy]["memory"],
            "forgetting_factor": specs[strategy]["forgetting_factor"],
            "episodes": len(run_lengths), "events": int(observed.sum()),
            "censored": int((~observed).sum()),
            "censor_rate": float((~observed).mean()),
            "arl0_observations": float(observations.mean()),
            "arl0_score_steps": float(run_lengths.mean()),
            "target_arl0_observations": cfg.target_arl0_observations,
            "estimate_type": (
                "empirical_mean" if observed.all() else "restricted_mean"
            ),
        })
        alarm_arrays["__".join(key)] = run_lengths
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    np.savez_compressed(alarm_path, **alarm_arrays)
    return frame


def _episode_feature_blocks(images: np.ndarray, vae,
                            acfg: AdaptivePCAConfig) -> Dict[str, np.ndarray]:
    features = EXTRACTOR.extract(images)
    features["vae_latent"] = vae_latents(vae, images)
    return {
        block: np.asarray(features[block], dtype=np.float64)
        for block in acfg.feature_blocks
    }


def evaluate_adaptive_seed(seed: int, cfg, acfg: AdaptivePCAConfig,
                           extractor, phase1: Dict[str, Any],
                           calibration: Dict[str, Any]) -> pd.DataFrame:
    build_seed_partition(seed, cfg)
    vae = train_seed_vae(seed, cfg)
    ic_orders, ooc_orders = master_episode_manifest(seed, cfg)
    seed_root = Path(acfg.out_root) / "drift" / f"seed{seed}_{SPLIT_HASH}"
    seed_root.mkdir(parents=True, exist_ok=True)
    frames = []
    specs = {row["strategy"]: row for row in strategy_grid(acfg)}

    for pattern in cfg.patterns:
        for mechanism in cfg.mechanisms:
            for severity in cfg.severity_levels:
                path = seed_root / f"{pattern}_{mechanism}_s{severity:.2f}.csv"
                if path.exists() and not acfg.force_recompute:
                    cached = pd.read_csv(path)
                    expected = cfg.mc_arl1_reps * len(detector_keys(acfg))
                    if len(cached) == expected:
                        frames.append(cached)
                        continue
                rows = []
                for replication in range(cfg.mc_arl1_reps):
                    images, observed_ids = build_episode(
                        seed, replication, pattern, mechanism, severity,
                        ic_orders[replication], ooc_orders[replication], cfg,
                    )
                    blocks = _episode_feature_blocks(images, vae, acfg)
                    for block in acfg.feature_blocks:
                        for spec in strategy_grid(acfg):
                            traces = score_adaptive_sequence(
                                blocks[block], phase1["blocks"][block],
                                spec, cfg, acfg,
                            )
                            methods = list(acfg.detector_methods)
                            if spec["kind"] == "moving":
                                methods.append("Subspace")
                            for method in methods:
                                key = (block, spec["strategy"], method)
                                threshold = calibration["thresholds"][key]["threshold"]
                                hit = first_alarm(traces[method], threshold, "high")
                                detected = hit is not None
                                delay = float(hit + 1) if detected else np.nan
                                endpoint = int(hit) if detected else len(traces[method]) - 1
                                gate_prefix = traces["gate_accepted"][:endpoint + 1]
                                basis_prefix = traces["basis_distance"][:endpoint + 1]
                                rows.append({
                                    "dataset": cfg.dataset_name,
                                    "seed": seed, "split_hash": SPLIT_HASH,
                                    "replication": replication,
                                    "pattern": pattern, "mechanism": mechanism,
                                    "severity": float(severity),
                                    "block": block, "strategy": spec["strategy"],
                                    "strategy_kind": spec["kind"],
                                    "memory": specs[spec["strategy"]]["memory"],
                                    "forgetting_factor": specs[spec["strategy"]]["forgetting_factor"],
                                    "method": method,
                                    "detected": bool(detected),
                                    "censored": bool(not detected),
                                    "alarm_score_index": hit if detected else np.nan,
                                    "alarm_observation": (
                                        cfg.changepoint + hit if detected else np.nan
                                    ),
                                    "arl1_delay_observations": delay,
                                    "restricted_delay_observations": (
                                        delay if detected else cfg.postchange_horizon
                                    ),
                                    "gate_acceptance_rate_to_stop": (
                                        float(gate_prefix.mean())
                                        if len(gate_prefix) else np.nan
                                    ),
                                    "basis_distance_at_stop": (
                                        float(basis_prefix[-1])
                                        if len(basis_prefix) else 0.0
                                    ),
                                    "cal_arl0_observations": calibration[
                                        "thresholds"
                                    ][key]["cal_arl0_observations"],
                                    "unique_identities": len(np.unique(observed_ids)),
                                })
                frame = pd.DataFrame(rows)
                frame.to_csv(path, index=False)
                frames.append(frame)
                print(
                    f"[seed {seed}] adaptive {pattern}/{mechanism}/"
                    f"s={severity:g} complete"
                )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(seed_root / "adaptive_episode_results.csv", index=False)
    return result


def aggregate_adaptive_results(results: pd.DataFrame,
                               heldout: pd.DataFrame,
                               cfg, acfg: AdaptivePCAConfig) -> Dict[str, pd.DataFrame]:
    aggregate = Path(acfg.out_root) / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    results.to_csv(aggregate / "adaptive_episode_results_all_seeds.csv", index=False)
    heldout.to_csv(aggregate / "adaptive_heldout_arl0_all_seeds.csv", index=False)

    condition_columns = [
        "dataset", "seed", "pattern", "mechanism", "severity",
        "block", "strategy", "strategy_kind", "memory",
        "forgetting_factor", "method",
    ]
    seed_level = results.groupby(condition_columns, dropna=False).agg(
        restricted_delay=("restricted_delay_observations", "mean"),
        detection_rate=("detected", "mean"),
        gate_acceptance_rate=("gate_acceptance_rate_to_stop", "mean"),
        basis_distance=("basis_distance_at_stop", "mean"),
        episodes=("replication", "size"),
    ).reset_index()
    seed_level.to_csv(aggregate / "adaptive_seed_level.csv", index=False)

    group_columns = [column for column in condition_columns if column != "seed"]
    rows = []
    for key, group in seed_level.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        values = group["restricted_delay"].to_numpy(dtype=float)
        n = len(values)
        se = values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        critical = scipy_stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        row.update({
            "mean_restricted_delay": float(values.mean()),
            "ci_low": float(values.mean() - critical * se) if n > 1 else np.nan,
            "ci_high": float(values.mean() + critical * se) if n > 1 else np.nan,
            "mean_detection_rate": float(group["detection_rate"].mean()),
            "mean_gate_acceptance_rate": float(group["gate_acceptance_rate"].mean()),
            "mean_basis_distance_at_stop": float(group["basis_distance"].mean()),
            "seeds": n,
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(aggregate / "adaptive_condition_summary.csv", index=False)

    fixed = summary[summary.strategy == "fixed"].copy()
    fixed = fixed.rename(columns={
        "mean_restricted_delay": "fixed_delay",
        "mean_detection_rate": "fixed_detection_rate",
    })
    adaptive = summary[summary.strategy != "fixed"].copy()
    merge_keys = ["dataset", "pattern", "mechanism", "severity", "block", "method"]
    comparison = adaptive.merge(
        fixed[merge_keys + ["fixed_delay", "fixed_detection_rate"]],
        on=merge_keys, how="left",
    )
    comparison["delay_ratio_vs_fixed"] = (
        comparison["mean_restricted_delay"] / comparison["fixed_delay"]
    )
    comparison["delay_difference_vs_fixed"] = (
        comparison["mean_restricted_delay"] - comparison["fixed_delay"]
    )
    comparison["detection_rate_difference_vs_fixed"] = (
        comparison["mean_detection_rate"] - comparison["fixed_detection_rate"]
    )
    comparison.to_csv(aggregate / "adaptive_vs_fixed_comparison.csv", index=False)

    heldout_summary = heldout.groupby(
        ["dataset", "block", "strategy", "memory", "forgetting_factor", "method"],
        dropna=False,
    ).agg(
        mean_heldout_arl0=("arl0_observations", "mean"),
        mean_censor_rate=("censor_rate", "mean"),
        seeds=("seed", "nunique"),
    ).reset_index()
    heldout_summary.to_csv(
        aggregate / "adaptive_heldout_arl0_summary.csv", index=False
    )
    return {
        "episode_results": results,
        "heldout": heldout,
        "seed_level": seed_level,
        "summary": summary,
        "comparison": comparison,
        "heldout_summary": heldout_summary,
    }


def run_adaptive_extension(cfg, acfg: AdaptivePCAConfig, extractor):
    Path(acfg.out_root).mkdir(parents=True, exist_ok=True)
    all_results, all_heldout = [], []
    for seed in cfg.seeds:
        print(f"\n=== adaptive PCA seed {seed} ({cfg.dataset_name}) ===")
        phase1 = build_adaptive_phase1(seed, cfg, acfg, extractor)
        calibration = calibrate_adaptive_seed(
            seed, cfg, acfg, extractor, phase1
        )
        all_heldout.append(verify_adaptive_heldout(
            seed, cfg, acfg, extractor, phase1, calibration
        ))
        all_results.append(evaluate_adaptive_seed(
            seed, cfg, acfg, extractor, phase1, calibration
        ))
    results = pd.concat(all_results, ignore_index=True)
    heldout = pd.concat(all_heldout, ignore_index=True)
    return aggregate_adaptive_results(results, heldout, cfg, acfg)


def adaptive_design_audit(cfg, acfg: AdaptivePCAConfig) -> pd.DataFrame:
    rows = strategy_grid(acfg)
    audit = pd.DataFrame(rows)
    audit["pca_dim"] = acfg.pca_dim
    audit["detector_window"] = cfg.window
    audit["update_every"] = acfg.update_every
    audit["gate_quantile"] = acfg.gate_quantile
    audit["feature_blocks"] = ", ".join(acfg.feature_blocks)
    audit["detectors"] = ", ".join(acfg.detector_methods)
    audit["subspace_detector"] = audit["kind"].eq("moving")
    return audit


# Lightweight mathematical checks that do not touch the image datasets.
_audit_rng = np.random.default_rng(101)
_audit_raw = _audit_rng.normal(size=(80, 30))
_audit_mean, _audit_components = _fit_basis(_audit_raw, 5)
assert _audit_components.shape == (5, 30)
assert np.allclose(_audit_components @ _audit_components.T, np.eye(5), atol=1e-8)
assert abs(_subspace_distance(_audit_components, _audit_components)) < 1e-6

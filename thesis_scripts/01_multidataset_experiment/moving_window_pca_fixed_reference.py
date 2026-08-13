"""Fixed-reference moving-window PCA extension for the primary image benchmark.

This module is executed by either moving-window dataset notebook after the corresponding
primary benchmark notebook has loaded its shared data, feature, drift, and ARL helpers.  It
implements an ungated, stride-one PCA fit on every complete 50-observation
window.  The local rank-k subspace is aligned to the frozen Phase-I subspace
before distributional monitoring.

The implementation is intentionally a moving-window extension, not an exact
reproduction of De Ketelaere, Hubert, and Schmitt (2015): the current
observation is included in the fitted window, the update is ungated, and the
local subspace is compared with a fixed Phase-I reference.  Their reviewed
MWPCA formulation normally scores a new observation with the preceding model
and withholds an update after an out-of-control result.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.covariance import ledoit_wolf


@dataclass(frozen=True)
class MovingWindowPCAConfig:
    feature_blocks: Tuple[str, ...] = ("layer1", "layer4", "vae_latent")
    detector_methods: Tuple[str, ...] = (
        "Subspace", "T2", "Gaussian_KL", "Gaussian_Hellinger", "MMD", "MEWMA"
    )
    pca_dim: int = 20
    window: int = 50
    update_every: int = 1
    ewma_lambda: float = 0.20
    calibration_episodes: int = 500
    heldout_episodes: int = 250
    bootstrap_block_steps: int = 100
    score_bank_observations: int = 10_000
    reference_n: int = 500
    mmd_reference_n: int = 100
    covariance_ridge: float = 1e-6
    progress_every_windows: int = 500
    run_full: bool = False
    force_recompute: bool = False
    out_root: str = ""

    @classmethod
    def from_base(cls, cfg) -> "MovingWindowPCAConfig":
        quick = bool(globals().get("QUICK", False))
        dataset = str(cfg.dataset_name)
        default_out = Path(globals()["RUN_BASE"]) / (
            f"outputs/{dataset}_moving_window_pca"
        )
        return cls(
            window=int(cfg.window),
            pca_dim=20,
            ewma_lambda=float(cfg.ewma_lambda),
            calibration_episodes=int(os.environ.get(
                "MWPCA_ARL0_CAL_EPISODES", "8" if quick else "500"
            )),
            heldout_episodes=int(os.environ.get(
                "MWPCA_ARL0_TEST_EPISODES", "8" if quick else "250"
            )),
            score_bank_observations=int(os.environ.get(
                "MWPCA_SCORE_BANK_OBSERVATIONS", "400" if quick else "10000"
            )),
            run_full=os.environ.get("MWPCA_RUN_FULL", "0") == "1",
            force_recompute=os.environ.get("MWPCA_FORCE_RECOMPUTE", "0") == "1",
            out_root=str(Path(os.environ.get(
                "MWPCA_OUT_ROOT", str(default_out)
            )).expanduser().absolute()),
        )


def _metadata(seed: int, cfg, mcfg: MovingWindowPCAConfig) -> Dict[str, Any]:
    return {
        "version": "v11_fixed_reference_mwpca_1",
        "dataset": str(cfg.dataset_name),
        "seed": int(seed),
        "split_hash": str(globals()["SPLIT_HASH"]),
        "feature_blocks": tuple(mcfg.feature_blocks),
        "detector_methods": tuple(mcfg.detector_methods),
        "pca_dim": int(mcfg.pca_dim),
        "window": int(mcfg.window),
        "stride": int(cfg.stride),
        "update_every": int(mcfg.update_every),
        "ewma_lambda": float(mcfg.ewma_lambda),
        "reference_n": int(mcfg.reference_n),
        "mmd_reference_n": int(mcfg.mmd_reference_n),
        "score_bank_observations": int(mcfg.score_bank_observations),
        "bootstrap_block_steps": int(mcfg.bootstrap_block_steps),
        "target_arl0_observations": float(cfg.target_arl0_observations),
    }


def _primary_state_candidates(seed: int, cfg) -> List[Path]:
    relative = Path("imagenet") / f"seed{seed}_{SPLIT_HASH}" / "seed_state.pkl"
    return [Path(cfg.out_root) / relative, *[
        Path(root) / relative for root in cfg.legacy_out_roots
    ]]


def load_primary_state_only(seed: int, cfg) -> Dict[str, Any]:
    """Load the fitted static Phase-I state; never trigger a refit."""
    build_seed_partition(seed, cfg)
    rejected: List[str] = []
    for path in _primary_state_candidates(seed, cfg):
        if not path.exists():
            continue
        with path.open("rb") as handle:
            state = pickle.load(handle)
        if (
            state.get("metadata", {}).get("dataset") == cfg.dataset_name
            and state.get("metadata", {}).get("seed") == seed
            and state.get("metadata", {}).get("split_hash") == SPLIT_HASH
            and mcfg_pca_available(state, 20)
        ):
            print(f"[seed {seed}] loaded primary Phase-I state {path}")
            return state
        rejected.append(str(path))
    detail = f" Rejected: {rejected}" if rejected else ""
    raise FileNotFoundError(
        f"No compatible primary Phase-I state was found for seed {seed}. "
        "Run the matching primary notebook first; this extension will not "
        f"silently refit it.{detail}"
    )


def mcfg_pca_available(state: Dict[str, Any], k: int) -> bool:
    return k in state.get("pca_models", {})


def load_existing_seed_vae_only(seed: int, cfg):
    """Load a metadata-compatible VAE checkpoint; never train a replacement."""
    build_seed_partition(seed, cfg)
    expected = split_metadata(seed, cfg)
    filename = f"seed{seed}_{SPLIT_HASH}_vae_e{cfg.vae_epochs}_d{cfg.vae_latent}.pt"
    candidates = [
        Path(cfg.out_root) / "vae_models" / filename,
        *[Path(root) / "vae_models" / filename for root in cfg.legacy_out_roots],
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("metadata") == expected:
            model = ConvVAE(cfg.vae_latent)
            model.load_state_dict(payload["state_dict"])
            print(f"[seed {seed}] loaded existing VAE {path}")
            return model.to(DEVICE).eval()
    raise FileNotFoundError(
        f"No compatible pre-trained VAE exists for seed {seed}. Expected "
        f"{filename} in the primary benchmark output roots. VAE retraining is "
        "disabled in this extension by design."
    )


def _align_to_phase1(local_components: np.ndarray,
                     phase1_components: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes alignment of local axes to frozen Phase-I axes."""
    cross = np.asarray(local_components) @ np.asarray(phase1_components).T
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    rotation = right_t.T @ left.T
    return rotation @ np.asarray(local_components)


def _fit_local_pca(raw_window: np.ndarray, k: int,
                   phase1_components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact compact PCA for one FIFO window, used by audits and fallbacks."""
    raw = np.asarray(raw_window, dtype=np.float64)
    mean = raw.mean(axis=0)
    centered = raw - mean
    _, _, right_t = np.linalg.svd(centered, full_matrices=False)
    components = right_t[:k]
    return mean, _align_to_phase1(components, phase1_components)


class RollingGramPCA:
    """Exact stride-one MWPCA using an up/down-dated observation Gram matrix.

    With H=50 and p as large as 1,024, updating the H-by-H raw Gram matrix is
    considerably cheaper than forming the p-by-p covariance matrix.  Removing
    its first row/column and adding the incoming observation is algebraically
    equivalent to deleting/adding observations in the covariance estimate.
    The centered eigensystem is solved after every slide, so update_every=1 is
    a scientific update cadence rather than merely a scoring cadence.
    """

    def __init__(self, raw_window: np.ndarray, k: int,
                 phase1_components: np.ndarray):
        self.buffer = np.asarray(raw_window, dtype=np.float64).copy()
        self.k = int(k)
        self.phase1_components = np.asarray(
            phase1_components, dtype=np.float64
        )
        self.raw_gram = self.buffer @ self.buffer.T

    def slide(self, incoming: np.ndarray) -> None:
        x = np.asarray(incoming, dtype=np.float64)
        remaining = self.buffer[1:]
        updated = np.empty_like(self.raw_gram)
        updated[:-1, :-1] = self.raw_gram[1:, 1:]
        cross = remaining @ x
        updated[:-1, -1] = cross
        updated[-1, :-1] = cross
        updated[-1, -1] = float(x @ x)
        self.buffer[:-1] = remaining
        self.buffer[-1] = x
        self.raw_gram = updated

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.buffer.mean(axis=0)
        row_mean = self.raw_gram.mean(axis=1)
        centered_gram = (
            self.raw_gram - row_mean[:, None] - row_mean[None, :]
            + float(self.raw_gram.mean())
        )
        centered_gram = 0.5 * (centered_gram + centered_gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(centered_gram)
        ids = np.argsort(eigenvalues)[::-1][:self.k]
        eigenvalues = np.clip(eigenvalues[ids], 1e-12, None)
        left = eigenvectors[:, ids]
        centered = self.buffer - mean
        components = (left.T @ centered) / np.sqrt(eigenvalues)[:, None]
        # Re-orthogonalize against accumulated roundoff before alignment.
        q, _ = np.linalg.qr(components.T)
        components = q[:, :self.k].T
        return mean, _align_to_phase1(
            components, self.phase1_components
        )


def _subspace_distance(current: np.ndarray, baseline: np.ndarray) -> float:
    overlap = float(np.square(np.asarray(current) @ np.asarray(baseline).T).sum())
    k = int(current.shape[0])
    return float(np.sqrt(max(0.0, 2.0 * k - 2.0 * overlap)))


def _regularized_covariance(values: np.ndarray, ridge: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cov, _ = ledoit_wolf(values, assume_centered=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    scale = max(float(np.trace(cov)) / cov.shape[0], 1e-12)
    return 0.5 * (cov + cov.T) + np.eye(cov.shape[0]) * ridge * scale


def _prepare_reference_scores(scores: np.ndarray,
                              mcfg: MovingWindowPCAConfig) -> Dict[str, Any]:
    ref = np.asarray(scores, dtype=np.float64)
    mu = ref.mean(axis=0)
    cov = _regularized_covariance(ref, mcfg.covariance_ridge)
    cov_inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Phase-I score covariance is not positive definite")
    chol = np.linalg.cholesky(cov)
    mmd_ref = ref[:min(len(ref), mcfg.mmd_reference_n)].copy()
    squared = np.maximum(
        np.square(mmd_ref).sum(axis=1)[:, None]
        + np.square(mmd_ref).sum(axis=1)[None, :]
        - 2.0 * mmd_ref @ mmd_ref.T,
        0.0,
    )
    upper = squared[np.triu_indices_from(squared, k=1)]
    gamma = 1.0 / max(float(np.median(upper)), 1e-12)
    kernel = np.exp(-gamma * squared)
    r = len(mmd_ref)
    term_rr = float((kernel.sum() - r) / (r * (r - 1)))
    return {
        "scores": ref, "mu": mu, "cov": cov, "cov_inv": cov_inv,
        "logdet": float(logdet), "chol": chol,
        "mmd_ref": mmd_ref, "mmd_gamma": gamma, "mmd_term_rr": term_rr,
    }


def _mmd_unbiased(window: np.ndarray, reference: Dict[str, Any]) -> float:
    x = np.asarray(window, dtype=np.float64)
    r = reference["mmd_ref"]
    gamma = float(reference["mmd_gamma"])
    xx = np.maximum(
        np.square(x).sum(axis=1)[:, None] + np.square(x).sum(axis=1)[None, :]
        - 2.0 * x @ x.T, 0.0,
    )
    xr = np.maximum(
        np.square(x).sum(axis=1)[:, None] + np.square(r).sum(axis=1)[None, :]
        - 2.0 * x @ r.T, 0.0,
    )
    m = len(x)
    term_xx = float((np.exp(-gamma * xx).sum() - m) / (m * (m - 1)))
    return float(reference["mmd_term_rr"] + term_xx - 2.0 * np.exp(-gamma * xr).mean())


def _window_scores(raw_window: np.ndarray, phase1: Dict[str, Any],
                   mcfg: MovingWindowPCAConfig,
                   local_components: Optional[np.ndarray] = None
                   ) -> Tuple[Dict[str, float], np.ndarray]:
    initial_mean = phase1["initial_mean"]
    initial_components = phase1["initial_components"]
    if local_components is None:
        _, local_components = _fit_local_pca(
            raw_window, mcfg.pca_dim, initial_components
        )
    # Fixed centering preserves mean drift. Procrustes alignment makes the
    # moving coordinates commensurable with the frozen Phase-I score axes.
    current = (np.asarray(raw_window, dtype=np.float64) - initial_mean) @ local_components.T
    reference = phase1["prepared_reference"]
    mean = current.mean(axis=0)
    cov = _regularized_covariance(current, mcfg.covariance_ridge)
    cov_inv = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise np.linalg.LinAlgError("Moving-window score covariance is not positive definite")
    difference = mean - reference["mu"]
    k = mcfg.pca_dim
    t2 = float(mcfg.window * difference @ reference["cov_inv"] @ difference)
    sym_kl = 0.25 * (
        np.trace(reference["cov_inv"] @ cov)
        + np.trace(cov_inv @ reference["cov"])
        + difference @ (reference["cov_inv"] + cov_inv) @ difference
        - 2.0 * k
    )
    average_cov = 0.5 * (reference["cov"] + cov)
    avg_inv = np.linalg.inv(average_cov)
    avg_sign, avg_logdet = np.linalg.slogdet(average_cov)
    if avg_sign <= 0:
        raise np.linalg.LinAlgError("Average covariance is not positive definite")
    log_coefficient = (
        0.25 * reference["logdet"] + 0.25 * logdet
        - 0.5 * avg_logdet - 0.125 * difference @ avg_inv @ difference
    )
    hellinger = float(np.sqrt(np.clip(1.0 - np.exp(log_coefficient), 0.0, 1.0)))
    innovation = np.sqrt(mcfg.window) * np.linalg.solve(
        reference["chol"], difference
    )
    scores = {
        "Subspace": _subspace_distance(local_components, initial_components),
        "T2": t2,
        "Gaussian_KL": float(max(sym_kl, 0.0)),
        "Gaussian_Hellinger": hellinger,
        "MMD": _mmd_unbiased(current, reference),
    }
    return scores, innovation


def score_mwpca_sequence(raw_sequence: np.ndarray, phase1: Dict[str, Any],
                         mcfg: MovingWindowPCAConfig,
                         stop_thresholds: Optional[Dict[str, float]] = None,
                         progress_label: str = "") -> Dict[str, np.ndarray]:
    """Score a raw sequence; optionally stop after every detector has alarmed."""
    raw = np.asarray(raw_sequence, dtype=np.float64)
    methods = [method for method in mcfg.detector_methods if method != "MEWMA"]
    traces: Dict[str, List[Any]] = {method: [] for method in methods}
    innovations: List[np.ndarray] = []
    mewma_values: List[float] = []
    z = np.zeros(mcfg.pca_dim, dtype=np.float64)
    alarm_indices: Dict[str, Optional[int]] = {
        method: None for method in mcfg.detector_methods
    }
    if len(raw) < mcfg.window:
        raise ValueError("Sequence is shorter than the moving PCA window")
    rolling = RollingGramPCA(
        raw[:mcfg.window], mcfg.pca_dim, phase1["initial_components"]
    )
    for score_index, end in enumerate(range(mcfg.window, len(raw) + 1)):
        if score_index > 0:
            rolling.slide(raw[end - 1])
        _, local_components = rolling.fit()
        scores, innovation = _window_scores(
            rolling.buffer, phase1, mcfg,
            local_components=local_components,
        )
        for method in methods:
            traces[method].append(scores[method])
        innovations.append(innovation)
        z = mcfg.ewma_lambda * innovation + (1.0 - mcfg.ewma_lambda) * z
        mewma = float(np.square(z).sum() * (2.0 - mcfg.ewma_lambda) / mcfg.ewma_lambda)
        mewma_values.append(mewma)
        if stop_thresholds is not None:
            for method in mcfg.detector_methods:
                if alarm_indices[method] is not None:
                    continue
                value = mewma if method == "MEWMA" else scores[method]
                if value > stop_thresholds[method]:
                    alarm_indices[method] = score_index
            if all(value is not None for value in alarm_indices.values()):
                break
        if (
            mcfg.progress_every_windows > 0
            and (score_index + 1) % mcfg.progress_every_windows == 0
            and progress_label
        ):
            print(f"{progress_label}: {score_index + 1} moving windows scored")
    result = {key: np.asarray(value, dtype=np.float64) for key, value in traces.items()}
    result["MEWMA"] = np.asarray(mewma_values, dtype=np.float64)
    result["innovations"] = np.asarray(innovations, dtype=np.float64)
    result["alarm_indices"] = alarm_indices
    return result


def _feature_bank(ids: np.ndarray, seed: int, purpose: int, cfg,
                  extractor, vae, mcfg: MovingWindowPCAConfig) -> Dict[str, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int64)
    images = augment_test_batch(
        IMAGES[ids], np.arange(len(ids)),
        cfg.stream_master_seed + seed * 1000 + purpose,
    )
    features = extractor.extract(images)
    features["vae_latent"] = vae_latents(vae, images)
    return {
        block: np.asarray(features[block], dtype=np.float64)
        for block in mcfg.feature_blocks
    }


def build_mwpca_phase1(seed: int, cfg, mcfg: MovingWindowPCAConfig,
                       extractor) -> Dict[str, Any]:
    build_seed_partition(seed, cfg)
    root = Path(mcfg.out_root) / "phase1" / f"seed{seed}_{SPLIT_HASH}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "mwpca_phase1.pkl"
    metadata = _metadata(seed, cfg, mcfg)
    if path.exists() and not mcfg.force_recompute:
        with path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("metadata") == metadata:
            return cached

    primary = load_primary_state_only(seed, cfg)
    vae = load_existing_seed_vae_only(seed, cfg)
    ref_ids = np.asarray(primary["ref_ids"], dtype=np.int64)[:mcfg.reference_n]
    reference_features = _feature_bank(ref_ids, seed, 91, cfg, extractor, vae, mcfg)
    blocks: Dict[str, Any] = {}
    for block in mcfg.feature_blocks:
        pca = primary["pca_models"][mcfg.pca_dim][block]
        initial_mean = np.asarray(pca.mean_, dtype=np.float64)
        initial_components = np.asarray(pca.components_, dtype=np.float64)
        reference_scores = (
            np.asarray(reference_features[block], dtype=np.float64) - initial_mean
        ) @ initial_components.T
        blocks[block] = {
            "initial_mean": initial_mean,
            "initial_components": initial_components,
            "reference_scores": reference_scores,
            "prepared_reference": _prepare_reference_scores(reference_scores, mcfg),
        }
    payload = {
        "metadata": metadata,
        "fit_ids": np.asarray(primary["fit_ids"], dtype=np.int64),
        "reference_ids": ref_ids,
        "blocks": blocks,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def _score_bank_path(root: Path, block: str, purpose: str) -> Path:
    return root / f"{purpose}_{block}_score_bank.npz"


def _load_or_score_ic_bank(seed: int, cfg, mcfg: MovingWindowPCAConfig,
                           extractor, phase1: Dict[str, Any], purpose: str,
                           ids: np.ndarray, purpose_code: int) -> Dict[str, Dict[str, np.ndarray]]:
    root = Path(mcfg.out_root) / "phase1" / f"seed{seed}_{SPLIT_HASH}"
    selected = np.asarray(ids, dtype=np.int64)[:mcfg.score_bank_observations]
    missing = [
        block for block in mcfg.feature_blocks
        if mcfg.force_recompute
        or not _score_bank_path(root, block, purpose).exists()
    ]
    banks = None
    if missing:
        vae = load_existing_seed_vae_only(seed, cfg)
        banks = _feature_bank(selected, seed, purpose_code, cfg, extractor, vae, mcfg)
    output: Dict[str, Dict[str, np.ndarray]] = {}
    for block in mcfg.feature_blocks:
        path = _score_bank_path(root, block, purpose)
        if path.exists() and not mcfg.force_recompute:
            loaded = np.load(path)
            output[block] = {key: loaded[key] for key in loaded.files}
            continue
        trace = score_mwpca_sequence(
            banks[block], phase1["blocks"][block], mcfg,
            progress_label=f"[seed {seed}] {purpose} {block}",
        )
        serializable = {
            key: value for key, value in trace.items()
            if key != "alarm_indices"
        }
        np.savez_compressed(path, **serializable)
        output[block] = serializable
    return output


def _calibration_plan(n_scores: int, episodes: int, cfg,
                      mcfg: MovingWindowPCAConfig, seed: int, purpose: int):
    return moving_block_plan(
        n_scores, episodes, cfg.arl0_horizon_steps,
        mcfg.bootstrap_block_steps,
        np.random.default_rng(np.random.SeedSequence([
            cfg.stream_master_seed, seed, purpose
        ])),
    )


def _records_for_method(bank: Dict[str, np.ndarray], method: str, plan,
                        cfg, mcfg: MovingWindowPCAConfig):
    if method == "MEWMA":
        return _ewma_records(
            bank["innovations"], mcfg.ewma_lambda, plan,
            cfg.arl0_horizon_steps, mcfg.bootstrap_block_steps,
        )
    return _records_from_values(
        bank[method], "high", plan, cfg.arl0_horizon_steps,
        mcfg.bootstrap_block_steps,
    )


def calibrate_mwpca_seed(seed: int, cfg, mcfg: MovingWindowPCAConfig,
                         extractor, phase1: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(mcfg.out_root) / "phase1" / f"seed{seed}_{SPLIT_HASH}"
    path = root / "mwpca_thresholds.pkl"
    metadata = {
        **_metadata(seed, cfg, mcfg),
        "calibration_episodes": int(mcfg.calibration_episodes),
    }
    if path.exists() and not mcfg.force_recompute:
        with path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("metadata") == metadata:
            return cached
    validation_ids = np.random.default_rng(np.random.SeedSequence([
        cfg.stream_master_seed, seed, 92
    ])).permutation(VALIDATION_IC_IDS)
    banks = _load_or_score_ic_bank(
        seed, cfg, mcfg, extractor, phase1, "validation",
        validation_ids, 92,
    )
    thresholds: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for block in mcfg.feature_blocks:
        n_scores = len(banks[block]["Subspace"])
        plan = _calibration_plan(
            n_scores, mcfg.calibration_episodes, cfg, mcfg, seed, 93
        )
        for method in mcfg.detector_methods:
            records = _records_for_method(banks[block], method, plan, cfg, mcfg)
            thresholds[(block, method)] = calibrate_records(records, "high", cfg)
    payload = {"metadata": metadata, "thresholds": thresholds}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    rows = [{
        "dataset": cfg.dataset_name, "seed": seed, "split_hash": SPLIT_HASH,
        "block": block, "method": method, **values,
    } for (block, method), values in thresholds.items()]
    pd.DataFrame(rows).to_csv(root / "mwpca_thresholds.csv", index=False)
    return payload


def verify_mwpca_heldout(seed: int, cfg, mcfg: MovingWindowPCAConfig,
                         extractor, phase1: Dict[str, Any],
                         calibration: Dict[str, Any]) -> pd.DataFrame:
    root = Path(mcfg.out_root) / "heldout" / f"seed{seed}_{SPLIT_HASH}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "heldout_arl0_summary.csv"
    if path.exists() and not mcfg.force_recompute:
        cached = pd.read_csv(path)
        if len(cached) == len(mcfg.feature_blocks) * len(mcfg.detector_methods):
            return cached
    ids = np.random.default_rng(np.random.SeedSequence([
        cfg.stream_master_seed, seed, 94
    ])).permutation(TEST_IC_IDS)
    # Held-out score banks live beside Phase-I caches because they are reusable;
    # their identities remain TEST_IC only and never enter calibration.
    banks = _load_or_score_ic_bank(
        seed, cfg, mcfg, extractor, phase1, "heldout", ids, 94
    )
    rows = []
    for block in mcfg.feature_blocks:
        plan = _calibration_plan(
            len(banks[block]["Subspace"]), mcfg.heldout_episodes,
            cfg, mcfg, seed, 95,
        )
        for method in mcfg.detector_methods:
            records = _records_for_method(banks[block], method, plan, cfg, mcfg)
            threshold = calibration["thresholds"][(block, method)]["threshold"]
            lengths, observed = _run_lengths(
                records, threshold, cfg.arl0_horizon_steps
            )
            observations = proper_steps_to_observations(lengths, cfg)
            rows.append({
                "dataset": cfg.dataset_name, "seed": seed,
                "split_hash": SPLIT_HASH, "block": block, "method": method,
                "episodes": len(lengths), "events": int(observed.sum()),
                "censored": int((~observed).sum()),
                "censor_rate": float((~observed).mean()),
                "arl0_observations": float(observations.mean()),
                "arl0_score_steps": float(lengths.mean()),
                "target_arl0_observations": cfg.target_arl0_observations,
                "in_10pct_band": bool(
                    cfg.target_arl0_observations * (1 - cfg.arl0_tol)
                    <= observations.mean()
                    <= cfg.target_arl0_observations * (1 + cfg.arl0_tol)
                ),
                "estimate_type": (
                    "empirical_mean" if observed.all() else "restricted_mean"
                ),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _episode_blocks(images: np.ndarray, vae,
                    mcfg: MovingWindowPCAConfig) -> Dict[str, np.ndarray]:
    features = EXTRACTOR.extract(images)
    features["vae_latent"] = vae_latents(vae, images)
    return {block: np.asarray(features[block], dtype=np.float64)
            for block in mcfg.feature_blocks}


def evaluate_mwpca_seed(seed: int, cfg, mcfg: MovingWindowPCAConfig,
                        extractor, phase1: Dict[str, Any],
                        calibration: Dict[str, Any]) -> pd.DataFrame:
    build_seed_partition(seed, cfg)
    vae = load_existing_seed_vae_only(seed, cfg)
    ic_orders, ooc_orders = master_episode_manifest(seed, cfg)
    root = Path(mcfg.out_root) / "drift" / f"seed{seed}_{SPLIT_HASH}"
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for pattern in cfg.patterns:
        for mechanism in cfg.mechanisms:
            for severity in cfg.severity_levels:
                path = root / f"{pattern}_{mechanism}_s{severity:.2f}.csv"
                expected = cfg.mc_arl1_reps * len(mcfg.feature_blocks) * len(mcfg.detector_methods)
                if path.exists() and not mcfg.force_recompute:
                    cached = pd.read_csv(path)
                    if len(cached) == expected:
                        frames.append(cached)
                        continue
                rows = []
                for replication in range(cfg.mc_arl1_reps):
                    images, observed_ids = build_episode(
                        seed, replication, pattern, mechanism, severity,
                        ic_orders[replication], ooc_orders[replication], cfg,
                    )
                    blocks = _episode_blocks(images, vae, mcfg)
                    for block in mcfg.feature_blocks:
                        thresholds = {
                            method: calibration["thresholds"][(block, method)]["threshold"]
                            for method in mcfg.detector_methods
                        }
                        trace = score_mwpca_sequence(
                            blocks[block], phase1["blocks"][block], mcfg,
                            stop_thresholds=thresholds,
                        )
                        for method in mcfg.detector_methods:
                            hit = trace["alarm_indices"][method]
                            detected = hit is not None
                            delay = float(hit + 1) if detected else np.nan
                            rows.append({
                                "dataset": cfg.dataset_name, "seed": seed,
                                "split_hash": SPLIT_HASH,
                                "replication": replication,
                                "pattern": pattern, "mechanism": mechanism,
                                "severity": float(severity), "block": block,
                                "pca_dim": mcfg.pca_dim,
                                "pca_window": mcfg.window,
                                "update_every": mcfg.update_every,
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
                                "cal_arl0_observations": calibration[
                                    "thresholds"
                                ][(block, method)]["cal_arl0_observations"],
                                "unique_identities": len(np.unique(observed_ids)),
                            })
                frame = pd.DataFrame(rows)
                frame.to_csv(path, index=False)
                frames.append(frame)
                print(
                    f"[seed {seed}] MWPCA {pattern}/{mechanism}/"
                    f"severity={severity:g} complete"
                )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(root / "mwpca_episode_results.csv", index=False)
    return result


def aggregate_mwpca_results(results: pd.DataFrame, heldout: pd.DataFrame,
                            cfg, mcfg: MovingWindowPCAConfig) -> Dict[str, pd.DataFrame]:
    root = Path(mcfg.out_root) / "aggregate"
    root.mkdir(parents=True, exist_ok=True)
    results.to_csv(root / "mwpca_episode_results_all_seeds.csv", index=False)
    heldout.to_csv(root / "mwpca_heldout_arl0_all_seeds.csv", index=False)
    condition = [
        "dataset", "seed", "pattern", "mechanism", "severity", "block", "method"
    ]
    seed_level = results.groupby(condition, dropna=False).agg(
        restricted_delay=("restricted_delay_observations", "mean"),
        detection_rate=("detected", "mean"),
        episodes=("replication", "size"),
    ).reset_index()
    seed_level.to_csv(root / "mwpca_seed_level.csv", index=False)
    group_columns = [column for column in condition if column != "seed"]
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
            "seeds": int(group["seed"].nunique()),
            "episodes_per_seed": int(group["episodes"].min()),
        })
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "mwpca_condition_summary.csv", index=False)
    heldout_summary = heldout.groupby(
        ["dataset", "block", "method"], dropna=False
    ).agg(
        mean_heldout_arl0=("arl0_observations", "mean"),
        mean_censor_rate=("censor_rate", "mean"),
        seeds=("seed", "nunique"),
        seeds_in_10pct_band=("in_10pct_band", "sum"),
    ).reset_index()
    heldout_summary.to_csv(root / "mwpca_heldout_arl0_summary.csv", index=False)
    return {
        "episode_results": results,
        "heldout": heldout,
        "seed_level": seed_level,
        "summary": summary,
        "heldout_summary": heldout_summary,
    }


def run_mwpca_extension(cfg, mcfg: MovingWindowPCAConfig, extractor):
    if cfg.window != mcfg.window or cfg.stride != 1:
        raise ValueError("This extension requires the primary 50/1 window and stride")
    if cfg.changepoint != cfg.window:
        raise ValueError("The first drifted observation must complete the first score window")
    Path(mcfg.out_root).mkdir(parents=True, exist_ok=True)
    all_results, all_heldout = [], []
    for seed in cfg.seeds:
        print(f"\n=== fixed-reference MWPCA seed {seed} ({cfg.dataset_name}) ===")
        phase1 = build_mwpca_phase1(seed, cfg, mcfg, extractor)
        calibration = calibrate_mwpca_seed(seed, cfg, mcfg, extractor, phase1)
        all_heldout.append(verify_mwpca_heldout(
            seed, cfg, mcfg, extractor, phase1, calibration
        ))
        all_results.append(evaluate_mwpca_seed(
            seed, cfg, mcfg, extractor, phase1, calibration
        ))
    return aggregate_mwpca_results(
        pd.concat(all_results, ignore_index=True),
        pd.concat(all_heldout, ignore_index=True), cfg, mcfg,
    )


def mwpca_design_audit(cfg, mcfg: MovingWindowPCAConfig) -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": cfg.dataset_name,
        "seeds": len(cfg.seeds),
        "episodes_per_exact_drift_condition_per_seed": cfg.mc_arl1_reps,
        "feature_blocks": ", ".join(mcfg.feature_blocks),
        "pca_dim": mcfg.pca_dim,
        "moving_window": mcfg.window,
        "stride": cfg.stride,
        "pca_updates_every_observations": mcfg.update_every,
        "update_gate": "none",
        "current_observation_in_local_fit": True,
        "fixed_phase1_comparator": True,
        "detectors": ", ".join(mcfg.detector_methods),
        "target_arl0_observations": cfg.target_arl0_observations,
        "calibration_episodes": mcfg.calibration_episodes,
        "heldout_episodes": mcfg.heldout_episodes,
        "vae_policy": "load compatible primary checkpoint; never retrain",
    }])


# Lightweight mathematical checks; these do not touch either image dataset.
_rng = np.random.default_rng(811)
_raw = _rng.normal(size=(50, 32))
_phase = np.linalg.svd(_rng.normal(size=(20, 32)), full_matrices=False)[2][:20]
_mean, _components = _fit_local_pca(_raw, 20, _phase)
_rolling = RollingGramPCA(_raw, 20, _phase)
_rolling_mean, _rolling_components = _rolling.fit()
assert _components.shape == (20, 32)
assert np.allclose(_components @ _components.T, np.eye(20), atol=1e-8)
assert np.allclose(_rolling_mean, _mean, atol=1e-10)
assert np.allclose(
    _rolling_components.T @ _rolling_components,
    _components.T @ _components, atol=1e-7,
)
_incoming = _rng.normal(size=32)
_rolling.slide(_incoming)
_shifted = np.vstack([_raw[1:], _incoming])
_, _shifted_components = _fit_local_pca(_shifted, 20, _phase)
_, _rolling_shifted = _rolling.fit()
assert np.allclose(
    _rolling_shifted.T @ _rolling_shifted,
    _shifted_components.T @ _shifted_components, atol=1e-7,
)
assert abs(_subspace_distance(_phase, _phase)) < 1e-6

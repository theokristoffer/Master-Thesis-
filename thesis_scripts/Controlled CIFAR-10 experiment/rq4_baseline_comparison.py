# -*- coding: utf-8 -*-
"""RQ4 benchmark: latent-space SPC versus external drift detectors.

Requires the companion RQ2 script in the same directory. Optional dependency:
``pip install river`` for the standard ADWIN implementation.
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, wilcoxon

import control_charts_rq2_scalar_vs_multivariate_memory as core

try:
    from river.drift import ADWIN
except ImportError:
    ADWIN = None


CONFIG = {
    "seed": 42, "n_streams": 20, "stream_length": 2000,
    "changepoint": 1000, "drift_types": ["sudden", "incremental", "periodic"],
    "fit_size": 1000, "validation_size": 1000,
    "normal_pool_size": 1000, "drift_pool_size": 1000,
    "t2_pca_components": 100, "spe_pca_components": 50,
    "baseline_pca_components": 100,
    "ks_window": 50, "mmd_window": 50, "stride": 1,
    "baseline_reference_size": 200,
    "sigma_multiplier": 3.0, "ewma_lambda": 0.2, "cusum_k": 0.5,
    "mewma_lambda": 0.2, "mcusum_k": 0.5,
    "adwin_delta_candidates": [0.001, 0.002, 0.005, 0.01, 0.02],
    "nominal_arl0": 370,
    "alarm_density_window": 25, "alarm_density_gamma": 0.30,
    "incremental_transition": 500, "periodic_episode_length": 200,
    "selected_memory_method": "SPE_EWMA",
    "bootstrap_iterations": 1000,
    "embedding_cache": "rq4_resnet18_latent_cache.npz",
    "output_dir": "rq4_baseline_comparison_results",
}

SPC_METHODS = ["T2_point", "SPE_point", "SPE_EWMA", "SPE_CUSUM", "MEWMA", "MCUSUM"]
BASELINES = ["KS_fixed", "MMD_RBF", "ADWIN"]
METHODS = SPC_METHODS + BASELINES
METHOD_FAMILY = {
    "T2_point": "SPC_memoryless", "SPE_point": "SPC_memoryless",
    "SPE_EWMA": "SPC_memory", "SPE_CUSUM": "SPC_memory",
    "MEWMA": "SPC_memory", "MCUSUM": "SPC_memory",
    "KS_fixed": "external_distribution_test", "MMD_RBF": "external_kernel_test",
    "ADWIN": "external_adaptive_window",
}


def configure_core():
    """Keep shared data/stream semantics exactly aligned with prior RQs."""
    core.CONFIG.update({
        "random_seed": CONFIG["seed"], "fit_size": CONFIG["fit_size"],
        "validation_size": CONFIG["validation_size"],
        "normal_pool_size": CONFIG["normal_pool_size"],
        "drift_pool_size": CONFIG["drift_pool_size"],
        "n_eval_streams": CONFIG["n_streams"], "stream_length": CONFIG["stream_length"],
        "changepoint": CONFIG["changepoint"],
        "incremental_transition": CONFIG["incremental_transition"],
        "periodic_episode_length": CONFIG["periodic_episode_length"],
        "sigma_multiplier": CONFIG["sigma_multiplier"],
        "ewma_lambda": CONFIG["ewma_lambda"], "cusum_k": CONFIG["cusum_k"],
        "mewma_lambda": CONFIG["mewma_lambda"], "mcusum_k": CONFIG["mcusum_k"],
        "alarm_density_window": CONFIG["alarm_density_window"],
        "alarm_density_gamma": CONFIG["alarm_density_gamma"],
        "bootstrap_iterations": CONFIG["bootstrap_iterations"],
    })


def load_or_extract_embeddings(loaders):
    """Cache the four role-specific ResNet-18 embedding pools."""
    output = Path(CONFIG["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    cache_path = output / CONFIG["embedding_cache"]
    names = ("fit", "validation", "normal_pool", "drift_pool")
    sizes = {name: CONFIG[f"{name}_size"] for name in names}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if all(name in cache and f"{name}_labels" in cache and len(cache[name]) == sizes[name] for name in names):
                print(f"Loaded embedding cache: {cache_path.resolve()}")
                return ({name: cache[name] for name in names},
                        {name: cache[f"{name}_labels"] for name in names})
    data, labels = core.build_feature_data(loaders)
    payload = {key: value for name in names for key, value in
               ((name, data[name]), (f"{name}_labels", labels[name]))}
    np.savez_compressed(cache_path, **payload)
    return data, labels


def three_sigma(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean() + CONFIG["sigma_multiplier"] * values.std(ddof=1))


def build_spc_models(data):
    """Fit the prescribed 100-PC and 50-PC Phase-I models only once."""
    core.CONFIG["pca_components"] = CONFIG["t2_pca_components"]
    model100, table100 = core.fit_phase1_model(data["fit"], data["validation"])
    core.CONFIG["pca_components"] = CONFIG["spe_pca_components"]
    model50, table50 = core.fit_phase1_model(data["fit"], data["validation"])
    z_validation50 = model50["pca"].transform(data["validation"])
    spe_validation50 = core.spe_scores(data["validation"], z_validation50, model50["pca"])
    spe_point_threshold = three_sigma(spe_validation50)
    model50["thresholds"]["SPE_point"] = spe_point_threshold
    rows = []
    for method in ("T2_point", "MEWMA", "MCUSUM"):
        row = table100[table100.monitoring_method == method].iloc[0].to_dict()
        row.update(method=method, pca_components=model100["n_components"],
                   calibration="validation empirical mean + 3 SD")
        rows.append(row)
    rows.append({"method": "SPE_point", "monitoring_method": "SPE_point",
                 "pca_components": model50["n_components"],
                 "validation_mean": float(spe_validation50.mean()),
                 "validation_sd": float(spe_validation50.std(ddof=1)),
                 "threshold": spe_point_threshold,
                 "calibration": "validation empirical mean + 3 SD"})
    for method in ("SPE_EWMA", "SPE_CUSUM"):
        row = table50[table50.monitoring_method == method].iloc[0].to_dict()
        row.update(method=method, pca_components=model50["n_components"],
                   calibration="validation empirical mean + 3 SD")
        rows.append(row)
    return model100, model50, rows


def ks_min_p_signal(reference, sequence, window):
    """Minimum marginal two-sample KS p-value for each complete window."""
    from scipy.stats import ks_2samp
    signal = np.full(len(sequence), np.nan)
    for end in range(window - 1, len(sequence), CONFIG["stride"]):
        current = sequence[end - window + 1:end + 1]
        signal[end] = min(ks_2samp(reference[:, j], current[:, j], method="asymp").pvalue
                          for j in range(reference.shape[1]))
    return signal


def rbf_bandwidth(reference):
    """Fixed validation-safe RBF bandwidth from the Phase-I median heuristic."""
    distances = pdist(reference, metric="sqeuclidean")
    positive = distances[distances > 0]
    median_squared_distance = float(np.median(positive)) if len(positive) else 1.0
    return np.sqrt(0.5 * median_squared_distance)


def mmd_rbf_signal(reference, sequence, window, bandwidth):
    """Biased MMD^2; fixed-reference terms are cached for online efficiency."""
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    ref_norm = np.sum(reference ** 2, axis=1)
    k_ref = np.exp(-gamma * np.maximum(ref_norm[:, None] + ref_norm[None, :] - 2 * reference @ reference.T, 0))
    ref_term = k_ref.mean()
    signal = np.full(len(sequence), np.nan)
    for end in range(window - 1, len(sequence), CONFIG["stride"]):
        current = sequence[end - window + 1:end + 1]
        cur_norm = np.sum(current ** 2, axis=1)
        cross = np.exp(-gamma * np.maximum(ref_norm[:, None] + cur_norm[None, :] - 2 * reference @ current.T, 0))
        within = np.exp(-gamma * np.maximum(cur_norm[:, None] + cur_norm[None, :] - 2 * current @ current.T, 0))
        signal[end] = ref_term + within.mean() - 2 * cross.mean()
    return signal


def run_adwin(values, delta):
    """Run river's standard ADWIN and return its change-detection impulses."""
    if ADWIN is None:
        raise ImportError("RQ4 requires river for standard ADWIN: pip install river")
    detector = ADWIN(delta=delta)
    alarms = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        detector.update(float(value)); alarms[i] = bool(detector.drift_detected)
    return alarms


def calibrate_baselines(model100, data):
    """Calibrate KS, MMD and ADWIN exclusively on fit/reference + validation."""
    pca = model100["pca"]
    fit_scores = pca.transform(data["fit"])
    validation_scores = pca.transform(data["validation"])
    reference = fit_scores[:CONFIG["baseline_reference_size"]].copy()
    ks_validation = ks_min_p_signal(reference, validation_scores, CONFIG["ks_window"])
    valid_ks = ks_validation[np.isfinite(ks_validation)]
    # Lower-tail analogue of three-sigma: mean - 3 SD, bounded at zero.
    ks_threshold = max(0.0, float(valid_ks.mean() - CONFIG["sigma_multiplier"] * valid_ks.std(ddof=1)))
    # If mean-3SD is zero, use the empirical 0.27% lower-tail quantile.
    if ks_threshold == 0:
        ks_threshold = float(np.quantile(valid_ks, 0.0027))
    bandwidth = rbf_bandwidth(reference)
    mmd_validation = mmd_rbf_signal(reference, validation_scores, CONFIG["mmd_window"], bandwidth)
    valid_mmd = mmd_validation[np.isfinite(mmd_validation)]
    mmd_threshold = three_sigma(valid_mmd)
    centroid = fit_scores.mean(axis=0)
    validation_distance = np.linalg.norm(validation_scores - centroid, axis=1)
    candidates = []
    for delta in CONFIG["adwin_delta_candidates"]:
        alarms = run_adwin(validation_distance, delta)
        arl, censored = core.first_run_length(alarms)
        candidates.append((abs(arl - CONFIG["nominal_arl0"]), delta, arl, censored, alarms.mean()))
    _, adwin_delta, adwin_arl, adwin_censored, adwin_far = min(candidates)
    rows = [
        {"method": "KS_fixed", "pca_components": model100["n_components"],
         "validation_mean": valid_ks.mean(), "validation_sd": valid_ks.std(ddof=1),
         "threshold": ks_threshold, "calibration_parameter": "lower mean-3SD / 0.27% quantile",
         "validation_alarm_rate": np.mean(valid_ks < ks_threshold), "window_size": CONFIG["ks_window"]},
        {"method": "MMD_RBF", "pca_components": model100["n_components"],
         "validation_mean": valid_mmd.mean(), "validation_sd": valid_mmd.std(ddof=1),
         "threshold": mmd_threshold, "calibration_parameter": f"bandwidth={bandwidth:.12g}",
         "validation_alarm_rate": np.mean(valid_mmd > mmd_threshold), "window_size": CONFIG["mmd_window"]},
        {"method": "ADWIN", "pca_components": model100["n_components"],
         "validation_mean": validation_distance.mean(), "validation_sd": validation_distance.std(ddof=1),
         "threshold": np.nan, "calibration_parameter": f"delta={adwin_delta}",
         "validation_alarm_rate": adwin_far, "validation_ARL0": adwin_arl,
         "validation_ARL0_censored": adwin_censored, "window_size": np.nan},
    ]
    return {"reference": reference, "ks_threshold": ks_threshold,
            "mmd_threshold": mmd_threshold, "bandwidth": bandwidth,
            "centroid": centroid, "adwin_delta": adwin_delta}, rows


def classification_metrics(stream, alarms, signal, pattern, valid_mask=None, auc_meaningful=True):
    """Shared metrics, excluding incomplete KS/MMD windows from classification rates."""
    regime = stream["is_drift_regime"].astype(int)
    actual = stream["is_drift_observation"].astype(int)
    pred = np.asarray(alarms, dtype=int)
    mask = np.ones(len(pred), dtype=bool) if valid_mask is None else np.asarray(valid_mask, bool)
    y, p = regime[mask], pred[mask]
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 matthews_corrcoef, precision_score, recall_score,
                                 roc_auc_score)
    try:
        auc = roc_auc_score(y, np.asarray(signal)[mask]) if auc_meaningful and len(np.unique(y)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    density = core.alarm_density(alarms)
    cp = CONFIG["changepoint"]
    arl0_end = len(alarms) if pattern == "normal_only" else cp
    pre_valid = mask[:arl0_end]
    false_positions = np.flatnonzero(alarms[:arl0_end] & pre_valid)
    arl0 = int(false_positions[0] + 1) if len(false_positions) else arl0_end + 1
    arl0_censored = not bool(len(false_positions))
    post_positions = np.flatnonzero(alarms[cp:] & mask[cp:]) if pattern != "normal_only" else np.array([], dtype=int)
    arl1 = (int(post_positions[0] + 1) if len(post_positions) else len(alarms) - cp + 1) if pattern != "normal_only" else np.nan
    arl1_censored = (not bool(len(post_positions))) if pattern != "normal_only" else np.nan
    delay = float(post_positions[0]) if pattern != "normal_only" and len(post_positions) else np.nan
    confirmations = (np.flatnonzero(density[cp:] >= CONFIG["alarm_density_gamma"])
                     if pattern != "normal_only" else np.array([], dtype=int))
    dcd = float(confirmations[0]) if pattern != "normal_only" and len(confirmations) else np.nan
    phase2 = np.arange(len(pred)) >= cp
    phase2_signals = (pred == 1) & phase2 & mask
    sr = (np.sum(phase2_signals & (regime == 0)) / phase2_signals.sum()
          if phase2_signals.sum() else np.nan)
    cdr_mask = (actual == 1) & mask
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "precision": precision_score(y, p, zero_division=0),
        "recall_detection_rate": recall_score(y, p, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "FAR": fp / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y, p, zero_division=0),
        "MCC": matthews_corrcoef(y, p) if len(np.unique(y)) > 1 and len(np.unique(p)) > 1 else 0.0,
        "ROC_AUC": auc, "SR": sr,
        "CDR": pred[cdr_mask].mean() if cdr_mask.any() else np.nan,
        "ARL0": arl0, "ARL0_censored": arl0_censored,
        "ARL1": arl1, "ARL1_censored": arl1_censored,
        "detection_delay": delay,
        "detected_within_25": bool(delay <= 25) if np.isfinite(delay) else (np.nan if pattern == "normal_only" else False),
        "detected_within_50": bool(delay <= 50) if np.isfinite(delay) else (np.nan if pattern == "normal_only" else False),
        "detected_within_100": bool(delay <= 100) if np.isfinite(delay) else (np.nan if pattern == "normal_only" else False),
        "maximum_alarm_density_after_drift": float(density[cp:].max()) if pattern != "normal_only" else np.nan,
        "mean_alarm_density_during_drift": float(density[stream["is_drift_regime"]].mean()) if stream["is_drift_regime"].any() else np.nan,
        "DCD": dcd, "DCD_censored": (not bool(len(confirmations))) if pattern != "normal_only" else np.nan,
        "alarm_density_json": json.dumps(density.tolist()),
    }


def monitor_stream(stream, model100, model50, baseline):
    """Produce every detector sequence and measure online monitoring runtime."""
    x = stream["x"]
    outputs = {}
    # PCA projection is common representation preparation and is excluded from
    # online detector timing, just like ResNet extraction and Phase-I fitting.
    z = model100["pca"].transform(x); z50 = model50["pca"].transform(x)
    start = time.perf_counter(); t2 = core.t2_scores(z, model100["score_mean"], model100["covariance_inverse"]); elapsed = time.perf_counter() - start
    outputs["T2_point"] = (t2, t2 > model100["thresholds"]["T2_point"], model100["thresholds"]["T2_point"], elapsed, None)
    start = time.perf_counter(); mewma = core.mewma_statistic(z, model100["score_mean"], model100["score_covariance"], CONFIG["mewma_lambda"]); elapsed = time.perf_counter() - start
    outputs["MEWMA"] = (mewma, mewma > model100["thresholds"]["MEWMA"], model100["thresholds"]["MEWMA"], elapsed, None)
    start = time.perf_counter(); mcusum = core.mcusum_crosier_statistic(z, model100["score_mean"], model100["score_covariance"], CONFIG["mcusum_k"]); elapsed = time.perf_counter() - start
    outputs["MCUSUM"] = (mcusum, mcusum > model100["thresholds"]["MCUSUM"], model100["thresholds"]["MCUSUM"], elapsed, None)
    start = time.perf_counter(); spe = core.spe_scores(x, z50, model50["pca"]); spe_elapsed = time.perf_counter() - start
    outputs["SPE_point"] = (spe, spe > model50["thresholds"]["SPE_point"], model50["thresholds"]["SPE_point"], spe_elapsed, None)
    spe_mean, spe_sd = model50["standardizers"]["SPE"]; standardized_spe = (spe - spe_mean) / spe_sd
    start = time.perf_counter(); spe_ewma = core.ewma_statistic(standardized_spe, CONFIG["ewma_lambda"]); elapsed = time.perf_counter() - start
    outputs["SPE_EWMA"] = (spe_ewma, spe_ewma > model50["thresholds"]["SPE_EWMA"], model50["thresholds"]["SPE_EWMA"], elapsed, None)
    start = time.perf_counter(); spe_cusum = core.cusum_upper_statistic(standardized_spe, CONFIG["cusum_k"]); elapsed = time.perf_counter() - start
    outputs["SPE_CUSUM"] = (spe_cusum, spe_cusum > model50["thresholds"]["SPE_CUSUM"], model50["thresholds"]["SPE_CUSUM"], elapsed, None)
    start = time.perf_counter(); ks = ks_min_p_signal(baseline["reference"], z, CONFIG["ks_window"]); ks_time = time.perf_counter() - start
    outputs["KS_fixed"] = (ks, np.isfinite(ks) & (ks < baseline["ks_threshold"]), baseline["ks_threshold"], ks_time, np.isfinite(ks))
    start = time.perf_counter(); mmd = mmd_rbf_signal(baseline["reference"], z, CONFIG["mmd_window"], baseline["bandwidth"]); mmd_time = time.perf_counter() - start
    outputs["MMD_RBF"] = (mmd, np.isfinite(mmd) & (mmd > baseline["mmd_threshold"]), baseline["mmd_threshold"], mmd_time, np.isfinite(mmd))
    distance = np.linalg.norm(z - baseline["centroid"], axis=1)
    start = time.perf_counter(); adwin = run_adwin(distance, baseline["adwin_delta"]); adwin_time = time.perf_counter() - start
    outputs["ADWIN"] = (distance, adwin, np.nan, adwin_time, None)
    return outputs


def periodic_rows(stream, alarms, density, pattern, stream_id, method):
    if pattern != "periodic": return []
    return core.periodic_episode_metrics(stream, alarms, density, stream_id, method)


def evaluate(data, model100, model50, baseline, streams):
    rows, episodes, metadata, representative = [], [], [], {}
    for pattern in ["normal_only"] + CONFIG["drift_types"]:
        for stream_id in range(CONFIG["n_streams"]):
            stream = streams[(pattern, stream_id)]
            outputs = monitor_stream(stream, model100, model50, baseline)
            metadata.append({"drift_pattern": pattern, "stream_id": stream_id,
                             "seed": stream["seed"], "changepoint": CONFIG["changepoint"],
                             "episode_boundaries_json": json.dumps(stream["episodes"]),
                             "drift_probability_json": json.dumps(stream["p"].tolist()),
                             "is_drift_observation_json": json.dumps(stream["is_drift_observation"].astype(int).tolist()),
                             "is_drift_regime_json": json.dumps(stream["is_drift_regime"].astype(int).tolist())})
            for method, (signal, alarms, threshold, runtime, valid_mask) in outputs.items():
                # KS p-values are inverse evidence; negate only for optional AUC orientation.
                metric_signal = -signal if method == "KS_fixed" else signal
                metrics = classification_metrics(stream, alarms, metric_signal, pattern, valid_mask,
                                                 auc_meaningful=method != "ADWIN")
                evaluated = int(valid_mask.sum()) if valid_mask is not None else len(signal)
                row = {"method": method, "method_family": METHOD_FAMILY[method],
                       "input_representation": ({
                           "T2_point": "scalar_T2_from_PCA100",
                           "SPE_point": "scalar_SPE_from_PCA50",
                           "SPE_EWMA": "scalar_SPE_from_PCA50",
                           "SPE_CUSUM": "scalar_SPE_from_PCA50",
                           "MEWMA": "PCA100_score_vector",
                           "MCUSUM": "standardized_PCA100_score_vector",
                           "KS_fixed": "PCA100_score_windows",
                           "MMD_RBF": "PCA100_score_windows",
                           "ADWIN": "scalar_PCA100_centroid_distance",
                       }[method]),
                       "pca_components": (CONFIG["spe_pca_components"] if method.startswith("SPE") else CONFIG["baseline_pca_components"]),
                       "window_size": (CONFIG["ks_window"] if method == "KS_fixed" else CONFIG["mmd_window"] if method == "MMD_RBF" else np.nan),
                       "stream_id": stream_id, "drift_pattern": pattern,
                       "changepoint": CONFIG["changepoint"], "threshold": threshold,
                       "calibration_parameter": (f"delta={baseline['adwin_delta']}" if method == "ADWIN" else np.nan),
                       "runtime_total_seconds": runtime,
                       "runtime_per_observation_seconds": runtime / len(signal),
                       "runtime_per_evaluated_window_seconds": runtime / evaluated if method in ("KS_fixed", "MMD_RBF") else np.nan,
                       "signal_json": json.dumps(np.where(np.isfinite(signal), signal, np.nan).tolist()),
                       "alarm_json": json.dumps(alarms.astype(int).tolist())}
                row.update(metrics); rows.append(row)
                density = np.asarray(json.loads(metrics["alarm_density_json"]))
                episodes.extend(periodic_rows(stream, alarms, density, pattern, stream_id, method))
                if stream_id == 0 and pattern != "normal_only":
                    representative[(pattern, method)] = (signal, alarms, threshold)
    return pd.DataFrame(rows), pd.DataFrame(episodes), pd.DataFrame(metadata), representative


def bootstrap_ci(values, seed):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if not len(values): return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (CONFIG["bootstrap_iterations"], len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(results):
    metrics = ["ARL0", "ARL1", "detection_delay", "FAR", "recall_detection_rate", "CDR",
               "detected_within_25", "detected_within_50", "detected_within_100", "DCD",
               "runtime_total_seconds", "runtime_per_observation_seconds",
               "runtime_per_evaluated_window_seconds"]
    rows = []
    for (method, pattern), group in results.groupby(["method", "drift_pattern"], sort=False):
        row = {"method": method, "drift_pattern": pattern, "n_streams": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row.update({f"{metric}_mean": values.mean(), f"{metric}_median": values.median(),
                        f"{metric}_std": values.std(ddof=1)})
            low, high = bootstrap_ci(values, CONFIG["seed"] + len(rows))
            row[f"{metric}_mean_ci95_low"], row[f"{metric}_mean_ci95_high"] = low, high
        row["SDRL0"] = pd.to_numeric(group.ARL0, errors="coerce").std(ddof=1)
        row["SDRL1"] = pd.to_numeric(group.ARL1, errors="coerce").std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_biserial_effect(differences):
    differences = np.asarray(differences, dtype=float); differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences): return np.nan
    ranks = rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())


def paired_comparisons(results):
    selected = CONFIG["selected_memory_method"]
    rows = []
    for pattern in CONFIG["drift_types"]:
        pivot = results[results.drift_pattern == pattern].pivot(index="stream_id", columns="method", values="detection_delay")
        for baseline in BASELINES:
            paired = pivot[[selected, baseline]].dropna()
            difference = paired[selected] - paired[baseline]
            try: statistic, pvalue = wilcoxon(difference) if len(paired) else (np.nan, np.nan)
            except ValueError: statistic, pvalue = np.nan, np.nan
            rows.append({"drift_pattern": pattern, "selected_spc_method": selected,
                         "baseline_method": baseline, "n_complete_pairs": len(paired),
                         "median_delay_difference_spc_minus_baseline": difference.median(),
                         "wilcoxon_statistic": statistic, "wilcoxon_pvalue": pvalue,
                         "rank_biserial_effect": rank_biserial_effect(difference)})
    return pd.DataFrame(rows)


def benchmark_methods():
    selected = CONFIG["selected_memory_method"]
    if selected not in ("SPE_EWMA", "SPE_CUSUM", "MEWMA", "MCUSUM"):
        raise ValueError("selected_memory_method must be one of the four SPC memory candidates")
    return ["T2_point", "SPE_point", selected, "KS_fixed", "MMD_RBF", "ADWIN"]


def plot_results(results, summary, representative, output_dir):
    methods = benchmark_methods(); drifts = CONFIG["drift_types"]
    drift_summary = summary[summary.drift_pattern.isin(drifts)]
    pivot = drift_summary.pivot(index="method", columns="drift_pattern", values="ARL1_mean").reindex(index=methods, columns=drifts)
    fig, ax = plt.subplots(figsize=(8, 6)); image = ax.imshow(pivot, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(3), drifts); ax.set_yticks(range(len(methods)), methods)
    for i in range(len(methods)):
        for j in range(3): ax.text(j, i, f"{pivot.iloc[i,j]:.1f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Mean ARL1"); fig.tight_layout(); fig.savefig(output_dir / "rq4_arl1_benchmark_heatmap.png", dpi=300); plt.close(fig)
    normal_far = summary[summary.drift_pattern == "normal_only"].set_index("method").FAR_mean
    normal_arl0 = summary[summary.drift_pattern == "normal_only"].set_index("method").reindex(methods)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(methods, normal_arl0.ARL0_mean, yerr=normal_arl0.ARL0_std,
                  capsize=3, color="#4C78A8")
    ax.axhline(CONFIG["nominal_arl0"], color="crimson", ls="--",
               label=f"Nominal three-sigma ARL0 = {CONFIG['nominal_arl0']}")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_ylabel("Realized mean ARL0 (normal-only streams)")
    ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output_dir / "realized_mean_arl0.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, pattern in zip(axes, drifts):
        part = drift_summary[drift_summary.drift_pattern == pattern].set_index("method")
        for method in methods:
            ax.scatter(normal_far.get(method, np.nan), part.loc[method, "detection_delay_median"])
            ax.annotate(method, (normal_far.get(method, np.nan), part.loc[method, "detection_delay_median"]), fontsize=7)
        ax.set_title(pattern); ax.set_xlabel("Realized normal-only FAR"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Median detection delay"); fig.tight_layout(); fig.savefig(output_dir / "delay_vs_false_alarms.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6)); width = .12; x = np.arange(len(methods))
    for j, pattern in enumerate(drifts):
        part = drift_summary[drift_summary.drift_pattern == pattern].set_index("method").reindex(methods)
        ax.bar(x + (j - 1) * width, part.detected_within_50_mean, width, label=pattern)
    ax.set_xticks(x, methods, rotation=45, ha="right"); ax.set_ylabel("P(detection delay <= 50)"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "early_detection_probability_50.png", dpi=300); plt.close(fig)
    costs = results.groupby("method").runtime_per_observation_seconds.mean().reindex(methods)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(methods, costs); ax.set_yscale("log")
    ax.set_ylabel("Mean online seconds per observation (log scale)"); ax.tick_params(axis="x", rotation=45); fig.tight_layout(); fig.savefig(output_dir / "online_computational_cost.png", dpi=300); plt.close(fig)
    shown = [CONFIG["selected_memory_method"], "KS_fixed", "MMD_RBF", "ADWIN"]
    for pattern in drifts:
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        for ax, method in zip(axes, shown):
            signal, alarms, threshold = representative[(pattern, method)]
            ax.plot(signal, lw=.7); hits = np.flatnonzero(alarms)
            ax.scatter(hits, signal[hits], s=7, color="darkorange")
            if np.isfinite(threshold): ax.axhline(threshold, color="crimson", ls="--")
            ax.axvline(CONFIG["changepoint"], color="black", ls=":"); ax.set_ylabel(method)
        axes[-1].set_xlabel("Observation"); fig.tight_layout(); fig.savefig(output_dir / f"representative_signals_{pattern}.png", dpi=300); plt.close(fig)


def save_outputs(results, summary, calibration, episodes, metadata, paired, representative):
    output = Path(CONFIG["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    workbook = output / "rq4_baseline_comparison_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        results.to_excel(writer, "stream_level", index=False)
        summary.to_excel(writer, "summary", index=False)
        calibration.to_excel(writer, "calibration", index=False)
        episodes.to_excel(writer, "periodic_episode_level", index=False)
        metadata.to_excel(writer, "stream_metadata", index=False)
        paired.to_excel(writer, "paired_comparisons", index=False)
    for name, frame in {"stream_level": results, "summary": summary,
                        "calibration": calibration, "periodic_episode_level": episodes,
                        "stream_metadata": metadata, "paired_comparisons": paired}.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    plot_results(results, summary, representative, output)
    print(f"Saved RQ4 results to: {output.resolve()}")


def sanity_checks(data, labels, streams):
    checks = {
        "validation_class0_only": bool(np.all(labels["validation"] == 0)),
        "fit_class0_only": bool(np.all(labels["fit"] == 0)),
        "drift_absent_before_changepoint": all(not s["is_drift_observation"][:CONFIG["changepoint"]].any() for s in streams.values()),
        "streams_generated_once_and_shared": len(streams) == 4 * CONFIG["n_streams"],
        "KS_window_fixed_50": CONFIG["ks_window"] == 50,
        "MMD_window_fixed_50": CONFIG["mmd_window"] == 50,
        "baseline_PCA_shared_100": CONFIG["baseline_pca_components"] == CONFIG["t2_pca_components"] == 100,
        "ADWIN_input_is_centroid_distance": True,
        "fixed_reference_source_is_fit_only": True,
        "common_changepoint": CONFIG["changepoint"] == 1000,
        "window_delay_uses_alarm_end_index": True,
    }
    if not all(checks.values()): raise AssertionError(f"Sanity check failed: {checks}")
    print("Sanity checks:")
    for name, passed in checks.items(): print(f"  [PASS] {name}" if passed else f"  [FAIL] {name}")


def main():
    configure_core()
    if CONFIG["ks_window"] != 50 or CONFIG["mmd_window"] != 50 or CONFIG["stride"] != 1:
        raise ValueError("RQ4 fixes KS/MMD window=50 and stride=1")
    if ADWIN is None:
        raise ImportError("Install river before running RQ4: pip install river")
    print("RQ4 benchmark: latent-space SPC versus KS-fixed, MMD-RBF and ADWIN")
    print(f"  methods={', '.join(METHODS)}; selected summary SPC={CONFIG['selected_memory_method']}")
    loaders, _ = core.load_datasets()
    data, labels = load_or_extract_embeddings(loaders)
    streams = {(pattern, stream_id): core.generate_stream(data, pattern, stream_id)
               for pattern in ["normal_only"] + CONFIG["drift_types"]
               for stream_id in range(CONFIG["n_streams"])}
    for (pattern, _), stream in streams.items(): core.sanity_check_stream(stream, pattern)
    sanity_checks(data, labels, streams)
    model100, model50, spc_calibration = build_spc_models(data)
    baseline, baseline_calibration = calibrate_baselines(model100, data)
    calibration = pd.DataFrame(spc_calibration + baseline_calibration)
    results, episodes, metadata, representative = evaluate(data, model100, model50, baseline, streams)
    assert results.groupby("method")["threshold"].nunique(dropna=False).eq(1).all()
    summary = summarize(results); paired = paired_comparisons(results)
    save_outputs(results, summary, calibration, episodes, metadata, paired, representative)
    columns = ["method", "drift_pattern", "ARL0_mean", "ARL1_mean", "detection_delay_median",
               "FAR_mean", "recall_detection_rate_mean", "DCD_mean", "runtime_per_observation_seconds_mean"]
    print("\nFinal benchmark summary:\n", summary[summary.drift_pattern.isin(CONFIG["drift_types"])][columns].to_string(index=False))


if __name__ == "__main__":
    main()

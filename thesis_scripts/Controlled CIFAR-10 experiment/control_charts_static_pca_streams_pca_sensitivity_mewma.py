# -*- coding: utf-8 -*-
"""RQ1: memoryless versus memory control charts under composition drift.

CIFAR-10 class 0 defines Phase I/in-control and class 1 is the only source of
drift.  PCA and all six thresholds are fixed before any evaluation stream is
generated.  The experiment is deliberately configured from the single CONFIG
dictionary below.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet18_Weights, resnet18

warnings.filterwarnings("ignore")

CONFIG = {
    "random_seed": 42, "target_class": 0, "drift_class": 1,
    "fit_size": 1000, "validation_size": 1000,
    "normal_pool_size": 1000, "drift_pool_size": 1000,
    "n_eval_streams": 20, "stream_length": 2000, "changepoint": 1000,
    "incremental_transition": 500, "periodic_episode_length": 200,
    "pca_components": 50, "sigma_multiplier": 3.0,
    "ewma_lambda": 0.2, "cusum_k": 0.5,
    "alarm_density_window": 25, "alarm_density_gamma": 0.30,
    "batch_size": 64, "num_workers": 0, "feature_space": "resnet18_latent",
    "bootstrap_iterations": 1000,
    "output_dir": "control_charts_static_pca_streams_rq1_results",
}
METHODS = ["T2_point", "SPE_point", "T2_EWMA", "SPE_EWMA",
           "T2_CUSUM", "SPE_CUSUM"]
DRIFT_PATTERNS = ["sudden", "incremental", "periodic"]


def make_loader(dataset):
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False,
                      num_workers=CONFIG["num_workers"])


def load_datasets():
    """Create four disjoint, role-specific pools and retain source indices."""
    transform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                         transform=transform)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                        transform=transform)
    c0, c1 = CONFIG["target_class"], CONFIG["drift_class"]
    train0 = np.flatnonzero(np.asarray(train.targets) == c0)
    test0 = np.flatnonzero(np.asarray(test.targets) == c0)
    test1 = np.flatnonzero(np.asarray(test.targets) == c1)
    required = CONFIG["fit_size"] + CONFIG["validation_size"]
    if len(train0) < required or len(test0) < CONFIG["normal_pool_size"] or len(test1) < CONFIG["drift_pool_size"]:
        raise ValueError("CIFAR-10 does not contain enough observations for CONFIG")
    indices = {
        "fit": train0[:CONFIG["fit_size"]],
        "validation": train0[CONFIG["fit_size"]:required],
        "normal_pool": test0[:CONFIG["normal_pool_size"]],
        "drift_pool": test1[:CONFIG["drift_pool_size"]],
    }
    assert set(indices["fit"]).isdisjoint(indices["validation"])
    assert set(indices["normal_pool"]).isdisjoint(indices["drift_pool"])
    return ({k: make_loader(Subset(train if k in ("fit", "validation") else test, v))
             for k, v in indices.items()}, indices)


def extract_resnet18_latent(loader, model, device):
    """Extract frozen 512-dimensional ImageNet ResNet-18 representations."""
    xs, ys = [], []
    with torch.no_grad():
        for images, labels in loader:
            xs.append(model(images.to(device)).flatten(1).cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def build_feature_data(loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    data, labels = {}, {}
    for name, loader in loaders.items():
        data[name], labels[name] = extract_resnet18_latent(loader, model, device)
        print(f"  {name:12s}: {data[name].shape}")
    return data, labels


def t2_scores(z, mean, covariance_inverse):
    difference = z - mean
    return np.einsum("ij,jk,ik->i", difference, covariance_inverse, difference)


def spe_scores(x, z, pca):
    return np.square(x - pca.inverse_transform(z)).sum(axis=1)


def standardization(values):
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Validation statistic has zero/non-finite SD")
    return mean, sd


def ewma_statistic(z, lambda_):
    out, previous = np.empty(len(z), dtype=float), 0.0
    for i, value in enumerate(z):
        previous = lambda_ * value + (1 - lambda_) * previous
        out[i] = previous
    return out


def cusum_upper_statistic(z, k):
    out, previous = np.empty(len(z), dtype=float), 0.0
    for i, value in enumerate(z):
        previous = max(0.0, previous + value - k)
        out[i] = previous
    return out


def fit_phase1_model(x_fit, x_validation):
    """Fit PCA/reference on fit only; construct all chart limits on validation."""
    n_components = min(CONFIG["pca_components"], len(x_fit) - 1, x_fit.shape[1])
    pca = PCA(n_components=n_components, random_state=CONFIG["random_seed"])
    z_fit = pca.fit_transform(x_fit)
    z_validation = pca.transform(x_validation)
    score_mean = z_fit.mean(axis=0)
    covariance = np.atleast_2d(np.cov(z_fit, rowvar=False))
    model = {"pca": pca, "n_components": n_components, "score_mean": score_mean,
             "covariance_inverse": np.linalg.pinv(covariance)}
    base = {"T2": t2_scores(z_validation, score_mean, model["covariance_inverse"]),
            "SPE": spe_scores(x_validation, z_validation, pca)}
    model["standardizers"], model["validation_charts"] = {}, {}
    for statistic, values in base.items():
        mean, sd = standardization(values)
        model["standardizers"][statistic] = (mean, sd)
        z = (values - mean) / sd
        model["validation_charts"][f"{statistic}_point"] = values
        model["validation_charts"][f"{statistic}_EWMA"] = ewma_statistic(z, CONFIG["ewma_lambda"])
        model["validation_charts"][f"{statistic}_CUSUM"] = cusum_upper_statistic(z, CONFIG["cusum_k"])
    rows, model["thresholds"] = [], {}
    for method in METHODS:
        values = model["validation_charts"][method]
        mean, sd = standardization(values)
        threshold = mean + CONFIG["sigma_multiplier"] * sd
        model["thresholds"][method] = threshold
        rows.append({"monitoring_method": method, "validation_mean": mean,
                     "validation_sd": sd, "threshold": threshold,
                     "sigma_multiplier": CONFIG["sigma_multiplier"],
                     "n_validation": len(values)})
    return model, pd.DataFrame(rows)


def score_stream(model, x):
    """Apply the same fixed model and validation standardizers to one stream."""
    z_pca = model["pca"].transform(x)
    base = {"T2": t2_scores(z_pca, model["score_mean"], model["covariance_inverse"]),
            "SPE": spe_scores(x, z_pca, model["pca"])}
    charts = {}
    for statistic, values in base.items():
        mean, sd = model["standardizers"][statistic]
        standardized = (values - mean) / sd
        charts[f"{statistic}_point"] = values
        charts[f"{statistic}_EWMA"] = ewma_statistic(standardized, CONFIG["ewma_lambda"])
        charts[f"{statistic}_CUSUM"] = cusum_upper_statistic(standardized, CONFIG["cusum_k"])
    return charts


def seed_for(pattern, stream_id):
    code = {"normal_only": 10, "sudden": 20, "incremental": 30, "periodic": 40}[pattern]
    return CONFIG["random_seed"] * 100000 + code * 1000 + stream_id


def drift_probability(pattern):
    n, cp = CONFIG["stream_length"], CONFIG["changepoint"]
    p = np.zeros(n, dtype=float)
    episodes = []
    if pattern == "sudden":
        p[cp:] = 1.0; episodes = [(cp, n - 1)]
    elif pattern == "incremental":
        p[cp:] = np.minimum(1.0, (np.arange(n - cp) + 1) / CONFIG["incremental_transition"])
        episodes = [(cp, n - 1)]
    elif pattern == "periodic":
        length = CONFIG["periodic_episode_length"]
        for start in range(cp, n, 2 * length):
            end = min(start + length, n)
            p[start:end] = 1.0
            episodes.append((start, end - 1))
    elif pattern != "normal_only":
        raise ValueError(pattern)
    return p, episodes


def generate_stream(data, pattern, stream_id):
    """Sample a paired feature stream; the returned object is reused by all charts."""
    rng = np.random.default_rng(seed_for(pattern, stream_id))
    p, episodes = drift_probability(pattern)
    is_drift_observation = rng.random(len(p)) < p
    normal_indices = rng.integers(0, len(data["normal_pool"]), len(p))
    drift_indices = rng.integers(0, len(data["drift_pool"]), len(p))
    x = data["normal_pool"][normal_indices].copy()
    x[is_drift_observation] = data["drift_pool"][drift_indices[is_drift_observation]]
    is_drift_regime = np.zeros(len(p), dtype=bool)
    if pattern in ("sudden", "incremental"):
        is_drift_regime[CONFIG["changepoint"]:] = True
    elif pattern == "periodic":
        is_drift_regime = p == 1
    return {"x": x, "p": p, "is_drift_observation": is_drift_observation,
            "is_drift_regime": is_drift_regime, "episodes": episodes,
            "seed": seed_for(pattern, stream_id)}


def alarm_density(alarms):
    """Trailing alarm mean with min_periods=1 for the first W-1 points."""
    return pd.Series(np.asarray(alarms, dtype=float)).rolling(
        CONFIG["alarm_density_window"], min_periods=1).mean().to_numpy()


def first_run_length(alarms):
    found = np.flatnonzero(alarms)
    return (int(found[0] + 1), False) if len(found) else (len(alarms) + 1, True)


def safe_auc(y, score):
    try:
        return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan
    except ValueError:
        return np.nan


def stream_metrics(stream, alarms, scores, pattern):
    """Compute point, run-length and process-confirmation metrics.

    FAR uses all truly in-control evaluation indices. SR is the fraction of all
    Phase-II signals (indices >= changepoint) that occur at in-control-regime
    indices. CDR is recall restricted to actual class-1 observations.
    """
    regime = stream["is_drift_regime"].astype(int)
    actual = stream["is_drift_observation"].astype(int)
    pred = alarms.astype(int)
    tp = int(((regime == 1) & (pred == 1)).sum()); tn = int(((regime == 0) & (pred == 0)).sum())
    fp = int(((regime == 0) & (pred == 1)).sum()); fn = int(((regime == 1) & (pred == 0)).sum())
    density = alarm_density(alarms)
    arl0, arl0_censored = first_run_length(alarms if pattern == "normal_only" else alarms[:CONFIG["changepoint"]])
    post = alarms[CONFIG["changepoint"]:]
    arl1, arl1_censored = (np.nan, np.nan) if pattern == "normal_only" else first_run_length(post)
    first_alarm = np.nan if pattern == "normal_only" or arl1_censored else CONFIG["changepoint"] + arl1 - 1
    delay = np.nan if np.isnan(first_alarm) else first_alarm - CONFIG["changepoint"]
    confirmations = np.flatnonzero(density[CONFIG["changepoint"]:] >= CONFIG["alarm_density_gamma"])
    dcd_censored = pattern != "normal_only" and len(confirmations) == 0
    dcd = np.nan if pattern == "normal_only" or dcd_censored else int(confirmations[0])
    phase2 = np.arange(len(alarms)) >= CONFIG["changepoint"]
    phase2_signals = pred[phase2] == 1
    sr = (np.sum(phase2_signals & (regime[phase2] == 0)) / np.sum(phase2_signals)
          if np.sum(phase2_signals) else np.nan)
    cdr = pred[actual == 1].mean() if actual.sum() else np.nan
    drift_density = density[regime == 1]
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": accuracy_score(regime, pred),
        "balanced_accuracy": balanced_accuracy_score(regime, pred) if len(np.unique(regime)) == 2 else np.nan,
        "precision": precision_score(regime, pred, zero_division=0),
        "recall_detection_rate": recall_score(regime, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "FAR": fp / (tn + fp) if tn + fp else np.nan,
        "false_alarm_rate": fp / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(regime, pred, zero_division=0),
        "MCC": matthews_corrcoef(regime, pred) if len(np.unique(pred)) > 1 and len(np.unique(regime)) > 1 else 0.0,
        "ROC_AUC": safe_auc(regime, scores), "SR": sr, "CDR": cdr,
        "ARL0": arl0, "ARL0_censored": arl0_censored,
        "ARL1": arl1, "ARL1_censored": arl1_censored,
        "first_alarm_after_changepoint": first_alarm, "detection_delay": delay,
        "detected_within_25": bool(not np.isnan(delay) and delay < 25),
        "detected_within_50": bool(not np.isnan(delay) and delay < 50),
        "detected_within_100": bool(not np.isnan(delay) and delay < 100),
        "maximum_alarm_density_after_drift": float(np.max(density[CONFIG["changepoint"]:])) if pattern != "normal_only" else np.nan,
        "mean_alarm_density_during_drift": float(np.mean(drift_density)) if len(drift_density) else np.nan,
        "DCD": dcd, "DCD_censored": dcd_censored,
    }, density


def periodic_episode_metrics(stream, alarms, density, stream_id, method):
    rows, gamma = [], CONFIG["alarm_density_gamma"]
    for episode_id, (start, end) in enumerate(stream["episodes"], 1):
        alarm_hits = np.flatnonzero(alarms[start:end + 1])
        confirmations = np.flatnonzero(density[start:end + 1] >= gamma)
        next_start = stream["episodes"][episode_id][0] if episode_id < len(stream["episodes"]) else len(alarms)
        recovery = np.flatnonzero(density[end + 1:next_start] < gamma)
        rows.append({"drift_pattern": "periodic", "stream_id": stream_id,
                     "monitoring_method": method, "episode_id": episode_id,
                     "episode_start": start, "episode_end": end,
                     "episode_detected": bool(len(alarm_hits)),
                     "first_detection_delay": int(alarm_hits[0]) if len(alarm_hits) else np.nan,
                     "DCD_episode": int(confirmations[0]) if len(confirmations) else np.nan,
                     "DCD_episode_censored": not bool(len(confirmations)),
                     "recovery_delay": int(recovery[0] + 1) if len(recovery) else np.nan,
                     "recovery_censored": not bool(len(recovery))})
    return rows


def sanity_check_stream(stream, pattern):
    cp = CONFIG["changepoint"]
    assert not stream["is_drift_observation"][:cp].any()
    assert np.all((stream["p"] >= 0) & (stream["p"] <= 1))
    if pattern == "sudden": assert np.all(stream["p"][cp:] == 1)
    if pattern == "incremental":
        assert np.all(np.diff(stream["p"][cp:]) >= 0) and stream["p"][-1] == 1
    if pattern == "periodic":
        expected, _ = drift_probability("periodic")
        assert np.array_equal(stream["p"], expected)


def evaluate_all(data, model):
    rows, episode_rows, metadata, traces = [], [], [], {}
    for pattern in ["normal_only"] + DRIFT_PATTERNS:
        for stream_id in range(CONFIG["n_eval_streams"]):
            stream = generate_stream(data, pattern, stream_id)
            sanity_check_stream(stream, pattern)
            charts = score_stream(model, stream["x"])
            assert set(charts) == set(METHODS)
            metadata.append({"drift_pattern": pattern, "stream_id": stream_id,
                             "seed": stream["seed"], "stream_length": len(stream["p"]),
                             "changepoint": CONFIG["changepoint"],
                             "incremental_transition": CONFIG["incremental_transition"],
                             "periodic_episode_length": CONFIG["periodic_episode_length"],
                             "episode_boundaries_json": json.dumps(stream["episodes"]),
                             "drift_probability_json": json.dumps(stream["p"].tolist()),
                             "is_drift_observation_json": json.dumps(stream["is_drift_observation"].astype(int).tolist()),
                             "is_drift_regime_json": json.dumps(stream["is_drift_regime"].astype(int).tolist())})
            for method in METHODS:
                scores, threshold = charts[method], model["thresholds"][method]
                alarms = scores > threshold
                metrics, density = stream_metrics(stream, alarms, scores, pattern)
                assert np.all((density >= 0) & (density <= 1))
                row = {"drift_pattern": pattern, "stream_id": stream_id,
                       "monitoring_method": method, "threshold": threshold,
                       "feature_space": CONFIG["feature_space"],
                       "pca_components": model["n_components"],
                       "ewma_lambda": CONFIG["ewma_lambda"], "cusum_k": CONFIG["cusum_k"],
                       "alarm_density_window": CONFIG["alarm_density_window"],
                       "alarm_density_gamma": CONFIG["alarm_density_gamma"],
                       "alarm_density_json": json.dumps(density.tolist())}
                row.update(metrics); rows.append(row)
                if pattern == "periodic":
                    episode_rows.extend(periodic_episode_metrics(stream, alarms, density, stream_id, method))
                if stream_id == 0 and pattern != "normal_only":
                    traces[(pattern, method)] = {"scores": scores, "alarms": alarms,
                                                  "density": density, "stream": stream}
    result = pd.DataFrame(rows)
    episodes = pd.DataFrame(episode_rows)
    if not episodes.empty:
        detected = episodes.groupby(["stream_id", "monitoring_method"])["episode_detected"].agg(["sum", "count"])
        rate = (detected["sum"] / detected["count"]).rename("episode_detection_rate")
        result = result.merge(rate.reset_index(), on=["stream_id", "monitoring_method"], how="left")
    return result, episodes, pd.DataFrame(metadata), traces


def bootstrap_ci(values, seed):
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    if not len(values): return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (CONFIG["bootstrap_iterations"], len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def make_summary(results):
    metrics = ["ARL0", "ARL1", "detection_delay", "FAR", "SR", "CDR",
               "recall_detection_rate", "balanced_accuracy", "maximum_alarm_density_after_drift",
               "mean_alarm_density_during_drift", "DCD", "detected_within_25",
               "detected_within_50", "detected_within_100", "episode_detection_rate"]
    rows = []
    for (pattern, method), group in results.groupby(["drift_pattern", "monitoring_method"], sort=False):
        row = {"drift_pattern": pattern, "monitoring_method": method, "n_streams": len(group)}
        for metric in metrics:
            if metric not in group: continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row.update({f"{metric}_mean": values.mean(), f"{metric}_median": values.median(),
                        f"{metric}_std": values.std(ddof=1)})
            low, high = bootstrap_ci(values, CONFIG["random_seed"] + len(rows))
            row[f"{metric}_mean_ci95_low"], row[f"{metric}_mean_ci95_high"] = low, high
        row["SDRL0"] = pd.to_numeric(group["ARL0"], errors="coerce").std(ddof=1)
        row["SDRL1"] = pd.to_numeric(group["ARL1"], errors="coerce").std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(results, summary, traces, output_dir):
    """Create the focused publication figures requested for RQ1."""
    cp, colors = CONFIG["changepoint"], plt.cm.tab10.colors
    for source in ("T2", "SPE"):
        methods = [f"{source}_point", f"{source}_EWMA", f"{source}_CUSUM"]
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, method in zip(axes, methods):
            trace = traces[("sudden", method)]; ax.plot(trace["scores"], lw=.8)
            ax.axhline(results.loc[results.monitoring_method == method, "threshold"].iloc[0], color="crimson", ls="--")
            ax.axvline(cp, color="black", ls=":"); ax.set_ylabel(method)
        axes[-1].set_xlabel("Observation"); fig.tight_layout()
        fig.savefig(output_dir / f"example_sudden_{source}.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5)); stream = traces[("incremental", METHODS[0])]["stream"]
    ax.plot(stream["p"], color="black", lw=2, label="theoretical p(t)")
    for i, method in enumerate(METHODS):
        hits = np.flatnonzero(traces[("incremental", method)]["alarms"])
        ax.scatter(hits, np.full(len(hits), 1.08 + i * .055), s=5, color=colors[i], label=method)
    ax.set_ylim(-.05, 1.45); ax.set_xlabel("Observation"); ax.set_ylabel("Drift probability / alarms")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(output_dir / "example_incremental_alarms.png", dpi=300); plt.close(fig)
    drift = results[results.drift_pattern.isin(DRIFT_PATTERNS)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, pattern in zip(axes, DRIFT_PATTERNS):
        groups = [drift[(drift.drift_pattern == pattern) & (drift.monitoring_method == m)].detection_delay.dropna() for m in METHODS]
        ax.boxplot(groups, tick_labels=METHODS, showfliers=False); ax.tick_params(axis="x", rotation=70); ax.set_title(pattern)
    axes[0].set_ylabel("Detection delay"); fig.tight_layout(); fig.savefig(output_dir / "detection_delay_boxplots.png", dpi=300); plt.close(fig)
    arl = summary[summary.drift_pattern.isin(DRIFT_PATTERNS)]
    fig, ax = plt.subplots(figsize=(12, 6)); x = np.arange(len(METHODS)); width = .25
    for i, pattern in enumerate(DRIFT_PATTERNS):
        q = arl.set_index(["drift_pattern", "monitoring_method"]).loc[pattern].reindex(METHODS)
        ax.bar(x + (i - 1) * width, q.ARL1_mean, width, yerr=q.ARL1_std, label=pattern, capsize=2)
    ax.set_xticks(x, METHODS, rotation=45, ha="right"); ax.set_ylabel("Mean ARL1 (SD error bars)"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "arl1_summary.png", dpi=300); plt.close(fig)
    normal = summary[summary.drift_pattern == "normal_only"].set_index("monitoring_method").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(10, 5)); ax.bar(METHODS, normal.ARL0_mean); ax.axhline(370, color="crimson", ls="--", label="nominal 3-sigma ARL0 = 370")
    ax.tick_params(axis="x", rotation=45); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "realized_arl0.png", dpi=300); plt.close(fig)
    for pattern in ("sudden", "periodic"):
        method = "T2_EWMA"; trace = traces[(pattern, method)]; fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(trace["stream"]["p"], color="black"); axes[0].set_ylabel("p(t)")
        axes[1].step(np.arange(len(trace["alarms"])), trace["alarms"].astype(int), where="mid"); axes[1].set_ylabel("Alarm")
        axes[2].plot(trace["density"]); axes[2].axhline(CONFIG["alarm_density_gamma"], color="crimson", ls="--")
        axes[2].axvline(cp, color="black", ls=":"); confirmed = np.flatnonzero(trace["density"][cp:] >= CONFIG["alarm_density_gamma"])
        if len(confirmed): axes[2].axvline(cp + confirmed[0], color="green", ls="--")
        axes[2].set_ylabel("Alarm Density"); axes[2].set_xlabel("Observation"); fig.tight_layout()
        fig.savefig(output_dir / f"alarm_density_{pattern}.png", dpi=300); plt.close(fig)
    pivot = arl.pivot(index="monitoring_method", columns="drift_pattern", values="ARL1_mean").reindex(index=METHODS, columns=DRIFT_PATTERNS)
    fig, ax = plt.subplots(figsize=(7, 5)); image = ax.imshow(pivot, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(3), DRIFT_PATTERNS); ax.set_yticks(range(6), METHODS)
    for i in range(6):
        for j in range(3): ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Mean ARL1"); fig.tight_layout(); fig.savefig(output_dir / "arl1_heatmap.png", dpi=300); plt.close(fig)


def save_results(results, summary, episodes, thresholds, metadata, traces):
    output_dir = Path(CONFIG["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    workbook = output_dir / "rq1_control_chart_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        results.to_excel(writer, "stream_level", index=False)
        summary.to_excel(writer, "summary", index=False)
        episodes.to_excel(writer, "periodic_episode_level", index=False)
        thresholds.to_excel(writer, "thresholds", index=False)
        metadata.to_excel(writer, "stream_metadata", index=False)
    for name, frame in {"stream_level": results, "summary": summary,
                        "periodic_episode_level": episodes, "thresholds": thresholds,
                        "stream_metadata": metadata}.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    plot_results(results, summary, traces, output_dir)
    print(f"\nSaved workbook and figures to: {output_dir.resolve()}")


def validate_config():
    assert CONFIG["stream_length"] == 2000 and CONFIG["changepoint"] == 1000
    assert CONFIG["target_class"] == 0 and CONFIG["drift_class"] == 1
    assert 0 < CONFIG["ewma_lambda"] <= 1 and CONFIG["alarm_density_window"] > 0
    assert 0 <= CONFIG["alarm_density_gamma"] <= 1


def main():
    validate_config(); np.random.seed(CONFIG["random_seed"]); torch.manual_seed(CONFIG["random_seed"])
    print("RQ1 experiment: class-composition drift only")
    print(f"  PCA={CONFIG['pca_components']} PCs; methods={', '.join(METHODS)}")
    print(f"  {CONFIG['n_eval_streams']} paired streams/pattern + normal-only; length={CONFIG['stream_length']}; changepoint={CONFIG['changepoint']}")
    loaders, pool_indices = load_datasets()
    data, labels = build_feature_data(loaders)
    assert np.all(labels["fit"] == 0) and np.all(labels["validation"] == 0)
    assert np.all(labels["normal_pool"] == 0) and np.all(labels["drift_pool"] == 1)
    model, thresholds = fit_phase1_model(data["fit"], data["validation"])
    results, episodes, metadata, traces = evaluate_all(data, model)
    assert results.groupby("monitoring_method")["threshold"].nunique().eq(1).all()
    summary = make_summary(results)
    save_results(results, summary, episodes, thresholds, metadata, traces)
    key = summary[summary.drift_pattern.isin(DRIFT_PATTERNS)][["drift_pattern", "monitoring_method", "ARL1_mean", "detection_delay_median", "DCD_median", "FAR_mean"]]
    print("\nKey results:\n", key.to_string(index=False))


if __name__ == "__main__":
    main()


1+1



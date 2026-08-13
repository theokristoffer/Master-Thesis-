"""Seed-paired statistical analysis for the observation-level calibration benchmark."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _normalise_parameter(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_pipeline(root: Path, dataset: str, pipeline: str):
    if pipeline == "static_pca":
        episode_path = root / dataset / pipeline / "aggregate" / "arl1_episode_results_all_seeds.csv"
        arl0_path = root / dataset / pipeline / "aggregate" / "heldout_arl0_all_seeds.csv"
    else:
        episode_path = root / dataset / pipeline / "aggregate" / "mwpca_episode_results_all_seeds.csv"
        arl0_path = root / dataset / pipeline / "aggregate" / "mwpca_heldout_arl0_all_seeds.csv"
    if not episode_path.exists() or not arl0_path.exists():
        return None, None
    episodes = pd.read_csv(episode_path)
    arl0 = pd.read_csv(arl0_path)
    episodes["pipeline"] = pipeline
    arl0["pipeline"] = pipeline
    if pipeline == "static_pca":
        episodes["parameter"] = _normalise_parameter(episodes.get("ewma_lambda", np.nan))
        arl0["parameter"] = _normalise_parameter(arl0.get("ewma_lambda", np.nan))
        episodes.loc[episodes.method != "EWMA", "parameter"] = np.nan
        arl0.loc[arl0.method != "EWMA", "parameter"] = np.nan
    else:
        episodes["parameter"] = _normalise_parameter(episodes.get("parameter", np.nan))
        arl0["parameter"] = _normalise_parameter(arl0.get("parameter", np.nan))
    return episodes, arl0


def _config_label(method: str, parameter) -> str:
    return str(method) if pd.isna(parameter) else f"{method}(lambda={float(parameter):g})"


def _bootstrap_mean_ci(values, rng, reps=2000):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    means = np.empty(reps)
    for index in range(reps):
        means[index] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def _bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(ranked, 0, 1)
    return q


def analyse(root: Path, output: Path, bootstrap_reps: int = 2000,
            seed: int = 20260810):
    episode_frames, arl0_frames = [], []
    for dataset in ("cifar10", "patchcamelyon"):
        for pipeline in ("static_pca", "mwpca_all_features"):
            episodes, arl0 = _load_pipeline(root, dataset, pipeline)
            if episodes is not None:
                episode_frames.append(episodes)
                arl0_frames.append(arl0)
    if not episode_frames:
        raise FileNotFoundError(f"No completed aggregate results found below {root}")
    episodes = pd.concat(episode_frames, ignore_index=True)
    arl0 = pd.concat(arl0_frames, ignore_index=True)
    episodes["config"] = [
        _config_label(method, parameter)
        for method, parameter in zip(episodes.method, episodes.parameter)
    ]
    arl0["config"] = [
        _config_label(method, parameter)
        for method, parameter in zip(arl0.method, arl0.parameter)
    ]
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    exact = [
        "pipeline", "dataset", "pattern", "mechanism", "severity", "block",
        "method", "parameter", "config",
    ]
    # Seed is the independent unit. This also remains valid if later runs use
    # more than one episode, because episodes are averaged within seed first.
    seed_level = episodes.groupby(exact + ["seed"], dropna=False).agg(
        restricted_delay=("restricted_delay_observations", "mean"),
        detected=("detected", "mean"),
        detected_delay=("arl1_delay_observations", "mean"),
        episodes=("replication", "nunique"),
    ).reset_index()
    seed_level.to_csv(output / "seed_level_outcomes.csv", index=False)

    summary_rows = []
    for key, group in seed_level.groupby(exact, dropna=False):
        delay = group.restricted_delay.to_numpy(float)
        detection = group.detected.to_numpy(float)
        d_lo, d_hi = _bootstrap_mean_ci(delay, rng, bootstrap_reps)
        p_lo, p_hi = _bootstrap_mean_ci(detection, rng, bootstrap_reps)
        row = dict(zip(exact, key if isinstance(key, tuple) else (key,)))
        row.update({
            "seeds": int(group.seed.nunique()),
            "episodes_per_seed": int(group.episodes.min()),
            "restricted_mean_delay": float(delay.mean()),
            "restricted_delay_median": float(np.median(delay)),
            "restricted_delay_ci_low": d_lo,
            "restricted_delay_ci_high": d_hi,
            "detection_rate": float(detection.mean()),
            "detection_rate_ci_low": p_lo,
            "detection_rate_ci_high": p_hi,
        })
        summary_rows.append(row)
    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(output / "method_condition_summaries.csv", index=False)

    comparison_condition = [
        "pipeline", "dataset", "pattern", "mechanism", "severity", "block"
    ]
    comparison_rows = []
    for condition_key, condition in seed_level.groupby(comparison_condition, dropna=False):
        configs = sorted(condition.config.unique())
        for config_a, config_b in combinations(configs, 2):
            a = condition[condition.config == config_a][["seed", "restricted_delay", "detected"]]
            b = condition[condition.config == config_b][["seed", "restricted_delay", "detected"]]
            paired = a.merge(b, on="seed", suffixes=("_a", "_b"))
            if len(paired) < 2:
                continue
            differences = paired.restricted_delay_a - paired.restricted_delay_b
            ci_low, ci_high = _bootstrap_mean_ci(differences, rng, bootstrap_reps)
            try:
                wilcoxon_p = float(stats.wilcoxon(
                    differences, zero_method="pratt", alternative="two-sided"
                ).pvalue) if np.any(differences != 0) else 1.0
            except ValueError:
                wilcoxon_p = np.nan
            a_only = int(((paired.detected_a > 0) & (paired.detected_b == 0)).sum())
            b_only = int(((paired.detected_b > 0) & (paired.detected_a == 0)).sum())
            discordant = a_only + b_only
            mcnemar_p = (
                float(stats.binomtest(min(a_only, b_only), discordant, 0.5).pvalue)
                if discordant else 1.0
            )
            row = dict(zip(
                comparison_condition,
                condition_key if isinstance(condition_key, tuple) else (condition_key,),
            ))
            row.update({
                "config_a": config_a,
                "config_b": config_b,
                "paired_seeds": int(len(paired)),
                "mean_restricted_delay_difference_a_minus_b": float(differences.mean()),
                "median_restricted_delay_difference_a_minus_b": float(np.median(differences)),
                "paired_difference_ci_low": ci_low,
                "paired_difference_ci_high": ci_high,
                "proportion_a_faster": float((differences < 0).mean()),
                "ties": int((differences == 0).sum()),
                "wilcoxon_p": wilcoxon_p,
                "detected_a_only": a_only,
                "detected_b_only": b_only,
                "mcnemar_exact_p": mcnemar_p,
            })
            comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    if len(comparisons):
        comparisons["wilcoxon_q_bh"] = _bh_adjust(comparisons.wilcoxon_p)
        comparisons["mcnemar_q_bh"] = _bh_adjust(comparisons.mcnemar_exact_p)
    comparisons.to_csv(output / "paired_method_comparisons.csv", index=False)

    ranking_keys = comparison_condition
    rankings = summaries.copy()
    rankings["condition_rank"] = rankings.groupby(ranking_keys, dropna=False)[
        "restricted_mean_delay"
    ].rank(method="min", ascending=True)
    rankings.to_csv(output / "condition_rankings.csv", index=False)
    winners = rankings[rankings.condition_rank == 1].groupby(
        ["pipeline", "dataset", "block", "config"], dropna=False
    ).size().rename("conditions_won").reset_index()
    totals = rankings.groupby(
        ["pipeline", "dataset", "block"], dropna=False
    )[ranking_keys[-3]].count().rename("condition_config_rows").reset_index()
    winners = winners.merge(totals, on=["pipeline", "dataset", "block"], how="left")
    winners.to_csv(output / "winner_counts.csv", index=False)

    arl0_seed = arl0.groupby(
        ["pipeline", "dataset", "block", "method", "parameter", "config", "seed"],
        dropna=False,
    ).arl0_observations.mean().reset_index()
    delay_across_conditions = seed_level.groupby(
        ["pipeline", "dataset", "block", "method", "parameter", "config", "seed"],
        dropna=False,
    ).restricted_delay.mean().reset_index()
    operating = delay_across_conditions.merge(
        arl0_seed,
        on=["pipeline", "dataset", "block", "method", "parameter", "config", "seed"],
        how="inner",
    )
    operating.to_csv(output / "operating_point_seed_data.csv", index=False)
    association_rows = []
    group_keys = ["pipeline", "dataset", "block", "method", "parameter", "config"]
    for key, group in operating.groupby(group_keys, dropna=False):
        rho, pvalue = (
            stats.spearmanr(group.arl0_observations, group.restricted_delay)
            if len(group) >= 3 else (np.nan, np.nan)
        )
        row = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        row.update({"seeds": len(group), "spearman_rho": rho, "spearman_p": pvalue})
        association_rows.append(row)
    associations = pd.DataFrame(association_rows)
    if len(associations):
        associations["spearman_q_bh"] = _bh_adjust(associations.spearman_p)
    associations.to_csv(output / "arl0_delay_associations.csv", index=False)

    metadata = {
        "analysis_unit": "seed",
        "primary_estimand": "restricted mean detection delay at the Phase-II horizon",
        "paired_design": True,
        "bootstrap_reps": int(bootstrap_reps),
        "multiplicity_adjustment": "Benjamini-Hochberg FDR",
        "interpretation": "exploratory when only one Phase-II episode is used per seed",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))
    return {
        "seed_level": seed_level,
        "summaries": summaries,
        "comparisons": comparisons,
        "rankings": rankings,
        "winners": winners,
        "operating_point": operating,
        "associations": associations,
    }


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--root", type=Path,
        default=base / "outputs/observation_level_calibration",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    output = args.output or args.root / "statistical_analysis"
    result = analyse(args.root, output, args.bootstrap_reps)
    print(f"Saved statistical analysis to {output}")
    print(result["summaries"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

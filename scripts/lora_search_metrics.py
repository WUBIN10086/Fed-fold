"""Cluster-aware metrics and baseline-safe LoRA model selection."""

from collections import defaultdict
from statistics import fmean
from typing import Dict, Iterable, Mapping, Sequence


def cluster_macro_metrics(
    tm_by_label: Mapping[str, float],
    label_to_cluster: Mapping[str, str],
    baseline_tm_by_label: Mapping[str, float] = None,
) -> Dict[str, object]:
    if not tm_by_label:
        raise ValueError("tm_by_label must not be empty")
    cluster_values = defaultdict(list)
    for label, value in tm_by_label.items():
        cluster = label_to_cluster.get(label.upper(), f"singleton:{label.upper()}")
        cluster_values[str(cluster)].append(float(value))
    cluster_means = {
        cluster: fmean(values)
        for cluster, values in sorted(cluster_values.items())
    }
    result = {
        "sample_mean_tm": fmean(float(value) for value in tm_by_label.values()),
        "cluster_macro_tm": fmean(cluster_means.values()),
        "cluster_count": len(cluster_means),
        "cluster_tm": cluster_means,
    }
    if baseline_tm_by_label is not None:
        missing = sorted(set(tm_by_label) - set(baseline_tm_by_label))
        if missing:
            raise ValueError(f"Baseline is missing labels: {missing[:5]}")
        baseline_cluster_values = defaultdict(list)
        for label in tm_by_label:
            cluster = label_to_cluster.get(
                label.upper(),
                f"singleton:{label.upper()}",
            )
            baseline_cluster_values[str(cluster)].append(
                float(baseline_tm_by_label[label])
            )
        baseline_cluster_means = {
            cluster: fmean(values)
            for cluster, values in sorted(baseline_cluster_values.items())
        }
        deltas = {
            cluster: cluster_means[cluster] - baseline_cluster_means[cluster]
            for cluster in cluster_means
        }
        result.update({
            "cluster_delta": deltas,
            "worst_cluster_delta": min(deltas.values()),
            "baseline_cluster_macro_tm": fmean(
                baseline_cluster_means.values()
            ),
        })
    return result


def select_model_with_baseline(
    candidates: Iterable[Mapping[str, object]],
    baseline: Mapping[str, object],
    min_delta: float = 0.0,
) -> Dict[str, object]:
    baseline_score = float(baseline["cluster_macro_tm"])
    eligible = [
        dict(candidate)
        for candidate in candidates
        if float(candidate["cluster_macro_tm"])
        >= baseline_score + float(min_delta)
    ]
    if not eligible:
        selected = dict(baseline)
        selected["selected_model"] = "baseline"
        selected["selection_reason"] = "no_lora_met_cluster_macro_threshold"
        return selected
    eligible.sort(
        key=lambda row: (
            -float(row["cluster_macro_tm"]),
            -float(row.get("sample_mean_tm", float("-inf"))),
            str(row.get("candidate_id", "")),
        )
    )
    selected = eligible[0]
    selected["selected_model"] = selected.get("candidate_id", "lora")
    selected["selection_reason"] = "best_eligible_cluster_macro_tm"
    return selected


def scale_grid(values: Sequence[float] = (0, 0.1, 0.25, 0.5, 0.75, 1.0)):
    scales = tuple(float(value) for value in values)
    if not scales or scales[0] != 0 or any(value < 0 for value in scales):
        raise ValueError("Scale grid must begin with zero and be nonnegative")
    if len(set(scales)) != len(scales):
        raise ValueError("Scale grid contains duplicates")
    return scales

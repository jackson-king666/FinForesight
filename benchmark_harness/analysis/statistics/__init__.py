"""Deterministic, dependency-light statistics used by the offline reports."""

from benchmark_harness.analysis.statistics.bootstrap import paired_bootstrap
from benchmark_harness.analysis.statistics.cluster import (
    cluster_bootstrap,
    cluster_bootstrap_estimates,
    paired_cluster_bootstrap,
)
from benchmark_harness.analysis.statistics.effect_size import paired_standardized_effect
from benchmark_harness.analysis.statistics.multiple_testing import holm_bonferroni
from benchmark_harness.analysis.statistics.proxy import c2_proxy
from benchmark_harness.analysis.statistics.ranking import (
    kendall_tau,
    pairwise_win_matrix,
    pearson_correlation,
    spearman_correlation,
)

__all__ = [
    "c2_proxy",
    "cluster_bootstrap",
    "cluster_bootstrap_estimates",
    "holm_bonferroni",
    "kendall_tau",
    "paired_bootstrap",
    "paired_cluster_bootstrap",
    "paired_standardized_effect",
    "pairwise_win_matrix",
    "pearson_correlation",
    "spearman_correlation",
]

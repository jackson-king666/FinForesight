"""Golden checks for formal-utility/v1 metrics and the cluster bootstrap."""

from __future__ import annotations

import hashlib
import json
import math
import random
from statistics import mean
from typing import Any

import pytest

from benchmark_harness.analysis.contracts import AnalysisError
from benchmark_harness.analysis.metrics.formal import (
    FORMAL_UTILITY_VERSION,
    a1_formal_utility,
    a2_formal_utility,
    b_formal_utility,
    b_probability,
    brier_skill_score_unclipped,
    c_formal_utility,
    e_formal_utility,
    operational_utility,
)
from benchmark_harness.analysis.statistics.cluster import (
    cluster_bootstrap,
    cluster_bootstrap_estimates,
    make_cluster_key,
    paired_cluster_bootstrap,
)
from benchmark_harness.reference_evaluation.engine import (
    evaluate_a1,
    evaluate_a2,
    evaluate_b,
    evaluate_c,
    evaluate_e,
    parse_a1_response,
    parse_a2_response,
    parse_b_response,
    parse_c_response,
    parse_e_response,
)


def _a1_metrics(
    actual_prices: dict[str, float], *, base: float = 100.0, bull: float = 110.0, bear: float = 90.0
) -> dict[str, Any]:
    response = json.dumps(
        {"bull": bull, "base": base, "bear": bear, "reversion_horizon": "1-3个月"},
        ensure_ascii=False,
    )
    parsed = parse_a1_response(response)
    assert parsed["format_valid"] is True
    return evaluate_a1(parsed, {"ground_truth": {"actual_prices": actual_prices}})


def _a2_metrics(ranking: Any) -> dict[str, Any]:
    parsed = parse_a2_response(ranking if isinstance(ranking, str) else json.dumps(ranking))
    record = {
        "seed": {"stock_list": [{"code": "AAA"}, {"code": "BBB"}, {"code": "CCC"}]},
        "ground_truth": {
            "actual_returns": {"AAA": 0.01, "BBB": 0.02, "CCC": 0.03},
            "actual_ranking": ["AAA", "BBB", "CCC"],
        },
    }
    return evaluate_a2(parsed, record)


def _b_metrics(payload: dict[str, Any], actual_direction: str = "up") -> dict[str, Any]:
    parsed = parse_b_response(json.dumps(payload))
    return evaluate_b(parsed, {"ground_truth": {"actual_direction": actual_direction}})


def _c_metrics(predicted: float, future: float) -> dict[str, Any]:
    parsed = parse_c_response(json.dumps({"predicted_value": predicted}))
    record = {"seed": {"metric_name": "revenue"}, "ground_truth": {"future_value": future}}
    return evaluate_c(parsed, record)


def _e_metrics(answer: Any, correct: float) -> dict[str, Any]:
    parsed = parse_e_response(json.dumps({"formula_used": "f", "answer": answer, "unit": "u"}))
    record = {
        "ground_truth": {"correct_answer": correct, "correct_formula": "f", "answer_unit": "u"}
    }
    return evaluate_e(parsed, record)


def test_formal_utility_version_identity() -> None:
    assert FORMAL_UTILITY_VERSION == "formal-utility/v1"


def test_a1_bull_base_bear_golden_30_day_endpoint() -> None:
    metrics = _a1_metrics({"30": 105.0, "90": 103.0, "180": 99.0, "365": 130.0})
    assert metrics["by_window"]["30"]["target_price_error_abs"] == pytest.approx(0.05)
    assert metrics["by_window"]["30"]["target_price_error_mode"] == "relative_error"
    assert metrics["by_window"]["365"]["target_price_error_abs"] == pytest.approx(0.30)
    value, reason = a1_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(0.9)  # 1 - 2*min(0.05, 1); the 0.30 error at 365d is ignored


def test_a1_pending_long_horizons_do_not_block_30_day_endpoint() -> None:
    metrics = _a1_metrics({"30": 105.0})
    for window in ("90", "180", "365"):
        assert metrics["by_window"][window]["target_price_error_abs"] is None
    value, reason = a1_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(0.9)


def test_a1_missing_30_day_window_is_unavailable() -> None:
    metrics = _a1_metrics({"90": 103.0, "180": 99.0, "365": 130.0})
    assert a1_formal_utility(metrics) == (None, "a1_window_30_unavailable")
    assert a1_formal_utility({"by_window": {}}) == (None, "a1_window_30_unavailable")
    assert a1_formal_utility({}) == (None, "a1_window_30_unavailable")


def test_a1_error_clipped_at_one() -> None:
    metrics = _a1_metrics({"30": 250.0})
    assert metrics["by_window"]["30"]["target_price_error_abs"] == pytest.approx(1.5)
    value, reason = a1_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(-1.0)


def test_a1_base_zero_absolute_mode_is_not_a_percentage_error() -> None:
    metrics = _a1_metrics({"30": 0.5}, base=0.0, bull=1.0, bear=-1.0)
    assert metrics["by_window"]["30"]["target_price_error_mode"] == "absolute_error_base_zero"
    assert a1_formal_utility(metrics) == (None, "a1_error_mode_absolute_base_zero")


def test_a1_unknown_mode_and_non_finite_error_rejected() -> None:
    unknown_mode = {
        "by_window": {
            "30": {
                "target_price_error_abs": 0.05,
                "target_price_error_mode": "absolute",
                "range_hit": True,
            }
        }
    }
    assert a1_formal_utility(unknown_mode) == (None, "a1_error_mode_unknown")
    non_finite = {
        "by_window": {
            "30": {
                "target_price_error_abs": math.inf,
                "target_price_error_mode": "relative_error",
                "range_hit": False,
            }
        }
    }
    assert a1_formal_utility(non_finite) == (None, "a1_error_not_finite")


def test_a2_spearman_passthrough_golden() -> None:
    metrics = _a2_metrics(["BBB", "AAA", "CCC"])
    assert metrics["exact_set_match"] is True
    assert metrics["spearman_rho"] == pytest.approx(0.5)
    value, reason = a2_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(0.5)


def test_a2_constituent_set_mismatch_stays_unavailable() -> None:
    mismatched = _a2_metrics(["AAA", "BBB", "XXX"])
    assert mismatched["exact_set_match"] is False
    assert a2_formal_utility(mismatched) == (None, "a2_constituent_set_mismatch")
    unparsed = _a2_metrics("not json at all")
    assert unparsed["exact_set_match"] is None
    assert a2_formal_utility(unparsed) == (None, "a2_constituent_set_mismatch")


def test_a2_missing_spearman_is_unavailable() -> None:
    assert a2_formal_utility({"exact_set_match": True, "spearman_rho": None}) == (
        None,
        "a2_spearman_unavailable",
    )


def test_b_probability_valid_copy() -> None:
    parsed = parse_b_response(json.dumps({"direction": "up", "probability_up": 0.6}))
    value, reason = b_probability(parsed)
    assert reason is None
    assert value == pytest.approx(0.6)


def test_b_probability_direction_only_is_never_synthesized() -> None:
    parsed = parse_b_response(json.dumps({"direction": "up"}))
    assert parsed["format_valid"] is False
    value, reason = b_probability(parsed)
    assert value is None
    assert value not in (0.0, 1.0)
    assert reason == "b_probability_missing"
    for direction in ("up", "down"):
        value, reason = b_probability({"direction": direction})
        assert (value, reason) == (None, "b_probability_missing")


def test_b_probability_parser_rejects_out_of_range_as_missing() -> None:
    parsed = parse_b_response(json.dumps({"direction": "up", "probability_up": 1.2}))
    assert parsed["format_valid"] is False
    assert b_probability(parsed) == (None, "b_probability_missing")


@pytest.mark.parametrize("bad", [1.2, -0.1, math.nan, math.inf, -math.inf])
def test_b_probability_out_of_range(bad: float) -> None:
    assert b_probability({"direction": "up", "probability_up": bad}) == (
        None,
        "b_probability_out_of_range",
    )


@pytest.mark.parametrize("bad", [None, "0.7", True])
def test_b_probability_non_numeric_is_missing(bad: Any) -> None:
    assert b_probability({"direction": "up", "probability_up": bad}) == (
        None,
        "b_probability_missing",
    )


def test_b_formal_utility_golden() -> None:
    metrics = _b_metrics({"direction": "up", "probability_up": 0.6}, actual_direction="up")
    assert metrics["brier_score"] == pytest.approx(0.16)
    value, reason = b_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(0.68)


def test_b_formal_utility_missing_brier() -> None:
    metrics = _b_metrics({"direction": "up"})
    assert metrics["brier_score"] is None
    assert b_formal_utility(metrics) == (None, "b_brier_unavailable")


@pytest.mark.parametrize(
    ("brier", "expected"),
    [(0.30, -0.2), (0.25, 0.0), (0.0, 1.0), (0.16, 0.36)],
)
def test_brier_skill_score_unclipped_golden(brier: float, expected: float) -> None:
    assert brier_skill_score_unclipped(brier) == pytest.approx(expected)


def test_brier_skill_score_unclipped_rejects_non_finite() -> None:
    with pytest.raises(AnalysisError, match="finite"):
        brier_skill_score_unclipped(math.nan)


def test_c_formal_utility_golden() -> None:
    metrics = _c_metrics(210.0, 200.0)
    assert metrics["mape"] == pytest.approx(0.05)
    value, reason = c_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(0.9)


def test_c_formal_utility_clipped_at_one() -> None:
    metrics = _c_metrics(250.0, 100.0)
    assert metrics["mape"] == pytest.approx(1.5)
    value, reason = c_formal_utility(metrics)
    assert reason is None
    assert value == pytest.approx(-1.0)


def test_c_zero_denominator_rule_matches_frozen_engine() -> None:
    unavailable = _c_metrics(5.0, 0.0)
    assert unavailable["mape"] is None
    assert c_formal_utility(unavailable) == (None, "c_mape_unavailable")
    exact_zero = _c_metrics(0.0, 0.0)
    assert exact_zero["mape"] == pytest.approx(0.0)
    value, reason = c_formal_utility(exact_zero)
    assert reason is None
    assert value == pytest.approx(1.0)


def test_e_relative_tolerance_matches_frozen_engine() -> None:
    within = _e_metrics(100.1, 100.0)
    assert within["relative_error"] == pytest.approx(0.001)
    assert within["exact_match_0_2pct"] is True
    assert e_formal_utility(within) == (1.0, None)
    outside = _e_metrics(100.3, 100.0)
    assert outside["exact_match_0_2pct"] is False
    assert e_formal_utility(outside) == (-1.0, None)


def test_e_relative_tolerance_boundary_is_strict() -> None:
    boundary = _e_metrics(501.0, 500.0)
    assert boundary["relative_error"] == pytest.approx(0.002)
    assert boundary["exact_match_0_2pct"] is False  # strict < 0.002
    assert e_formal_utility(boundary) == (-1.0, None)


def test_e_zero_answer_absolute_tolerance() -> None:
    within = _e_metrics(1e-10, 0.0)
    assert within["exact_match_0_2pct"] is True  # abs error < 1e-9
    assert e_formal_utility(within) == (1.0, None)
    outside = _e_metrics(1e-8, 0.0)
    assert outside["exact_match_0_2pct"] is False
    assert e_formal_utility(outside) == (-1.0, None)


def test_e_missing_exact_match_is_unavailable() -> None:
    metrics = _e_metrics("not-a-number", 100.0)
    assert metrics["exact_match_0_2pct"] is None
    assert e_formal_utility(metrics) == (None, "e_exact_match_unavailable")


def test_operational_utility_failure_charges_minus_one() -> None:
    assert operational_utility(None, "timeout") == -1.0
    assert operational_utility(None, "invalid_output") == -1.0
    assert operational_utility(None, "model_failure") == -1.0


def test_operational_utility_raw_wins_and_non_failure_stays_none() -> None:
    assert operational_utility(0.42, "timeout") == pytest.approx(0.42)
    assert operational_utility(None, None) is None
    with pytest.raises(AnalysisError, match="finite"):
        operational_utility(math.nan, None)


@pytest.mark.parametrize(
    "utility",
    [
        a1_formal_utility,
        a2_formal_utility,
        b_formal_utility,
        b_probability,
        c_formal_utility,
        e_formal_utility,
    ],
)
def test_formal_utilities_reject_non_mapping_input(utility: Any) -> None:
    with pytest.raises(AnalysisError, match="must be a mapping"):
        utility(["not", "a", "mapping"])


def _expected_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def test_cluster_bootstrap_single_cluster_golden() -> None:
    result = cluster_bootstrap({"only": 0.5}, resamples=50, seed=7)
    assert result == {
        "estimate": 0.5,
        "ci_lower": 0.5,
        "ci_upper": 0.5,
        "cluster_count": 1,
        "resamples": 50,
        "seed": 7,
        "confidence": 0.95,
    }


def test_cluster_bootstrap_estimate_is_mean_of_cluster_means() -> None:
    result = cluster_bootstrap({"alpha": 1.0, "beta": 2.0, "gamma": 4.0}, resamples=10, seed=1)
    assert result["estimate"] == pytest.approx(7.0 / 3.0)
    assert result["cluster_count"] == 3
    repeated = cluster_bootstrap([("x", 1.0), ("x", 3.0), ("y", 2.0)], resamples=10, seed=1)
    assert repeated["estimate"] == pytest.approx(2.0)
    assert repeated["cluster_count"] == 2
    assert repeated["ci_lower"] == pytest.approx(2.0)
    assert repeated["ci_upper"] == pytest.approx(2.0)


def test_cluster_bootstrap_determinism_same_seed() -> None:
    values = {"a": 0.2, "b": -0.4, "c": 0.9, "d": 0.1, "e": -0.7}
    first = cluster_bootstrap(values, resamples=500, seed=123)
    second = cluster_bootstrap(values, resamples=500, seed=123)
    assert first == second


def test_cluster_bootstrap_reproduces_independent_draw_sequence() -> None:
    values = {"alpha": 1.0, "beta": 2.0, "gamma": 4.0}
    result = cluster_bootstrap(values, resamples=200, seed=42, confidence=0.90)
    rng = random.Random(42)
    clusters = ["alpha", "beta", "gamma"]
    draws = [
        mean(values[clusters[rng.randrange(len(clusters))]] for _ in clusters) for _ in range(200)
    ]
    assert result["ci_lower"] == _expected_quantile(draws, 0.05)
    assert result["ci_upper"] == _expected_quantile(draws, 0.95)


def test_cluster_bootstrap_string_seed_derivation_golden() -> None:
    derived = int(hashlib.sha256(b"formal-utility/v1").hexdigest()[:16], 16)
    values = {"a": 0.2, "b": -0.4, "c": 0.9}
    from_string = cluster_bootstrap(values, resamples=100, seed="formal-utility/v1")
    assert from_string["seed"] == derived
    assert from_string == cluster_bootstrap(values, resamples=100, seed=derived)
    assert from_string == cluster_bootstrap(values, resamples=100, seed="formal-utility/v1")


def test_cluster_bootstrap_estimates_alias() -> None:
    assert cluster_bootstrap_estimates is cluster_bootstrap


def test_paired_cluster_bootstrap_golden() -> None:
    result = paired_cluster_bootstrap(
        [10.0, 20.0, 30.0],
        [1.0, 2.0, 3.0],
        ["a", "b", "c"],
        resamples=100,
        seed=5,
    )
    assert result["estimate"] == pytest.approx(18.0)
    assert result["cluster_count"] == 3
    assert result["pair_count"] == 3


def test_paired_cluster_bootstrap_shared_draws_match_difference_bootstrap() -> None:
    values_a = [10.0, 20.0, 30.0, 5.0]
    values_b = [1.0, 2.0, 3.0, 4.0]
    keys = ["a", "b", "c", "d"]
    paired = paired_cluster_bootstrap(values_a, values_b, keys, resamples=300, seed=11)
    differences = [(key, a - b) for key, a, b in zip(keys, values_a, values_b, strict=True)]
    direct = cluster_bootstrap(differences, resamples=300, seed=11)
    assert paired["estimate"] == direct["estimate"]
    assert paired["ci_lower"] == direct["ci_lower"]
    assert paired["ci_upper"] == direct["ci_upper"]


def test_paired_cluster_bootstrap_single_cluster_golden() -> None:
    result = paired_cluster_bootstrap([4.0], [1.0], ["k"], resamples=25, seed=9)
    assert result["estimate"] == pytest.approx(3.0)
    assert result["ci_lower"] == pytest.approx(3.0)
    assert result["ci_upper"] == pytest.approx(3.0)
    assert result["pair_count"] == 1


def test_paired_cluster_bootstrap_determinism() -> None:
    args = ([0.5, 1.5, -0.5], [0.1, 0.2, 0.3], ["x", "y", "z"])
    first = paired_cluster_bootstrap(*args, resamples=200, seed="e1")
    second = paired_cluster_bootstrap(*args, resamples=200, seed="e1")
    assert first == second


def test_cluster_bootstrap_validation_errors() -> None:
    with pytest.raises(AnalysisError, match="at least one cluster"):
        cluster_bootstrap({})
    with pytest.raises(AnalysisError, match="positive integer"):
        cluster_bootstrap({"x": 1.0}, resamples=0)
    with pytest.raises(AnalysisError, match="between 0 and 1"):
        cluster_bootstrap({"x": 1.0}, confidence=1.0)
    with pytest.raises(AnalysisError, match="finite number"):
        cluster_bootstrap({"x": math.nan})
    with pytest.raises(AnalysisError, match="finite number"):
        cluster_bootstrap({"x": True})
    with pytest.raises(AnalysisError, match="non-empty string"):
        cluster_bootstrap({"  ": 1.0})
    with pytest.raises(AnalysisError, match="mapping or a sequence"):
        cluster_bootstrap("not-a-mapping")
    with pytest.raises(AnalysisError, match="cluster_key, value"):
        cluster_bootstrap([("x", 1.0, 2.0)])
    with pytest.raises(AnalysisError, match="integer or a non-empty string"):
        cluster_bootstrap({"x": 1.0}, seed=1.5)
    with pytest.raises(AnalysisError, match="integer or a non-empty string"):
        cluster_bootstrap({"x": 1.0}, seed=True)


def test_paired_cluster_bootstrap_validation_errors() -> None:
    with pytest.raises(AnalysisError, match="equally sized"):
        paired_cluster_bootstrap([1.0], [1.0, 2.0], ["x"])
    with pytest.raises(AnalysisError, match="at least one paired row"):
        paired_cluster_bootstrap([], [], [])
    with pytest.raises(AnalysisError, match="finite number"):
        paired_cluster_bootstrap([math.inf], [1.0], ["x"])
    with pytest.raises(AnalysisError, match="non-empty string"):
        paired_cluster_bootstrap([1.0], [1.0], [""])


def test_make_cluster_key() -> None:
    assert make_cluster_key("event-1", "block-7") == "event-1|block-7"
    assert make_cluster_key(" solo ") == "solo"
    with pytest.raises(AnalysisError, match="at least one part"):
        make_cluster_key()
    with pytest.raises(AnalysisError, match="non-empty string"):
        make_cluster_key("ok", " ")

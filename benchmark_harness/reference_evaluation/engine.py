"""Frozen evaluation implementation from Benchmark Research commit 6c3ba94.

This is a source-level consolidation of the upstream parser modules,
evaluator modules, and ``run_benchmark.summarize_results``.  Imports were
made local-only; evaluation behavior is intentionally preserved and guarded by
the golden parity tests.
"""

from __future__ import annotations

import json
import re
from math import floor
from statistics import mean
from typing import Any, Protocol

CODE_BLOCK_RE = re.compile(r"```(?:json)?\\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict | list):
        return parsed
    return None


def _extract_with_raw_decode(text: str) -> dict[str, Any] | list[Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict | list):
            return parsed
    return None


def extract_json_from_response(response: str) -> dict[str, Any] | list[Any]:
    """Extract a JSON object or array from a raw model response."""

    parsed = _try_parse_json(response)
    if parsed is not None:
        return parsed

    code_block_match = CODE_BLOCK_RE.search(response)
    if code_block_match:
        parsed = _try_parse_json(code_block_match.group(1))
        if parsed is not None:
            return parsed

    parsed = _extract_with_raw_decode(response)
    if parsed is not None:
        return parsed

    raise ValueError("Could not extract valid JSON object or array from response")


VALID_A1_HORIZONS = {
    "1-4周",
    "1-3个月",
    "3-6个月",
    "6-12个月",
    "超过12个月",
}


def parse_a1_response(response: str) -> dict[str, Any]:
    """Parse an A1 response into normalized fields."""

    try:
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        try:
            bull = float(parsed.get("bull"))
            base = float(parsed.get("base"))
            bear = float(parsed.get("bear"))
        except (TypeError, ValueError) as exc:
            raise ValueError("bull/base/bear must be numeric") from exc

        reversion_horizon = parsed.get("reversion_horizon")
        if reversion_horizon not in VALID_A1_HORIZONS:
            raise ValueError("reversion_horizon is not in the allowed enum")

        monotonic_valid = bear <= base <= bull
        return {
            "format_valid": True,
            "bull": bull,
            "base": base,
            "bear": bear,
            "reversion_horizon": reversion_horizon,
            "monotonic_valid": monotonic_valid,
            "error": None,
        }
    except ValueError as exc:
        return {
            "format_valid": False,
            "bull": None,
            "base": None,
            "bear": None,
            "reversion_horizon": None,
            "monotonic_valid": None,
            "error": str(exc),
        }


def parse_a2_response(response: str) -> dict[str, Any]:
    """Parse an A2 response into a ranking list."""

    try:
        parsed = extract_json_from_response(response)
        ranking: Any
        if isinstance(parsed, list):
            ranking = parsed
        elif isinstance(parsed, dict):
            ranking = parsed.get("ranking")
        else:
            raise ValueError("Expected JSON array or object with ranking field")

        if not isinstance(ranking, list):
            raise ValueError("ranking must be a list")
        if not all(isinstance(item, str) for item in ranking):
            raise ValueError("ranking items must all be strings")

        return {
            "format_valid": True,
            "ranking": ranking,
            "error": None,
        }
    except ValueError as exc:
        return {
            "format_valid": False,
            "ranking": None,
            "error": str(exc),
        }


VALID_B_DIRECTIONS = {"up", "down"}


def parse_b_response(response: str) -> dict[str, Any]:
    """Parse a B-task response into direction and probability_up."""

    try:
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        direction = parsed.get("direction")
        probability_up = parsed.get("probability_up")

        if direction not in VALID_B_DIRECTIONS:
            raise ValueError("direction must be up or down")
        if probability_up is None:
            raise ValueError("probability_up is required")

        probability_float = float(probability_up)
        if not 0.0 <= probability_float <= 1.0:
            raise ValueError("probability_up must be between 0 and 1")

        return {
            "format_valid": True,
            "direction": direction,
            "probability_up": probability_float,
            "error": None,
        }
    except (ValueError, TypeError) as exc:
        return {
            "format_valid": False,
            "direction": None,
            "probability_up": None,
            "error": str(exc),
        }


def parse_c_response(response: str) -> dict[str, Any]:
    """Parse a C-task response into predicted_value."""

    try:
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        predicted_value = parsed.get("predicted_value")
        if predicted_value is None:
            raise ValueError("predicted_value is required")
        predicted_float = float(predicted_value)

        metric = parsed.get("metric")
        return {
            "format_valid": True,
            "predicted_value": predicted_float,
            "metric": metric if isinstance(metric, str) else None,
            "error": None,
        }
    except (ValueError, TypeError) as exc:
        return {
            "format_valid": False,
            "predicted_value": None,
            "metric": None,
            "error": str(exc),
        }


VALID_D_DIRECTIONS = {"positive", "negative"}


def parse_d_response(response: str) -> dict[str, Any]:
    """Parse a D-task response into a normalized structure."""

    try:
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        direction = parsed.get("direction")
        reasoning = parsed.get("reasoning")

        if direction not in VALID_D_DIRECTIONS:
            return {
                "format_valid": False,
                "direction": None,
                "reasoning": reasoning if isinstance(reasoning, str) else None,
                "error": "direction must be positive or negative",
            }

        return {
            "format_valid": True,
            "direction": direction,
            "reasoning": reasoning if isinstance(reasoning, str) else None,
            "error": None,
        }
    except ValueError as exc:
        return {
            "format_valid": False,
            "direction": None,
            "reasoning": None,
            "error": str(exc),
        }


def parse_e_response(response: str) -> dict[str, Any]:
    """Parse an E-task response into a normalized structure."""

    try:
        parsed = extract_json_from_response(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        formula_used = parsed.get("formula_used")
        answer = parsed.get("answer")
        unit = parsed.get("unit")

        try:
            answer_float = float(answer)
        except (TypeError, ValueError):
            return {
                "format_valid": False,
                "formula_used": formula_used if isinstance(formula_used, str) else None,
                "answer": None,
                "unit": unit if isinstance(unit, str) else None,
                "error": "answer must be numeric",
            }

        return {
            "format_valid": True,
            "formula_used": formula_used if isinstance(formula_used, str) else None,
            "answer": answer_float,
            "unit": unit if isinstance(unit, str) else None,
            "error": None,
        }
    except ValueError as exc:
        return {
            "format_valid": False,
            "formula_used": None,
            "answer": None,
            "unit": None,
            "error": str(exc),
        }


HORIZON_TO_DAYS: dict[str, tuple[int, int]] = {
    "1-4周": (7, 28),
    "1-3个月": (30, 90),
    "3-6个月": (90, 180),
    "6-12个月": (180, 365),
    "超过12个月": (365, 100000),
}

WINDOW_DAYS = ("30", "90", "180", "365")


def _compute_error(actual: float, base: float) -> tuple[float, str]:
    """Compute target price error under normal or zero-base handling."""

    if base == 0:
        return abs(actual - base), "absolute_error_base_zero"
    return abs(actual - base) / abs(base), "relative_error"


def _compute_monotonic_valid(parsed: dict[str, Any]) -> bool | None:
    """Compute monotonic validity from parsed A1 values."""

    bull = parsed.get("bull")
    base = parsed.get("base")
    bear = parsed.get("bear")
    if bull is None or base is None or bear is None:
        return None
    return bear <= base <= bull


def _compute_reversion_horizon_hit(
    parsed: dict[str, Any], actual_prices: dict[str, Any]
) -> bool | None:
    """Check whether observed prices enter the predicted range within the mapped horizon."""

    horizon = parsed.get("reversion_horizon")
    bear = parsed.get("bear")
    bull = parsed.get("bull")
    if horizon not in HORIZON_TO_DAYS or bear is None or bull is None:
        return None

    start_day, end_day = HORIZON_TO_DAYS[horizon]
    candidates: list[float] = []
    for day_str, price in actual_prices.items():
        if price is None:
            continue
        day = int(day_str)
        if start_day <= day <= end_day:
            candidates.append(float(price))

    if not candidates:
        return None
    return any(bear <= price <= bull for price in candidates)


def evaluate_a1(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a parsed A1 prediction against future price windows."""

    ground_truth = record.get("ground_truth") or {}
    actual_prices = ground_truth.get("actual_prices") or {}
    base = parsed.get("base")
    bear = parsed.get("bear")
    bull = parsed.get("bull")

    by_window: dict[str, Any] = {}
    for day_str in WINDOW_DAYS:
        actual = actual_prices.get(day_str)
        if actual is None or base is None or bear is None or bull is None:
            by_window[day_str] = {
                "target_price_error_abs": None,
                "target_price_error_mode": None,
                "range_hit": None,
            }
            continue

        actual_float = float(actual)
        error_value, error_mode = _compute_error(actual_float, float(base))
        by_window[day_str] = {
            "target_price_error_abs": error_value,
            "target_price_error_mode": error_mode,
            "range_hit": bear <= actual_float <= bull,
        }

    monotonic_valid = parsed.get("monotonic_valid")
    if monotonic_valid is None:
        monotonic_valid = _compute_monotonic_valid(parsed)

    return {
        "format_valid": bool(parsed.get("format_valid")),
        "monotonic_valid": monotonic_valid,
        "by_window": by_window,
        "reversion_horizon_hit": _compute_reversion_horizon_hit(parsed, actual_prices),
    }


def _ranking_metadata(
    ranking: list[str] | None, expected_codes: list[str]
) -> tuple[bool | None, bool | None, bool | None]:
    """Compute structural ranking checks."""

    if ranking is None:
        return None, None, None
    ranking_length_match = len(ranking) == len(expected_codes)
    duplicate_free = len(ranking) == len(set(ranking))
    exact_set_match = set(ranking) == set(expected_codes)
    return exact_set_match, ranking_length_match, duplicate_free


def _ranks_from_order(order: list[str]) -> dict[str, int]:
    """Convert an ordered list into 1-indexed ranks."""

    return {code: index + 1 for index, code in enumerate(order)}


def _spearman_rho(pred_order: list[str], actual_order: list[str]) -> float | None:
    """Compute Spearman rank correlation without third-party libraries."""

    if not pred_order or len(pred_order) != len(actual_order):
        return None
    pred_ranks = _ranks_from_order(pred_order)
    actual_ranks = _ranks_from_order(actual_order)
    if set(pred_ranks) != set(actual_ranks):
        return None
    n = len(pred_order)
    if n < 2:
        return None
    diff_sq = sum((pred_ranks[code] - actual_ranks[code]) ** 2 for code in pred_ranks)
    return 1 - (6 * diff_sq) / (n * (n**2 - 1))


def evaluate_a2(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an A2-F ranking prediction against actual returns."""

    stock_list = record.get("seed", {}).get("stock_list", [])
    expected_codes = [stock["code"] for stock in stock_list]
    ranking = parsed.get("ranking")

    exact_set_match, ranking_length_match, duplicate_free = _ranking_metadata(
        ranking, expected_codes
    )

    ground_truth = record.get("ground_truth") or {}
    actual_returns = ground_truth.get("actual_returns")
    actual_ranking = ground_truth.get("actual_ranking")

    n = len(expected_codes)
    k_value = floor(n / 3) if n else None

    pred_top_k: list[str] | None = None
    pred_bottom_k: list[str] | None = None
    actual_top_k: list[str] | None = None
    spearman_rho: float | None = None
    top_k_hit_rate: float | None = None
    long_short_spread: float | None = None

    structurally_valid = (
        bool(parsed.get("format_valid"))
        and ranking is not None
        and exact_set_match is True
        and ranking_length_match is True
        and duplicate_free is True
    )

    if (
        structurally_valid
        and isinstance(actual_returns, dict)
        and isinstance(actual_ranking, list)
        and k_value is not None
        and k_value >= 1
    ):
        pred_top_k = ranking[:k_value]
        pred_bottom_k = ranking[-k_value:]
        actual_top_k = actual_ranking[:k_value]

        spearman_rho = _spearman_rho(ranking, actual_ranking)
        top_k_hit_rate = len(set(pred_top_k) & set(actual_top_k)) / k_value

        long_returns = [float(actual_returns[code]) for code in pred_top_k]
        short_returns = [float(actual_returns[code]) for code in pred_bottom_k]
        long_short_spread = (sum(long_returns) / k_value) - (sum(short_returns) / k_value)

    return {
        "format_valid": bool(parsed.get("format_valid")),
        "exact_set_match": exact_set_match,
        "ranking_length_match": ranking_length_match,
        "duplicate_free": duplicate_free,
        "spearman_rho": spearman_rho,
        "top_k_hit_rate": top_k_hit_rate,
        "long_short_spread": long_short_spread,
        "k_value": k_value,
        "pred_top_k": pred_top_k,
        "pred_bottom_k": pred_bottom_k,
        "actual_top_k": actual_top_k,
    }


def evaluate_b(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a parsed B-task prediction against actual direction."""

    ground_truth = record.get("ground_truth") or {}
    actual_direction = ground_truth.get("actual_direction")
    predicted_direction = parsed.get("direction")
    probability_up = parsed.get("probability_up")
    format_valid = bool(parsed.get("format_valid"))

    directional_accuracy: bool | None = None
    brier_score: float | None = None

    if actual_direction is not None and format_valid and predicted_direction is not None:
        directional_accuracy = predicted_direction == actual_direction

    if actual_direction is not None and format_valid and probability_up is not None:
        actual_up = 1.0 if actual_direction == "up" else 0.0
        brier_score = (float(probability_up) - actual_up) ** 2

    return {
        "format_valid": format_valid,
        "directional_accuracy": directional_accuracy,
        "brier_score": brier_score,
        "variant": record.get("variant"),
        "predicted_direction": predicted_direction,
        "actual_direction": actual_direction,
        "probability_up": probability_up,
    }


def evaluate_c(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a parsed C-task prediction against future_value ground truth."""

    ground_truth = record.get("ground_truth") or {}
    future_value = ground_truth.get("future_value")
    predicted_value = parsed.get("predicted_value")
    format_valid = bool(parsed.get("format_valid"))

    mape: float | None = None
    within_10pct: bool | None = None

    if format_valid and predicted_value is not None and future_value is not None:
        denominator = abs(float(future_value))
        if denominator == 0:
            mape = 0.0 if float(predicted_value) == 0 else None
        else:
            mape = abs(float(predicted_value) - float(future_value)) / denominator
        if mape is not None:
            within_10pct = mape <= 0.10

    return {
        "format_valid": format_valid,
        "mape": mape,
        "within_10pct": within_10pct,
        "predicted_value": predicted_value,
        "future_value": future_value,
        "metric_name": record.get("seed", {}).get("metric_name"),
    }


def evaluate_d(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a parsed D-task prediction against its record."""

    ground_truth = record.get("ground_truth") or {}
    expected_direction = ground_truth.get("logic_direction")
    predicted_direction = parsed.get("direction")
    format_valid = bool(parsed.get("format_valid"))

    logic_adherence: bool | None
    if expected_direction is None:
        logic_adherence = None
    else:
        logic_adherence = format_valid and predicted_direction == expected_direction

    return {
        "format_valid": format_valid,
        "logic_adherence": logic_adherence,
        "time_band": record.get("time_band"),
        "expected_direction": expected_direction,
        "predicted_direction": predicted_direction,
    }


def _formula_text_match(parsed_formula: str | None, correct_formula: str | None) -> bool | None:
    if not parsed_formula or not correct_formula:
        return None
    left = parsed_formula.lower()
    right = correct_formula.lower()
    return left in right or right in left


def _unit_match(parsed_unit: str | None, answer_unit: str | None) -> bool | None:
    if not parsed_unit or not answer_unit:
        return None
    return parsed_unit == answer_unit or parsed_unit.lower() == answer_unit.lower()


def evaluate_e(parsed: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a parsed E-task prediction against its ground truth."""

    ground_truth = record.get("ground_truth") or {}
    answer = parsed.get("answer")
    correct_answer = ground_truth.get("correct_answer")
    format_valid = bool(parsed.get("format_valid"))

    relative_error: float | None = None
    exact_match_0_2pct: bool | None = None

    if isinstance(answer, float) and isinstance(correct_answer, int | float):
        correct_answer_float = float(correct_answer)
        abs_error = abs(answer - correct_answer_float)
        if correct_answer_float == 0.0:
            relative_error = abs_error
            exact_match_0_2pct = abs_error < 1e-9
        else:
            relative_error = abs_error / abs(correct_answer_float)
            exact_match_0_2pct = relative_error < 0.002

    return {
        "format_valid": format_valid,
        "relative_error": relative_error,
        "exact_match_0_2pct": exact_match_0_2pct,
        "formula_text_match": _formula_text_match(
            parsed.get("formula_used"),
            ground_truth.get("correct_formula"),
        ),
        "unit_match": _unit_match(parsed.get("unit"), ground_truth.get("answer_unit")),
    }


def parse_prediction(category: str, response: str) -> dict[str, Any]:
    """Dispatch to the correct parser by category."""

    if category == "A1":
        return parse_a1_response(response)
    if category == "A2":
        return parse_a2_response(response)
    if category == "B":
        return parse_b_response(response)
    if category == "C":
        return parse_c_response(response)
    if category == "D":
        return parse_d_response(response)
    if category == "E":
        return parse_e_response(response)
    raise ValueError(f"Unsupported category for parsing: {category}")


def evaluate_prediction(
    category: str, parsed: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Dispatch to the correct evaluator by category."""

    if category == "A1":
        return evaluate_a1(parsed, record)
    if category == "A2":
        return evaluate_a2(parsed, record)
    if category == "B":
        return evaluate_b(parsed, record)
    if category == "C":
        return evaluate_c(parsed, record)
    if category == "D":
        return evaluate_d(parsed, record)
    if category == "E":
        return evaluate_e(parsed, record)
    raise ValueError(f"Unsupported category for evaluation: {category}")


def _rate(values: list[bool | None]) -> float | None:
    """Compute a boolean rate while ignoring null values."""

    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(1 for value in valid if value) / len(valid)


def summarize_results(
    predictions: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    agent: str,
    model: str,
) -> dict[str, Any]:
    """Build aggregate metrics for the completed run."""

    by_category: dict[str, Any] = {}
    categories = sorted({prediction["category"] for prediction in predictions})
    api_error_count = sum(1 for prediction in predictions if prediction.get("api_error"))
    parse_error_count = sum(
        1
        for prediction in predictions
        if isinstance(prediction.get("parsed"), dict)
        and prediction["parsed"].get("format_valid") is False
    )
    latencies = [
        prediction["latency_seconds"]
        for prediction in predictions
        if isinstance(prediction.get("latency_seconds"), int | float)
    ]

    for category in categories:
        category_predictions = [
            prediction for prediction in predictions if prediction["category"] == category
        ]
        metrics_list = [prediction["metrics"] for prediction in category_predictions]
        summary: dict[str, Any] = {
            "count": len(category_predictions),
            "format_valid_rate": _rate([metrics.get("format_valid") for metrics in metrics_list]),
        }

        if category == "A1":
            summary["monotonic_valid_rate"] = _rate(
                [metrics.get("monotonic_valid") for metrics in metrics_list]
            )
            by_window: dict[str, Any] = {}
            for window in ("30", "90", "180", "365"):
                window_metrics = [
                    metrics.get("by_window", {}).get(window, {}) for metrics in metrics_list
                ]
                errors = [
                    item.get("target_price_error_abs")
                    for item in window_metrics
                    if item.get("target_price_error_abs") is not None
                ]
                by_window[window] = {
                    "mean_target_price_error_abs": mean(errors) if errors else None,
                    "range_hit_rate": _rate([item.get("range_hit") for item in window_metrics]),
                }
            summary["by_window"] = by_window
            summary["reversion_horizon_hit_rate"] = _rate(
                [metrics.get("reversion_horizon_hit") for metrics in metrics_list]
            )

        if category == "A2":
            summary["exact_set_match_rate"] = _rate(
                [metrics.get("exact_set_match") for metrics in metrics_list]
            )
            spearman_values = [
                metrics["spearman_rho"]
                for metrics in metrics_list
                if metrics.get("spearman_rho") is not None
            ]
            top_k_values = [
                metrics["top_k_hit_rate"]
                for metrics in metrics_list
                if metrics.get("top_k_hit_rate") is not None
            ]
            long_short_values = [
                metrics["long_short_spread"]
                for metrics in metrics_list
                if metrics.get("long_short_spread") is not None
            ]
            summary["mean_spearman_rho"] = mean(spearman_values) if spearman_values else None
            summary["mean_top_k_hit_rate"] = mean(top_k_values) if top_k_values else None
            summary["mean_long_short_spread"] = (
                mean(long_short_values) if long_short_values else None
            )

            by_variant: dict[str, Any] = {}
            for variant in sorted(
                {prediction.get("variant") for prediction in category_predictions}
            ):
                variant_predictions = [
                    prediction
                    for prediction in category_predictions
                    if prediction.get("variant") == variant
                ]
                variant_metrics = [prediction["metrics"] for prediction in variant_predictions]
                by_variant[str(variant)] = {
                    "count": len(variant_predictions),
                    "format_valid_rate": _rate(
                        [metrics.get("format_valid") for metrics in variant_metrics]
                    ),
                    "exact_set_match_rate": _rate(
                        [metrics.get("exact_set_match") for metrics in variant_metrics]
                    ),
                    "mean_spearman_rho": mean(
                        [
                            metrics["spearman_rho"]
                            for metrics in variant_metrics
                            if metrics.get("spearman_rho") is not None
                        ]
                    )
                    if any(metrics.get("spearman_rho") is not None for metrics in variant_metrics)
                    else None,
                    "mean_top_k_hit_rate": mean(
                        [
                            metrics["top_k_hit_rate"]
                            for metrics in variant_metrics
                            if metrics.get("top_k_hit_rate") is not None
                        ]
                    )
                    if any(metrics.get("top_k_hit_rate") is not None for metrics in variant_metrics)
                    else None,
                    "mean_long_short_spread": mean(
                        [
                            metrics["long_short_spread"]
                            for metrics in variant_metrics
                            if metrics.get("long_short_spread") is not None
                        ]
                    )
                    if any(
                        metrics.get("long_short_spread") is not None for metrics in variant_metrics
                    )
                    else None,
                }
            summary["by_variant"] = by_variant

        if category == "B":
            summary["directional_accuracy_rate"] = _rate(
                [metrics.get("directional_accuracy") for metrics in metrics_list]
            )
            brier_values = [
                metrics["brier_score"]
                for metrics in metrics_list
                if metrics.get("brier_score") is not None
            ]
            summary["mean_brier_score"] = mean(brier_values) if brier_values else None
            by_variant = {}
            for variant in sorted(
                {prediction.get("variant") for prediction in category_predictions}
            ):
                variant_predictions = [
                    prediction
                    for prediction in category_predictions
                    if prediction.get("variant") == variant
                ]
                variant_metrics = [prediction["metrics"] for prediction in variant_predictions]
                by_variant[str(variant)] = {
                    "count": len(variant_predictions),
                    "format_valid_rate": _rate(
                        [metrics.get("format_valid") for metrics in variant_metrics]
                    ),
                    "directional_accuracy_rate": _rate(
                        [metrics.get("directional_accuracy") for metrics in variant_metrics]
                    ),
                    "mean_brier_score": mean(
                        [
                            metrics["brier_score"]
                            for metrics in variant_metrics
                            if metrics.get("brier_score") is not None
                        ]
                    )
                    if any(metrics.get("brier_score") is not None for metrics in variant_metrics)
                    else None,
                }
            summary["by_variant"] = by_variant

        if category == "C":
            mape_values = [
                metrics["mape"] for metrics in metrics_list if metrics.get("mape") is not None
            ]
            summary["within_10pct_rate"] = _rate(
                [metrics.get("within_10pct") for metrics in metrics_list]
            )
            summary["mean_mape"] = mean(mape_values) if mape_values else None

        if category == "D":
            summary["logic_adherence_rate"] = _rate(
                [metrics.get("logic_adherence") for metrics in metrics_list]
            )
            by_time_band: dict[str, Any] = {}
            for time_band in sorted({metrics.get("time_band") for metrics in metrics_list}):
                band_metrics = [
                    metrics for metrics in metrics_list if metrics.get("time_band") == time_band
                ]
                by_time_band[str(time_band)] = {
                    "count": len(band_metrics),
                    "logic_adherence_rate": _rate(
                        [metrics.get("logic_adherence") for metrics in band_metrics]
                    ),
                }
            summary["by_time_band"] = by_time_band

        if category == "E":
            relative_errors = [
                metrics["relative_error"]
                for metrics in metrics_list
                if metrics.get("relative_error") is not None
            ]
            summary["exact_match_0_2pct_rate"] = _rate(
                [metrics.get("exact_match_0_2pct") for metrics in metrics_list]
            )
            summary["mean_relative_error"] = mean(relative_errors) if relative_errors else None
            summary["formula_text_match_rate"] = _rate(
                [metrics.get("formula_text_match") for metrics in metrics_list]
            )
            summary["unit_match_rate"] = _rate(
                [metrics.get("unit_match") for metrics in metrics_list]
            )

        by_category[category] = summary

    return {
        "agent": agent,
        "model": model,
        "total_records": len(predictions) + len(skipped),
        "evaluated_records": len(predictions),
        "skipped_records": len(skipped),
        "api_error_count": api_error_count,
        "parse_error_count": parse_error_count,
        "total_latency_seconds": sum(latencies) if latencies else 0.0,
        "mean_latency_seconds": mean(latencies) if latencies else None,
        "by_category": by_category,
    }


def _build_a1_mock_response(record: dict[str, Any]) -> str:
    """Construct a deterministic A1 response from ground truth prices."""

    ground_truth = record.get("ground_truth") or {}
    cutoff_price = ground_truth.get("cutoff_price")
    actual_prices = ground_truth.get("actual_prices") or {}

    base_source = cutoff_price
    if base_source in (None, 0):
        for key in ("90", "180", "30", "365"):
            if actual_prices.get(key) is not None:
                base_source = actual_prices[key]
                break
    if base_source is None:
        raise ValueError("A1 record missing usable price for mock response")

    base = float(base_source)
    response = {
        "bull": round(base * 1.10, 4),
        "base": round(base, 4),
        "bear": round(base * 0.90, 4),
        "reversion_horizon": "3-6个月",
    }
    return json.dumps(response, ensure_ascii=False)


def mock_response(record: dict[str, Any]) -> str:
    """Return the frozen source mock response for non-performance smoke runs only."""

    category = record.get("category")
    ground_truth = record.get("ground_truth") or {}

    if category == "A1":
        return _build_a1_mock_response(record)

    if category == "A2":
        actual_ranking = ground_truth.get("actual_ranking")
        if isinstance(actual_ranking, list) and all(
            isinstance(item, str) for item in actual_ranking
        ):
            return json.dumps(actual_ranking, ensure_ascii=False)
        stock_list = record.get("seed", {}).get("stock_list", [])
        fallback_ranking = [stock["code"] for stock in stock_list]
        return json.dumps(fallback_ranking, ensure_ascii=False)

    if category == "D":
        direction = ground_truth.get("logic_direction")
        if direction is None:
            raise ValueError("D record missing ground_truth.logic_direction")
        return json.dumps(
            {
                "direction": direction,
                "reasoning": "根据虚构事件的基本定价逻辑判断。",
            },
            ensure_ascii=False,
        )

    if category == "E":
        correct_answer = ground_truth.get("correct_answer")
        correct_formula = ground_truth.get("correct_formula")
        answer_unit = ground_truth.get("answer_unit")
        if correct_answer is None or correct_formula is None or answer_unit is None:
            raise ValueError("E record missing complete ground truth")
        return json.dumps(
            {
                "formula_used": correct_formula,
                "answer": correct_answer,
                "unit": answer_unit,
            },
            ensure_ascii=False,
        )

    if category == "C":
        future_value = ground_truth.get("future_value")
        if future_value is None:
            raise ValueError("C record missing ground_truth.future_value")
        return json.dumps({"predicted_value": future_value}, ensure_ascii=False)

    if category == "B":
        actual_direction = ground_truth.get("actual_direction")
        if actual_direction not in {"up", "down"}:
            raise ValueError("B record missing ground_truth.actual_direction")
        probability_up = 1.0 if actual_direction == "up" else 0.0
        return json.dumps(
            {"direction": actual_direction, "probability_up": probability_up},
            ensure_ascii=False,
        )

    raise ValueError(f"Unsupported category for mock agent: {category}")


class EvaluationEngine(Protocol):
    """The runner-facing parser, evaluator, and summarizer boundary."""

    def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]: ...

    def evaluate_prediction(
        self, category: str, parsed: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]: ...

    def summarize_results(
        self,
        predictions: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, Any]: ...

    def mock_response(self, record: dict[str, Any]) -> str: ...


class BundledEvaluationEngine:
    """Use the immutable, bundled reference implementation for v0.3 scoring."""

    def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]:
        return parse_prediction(category, raw_text)

    def evaluate_prediction(
        self, category: str, parsed: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        return evaluate_prediction(category, parsed, record)

    def summarize_results(
        self,
        predictions: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, Any]:
        return summarize_results(predictions, skipped, agent=agent, model=model)

    def mock_response(self, record: dict[str, Any]) -> str:
        return mock_response(record)

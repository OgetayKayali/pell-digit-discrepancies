#!/usr/bin/env python3
"""Run exact verification and emit a reproducible JSON report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pell_digits import (  # noqa: E402
    decimal_theoretical_return_gap,
    digit_event_indices,
    pell_pairs,
    verify_period_one_grid,
    verify_rectangle,
)


def decimal_diagnostics(max_n: int) -> Dict[str, object]:
    """Return exact membership/gap counts plus labeled numeric diagnostics."""

    event_n: List[int] = digit_event_indices(max_n, 10)

    gaps = [right - left for left, right in zip(event_n, event_n[1:])]
    gap_counts = Counter(gaps)
    total_gaps = len(gaps)
    gap_by_left = {
        left: right - left for left, right in zip(event_n, event_n[1:])
    }
    theoretical_gap_counts: Counter[int] = Counter()
    return_partition_mismatches: List[Dict[str, int]] = []
    decimal_power = 1
    next_decimal_power = 10
    for n, x, y in pell_pairs(max_n):
        while x >= next_decimal_power:
            decimal_power = next_decimal_power
            next_decimal_power *= 10
        empirical_gap = gap_by_left.get(n)
        if empirical_gap is None:
            continue
        theoretical_gap = decimal_theoretical_return_gap(
            x, y, decimal_power
        )
        theoretical_gap_counts[theoretical_gap] += 1
        if theoretical_gap != empirical_gap:
            return_partition_mismatches.append(
                {
                    "n": n,
                    "theoretical_gap": theoretical_gap,
                    "empirical_gap": empirical_gap,
                }
            )

    event_set = set(event_n)
    comparable_n = range(2, max_n - 209 + 1)
    mismatch_209 = sum(
        (n in event_set) != (n + 209 in event_set) for n in comparable_n
    )
    source_events_209 = sum(n in event_set for n in comparable_n)
    preserved_events_209 = sum(
        n in event_set and n + 209 in event_set for n in comparable_n
    )

    sqrt2 = math.sqrt(2.0)
    theta = math.log10(1.0 + sqrt2)
    lam = math.log10(sqrt2)
    delta_209 = abs(209.0 * theta - round(209.0 * theta))

    return {
        "membership_method": "exact integer digit counts",
        "event_count": len(event_n),
        "empirical_event_density": len(event_n) / max(1, max_n - 1),
        "first_oeis_indices": [n - 1 for n in event_n[:30]],
        "observed_gap_set": sorted(gap_counts),
        "gap_counts": {str(gap): gap_counts[gap] for gap in sorted(gap_counts)},
        "theoretical_gap_counts": {
            str(gap): theoretical_gap_counts[gap]
            for gap in sorted(theoretical_gap_counts)
        },
        "return_partition_mismatch_count": len(
            return_partition_mismatches
        ),
        "return_partition_mismatches": return_partition_mismatches,
        "finite_horizon_endpoint_note": (
            "The final observed event is excluded from both gap counts "
            "because its successor may exceed decimal_max_n."
        ),
        "gap_frequencies": {
            str(gap): gap_counts[gap] / total_gaps
            for gap in sorted(gap_counts)
        }
        if total_gaps
        else {},
        "shift_209_empirical": {
            "comparable_indices": max(0, max_n - 210),
            "symmetric_mismatches": mismatch_209,
            "symmetric_mismatch_frequency": (
                mismatch_209 / max(1, max_n - 210)
            ),
            "source_events": source_events_209,
            "preserved_source_events": preserved_events_209,
            "conditional_preservation": (
                preserved_events_209 / source_events_209
                if source_events_209
                else None
            ),
        },
        "numeric_diagnostics_only": {
            "theta": theta,
            "lambda": lam,
            "expected_gap_frequencies": {
                "3": (lam - 3.0 * theta + 1.0) / lam,
                "5": (lam - 2.0 + 5.0 * theta) / lam,
                "8": (1.0 - 2.0 * theta - lam) / lam,
            },
            "delta_209": delta_209,
            "expected_symmetric_mismatch_frequency_209": 2.0 * delta_209,
            "conditional_preservation_209": 1.0 - delta_209 / lam,
        },
    }


def period_one_diagnostics(max_n: int = 150) -> Dict[str, object]:
    """Return an exact structured sweep for ``[a0; overline(a)]``."""

    parameters = [
        (a0, a) for a0 in range(1, 13) for a in range(1, 7)
    ]
    result = verify_period_one_grid(
        parameters=parameters,
        bases=list(range(2, 21)),
        max_n=max_n,
    )
    report = result.to_dict()
    report["arithmetic_method"] = (
        "exact integer comparisons in Q(sqrt(a^2+4))"
    )
    report["exclusion_note"] = (
        "Excluded cases are precisely labeled small indices: a dominant "
        "term is below 1, a conjugate correction is not yet below 1, or "
        "p_n or q_n is below the base (the theorem uses exponents j>=1)."
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-base", type=int, default=100)
    parser.add_argument("--max-n", type=int, default=2000)
    parser.add_argument(
        "--decimal-max-n",
        type=int,
        default=100_000,
        help="longer exact run used only for decimal gap/shift diagnostics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "pell_verification.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verification = verify_rectangle(args.max_base, args.max_n)
    report = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "bases": [2, args.max_base],
            "indices_n": [2, args.max_n],
            "oeis_indices_k": [1, args.max_n - 1],
            "decimal_indices_n": [2, args.decimal_max_n],
        },
        "verification": verification.to_dict(),
        "decimal": decimal_diagnostics(args.decimal_max_n),
        "period_one": period_one_diagnostics(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"checked {verification.checked_events:,} exact events; "
        f"mismatches={verification.mismatch_count}; "
        f"boundaries={verification.boundary_count}; "
        f"even-base boundaries={verification.even_base_boundary_count}"
    )
    print(f"wrote {args.output}")
    return 1 if verification.mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

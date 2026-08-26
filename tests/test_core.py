"""Unit tests for the exact Pell digit-discrepancy implementation."""

from __future__ import annotations

import sys
import random
import unittest
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pell_digits import (  # noqa: E402
    actual_digit_event,
    base_digits,
    boundary_correction,
    decimal_theoretical_return_gap,
    digit_event_indices,
    ideal_rotation_event,
    is_positive_base_power,
    pell_pair,
    pell_pairs,
    period_one_convergents,
    period_one_corrections_below_one,
    period_one_terms,
    period_one_xi_at_least_base,
    predicted_digit_event,
    sign_a_plus_b_sqrt2,
    sign_a_plus_b_sqrt_d,
    verify_period_one_grid,
    verify_rectangle,
)


class PellPairTests(unittest.TestCase):
    def test_initial_pairs(self) -> None:
        expected = [
            (0, 1, 0),
            (1, 1, 1),
            (2, 3, 2),
            (3, 7, 5),
            (4, 17, 12),
            (5, 41, 29),
            (6, 99, 70),
            (7, 239, 169),
        ]
        self.assertEqual(list(pell_pairs(7)), expected)

    def test_fast_pair_matches_iteration(self) -> None:
        for n, x, y in pell_pairs(100):
            self.assertEqual(pell_pair(n), (x, y))

    def test_norm_and_parity(self) -> None:
        for n, x, y in pell_pairs(200):
            self.assertEqual(x * x - 2 * y * y, (-1) ** n)
            self.assertEqual(x % 2, 1)
            self.assertEqual(y % 2, n % 2)


class ExactComparisonTests(unittest.TestCase):
    def test_quadratic_sign(self) -> None:
        self.assertEqual(sign_a_plus_b_sqrt2(3, -2), 1)
        self.assertEqual(sign_a_plus_b_sqrt2(7, -5), -1)
        self.assertEqual(sign_a_plus_b_sqrt2(-3, 2), -1)
        self.assertEqual(sign_a_plus_b_sqrt2(-7, 5), 1)
        self.assertEqual(sign_a_plus_b_sqrt2(0, 4), 1)
        self.assertEqual(sign_a_plus_b_sqrt2(-4, 0), -1)

    def test_base_digits(self) -> None:
        self.assertEqual(base_digits(99, 10), 2)
        self.assertEqual(base_digits(100, 10), 3)
        self.assertEqual(base_digits(7, 2), 3)

    def test_positive_base_power(self) -> None:
        self.assertEqual(is_positive_base_power(169, 13), 2)
        self.assertEqual(is_positive_base_power(81, 3), 4)
        self.assertIsNone(is_positive_base_power(99, 3))
        self.assertIsNone(is_positive_base_power(1, 10))


class BoundaryTheoremTests(unittest.TestCase):
    def test_known_plus_boundaries(self) -> None:
        for base, n, exponent in [(3, 2, 1), (17, 4, 1), (99, 6, 1)]:
            x, y = pell_pair(n)
            self.assertEqual(boundary_correction(n, x, y, base), ("plus", exponent))
            self.assertFalse(ideal_rotation_event(x, y, base))
            self.assertTrue(actual_digit_event(x, y, base))
            self.assertTrue(predicted_digit_event(n, x, y, base))

    def test_known_minus_boundaries(self) -> None:
        for base, n, exponent in [
            (5, 3, 1),
            (29, 5, 1),
            (13, 7, 2),
            (169, 7, 1),
        ]:
            x, y = pell_pair(n)
            self.assertEqual(boundary_correction(n, x, y, base), ("minus", exponent))
            self.assertTrue(ideal_rotation_event(x, y, base))
            self.assertFalse(actual_digit_event(x, y, base))
            self.assertFalse(predicted_digit_event(n, x, y, base))

    def test_small_rectangle(self) -> None:
        result = verify_rectangle(max_base=40, max_n=250)
        self.assertEqual(result.mismatch_count, 0)
        self.assertEqual(result.even_base_boundary_count, 0)

    def test_all_six_corrections_through_base_100(self) -> None:
        result = verify_rectangle(max_base=100, max_n=2000)
        observed = [
            (
                record["base"],
                record["n"],
                record["kind"],
                record["exponent"],
            )
            for record in result.boundaries
        ]
        self.assertEqual(
            observed,
            [
                (3, 2, "plus", 1),
                (5, 3, "minus", 1),
                (13, 7, "minus", 2),
                (17, 4, "plus", 1),
                (29, 5, "minus", 1),
                (99, 6, "plus", 1),
            ],
        )
        self.assertEqual(result.boundary_count, 6)
        self.assertEqual(result.mismatch_count, 0)

    def test_threshold_scan_matches_direct_events(self) -> None:
        pairs = list(pell_pairs(500))
        for base in range(2, 21):
            direct = [
                n
                for n, x, y in pairs
                if n >= 2 and actual_digit_event(x, y, base)
            ]
            self.assertEqual(digit_event_indices(500, base), direct)

    def test_published_a273980_prefix(self) -> None:
        expected_k = [
            8,
            13,
            21,
            26,
            34,
            39,
            47,
            55,
            60,
            68,
            73,
            81,
            86,
            89,
            94,
            102,
            107,
            115,
            120,
            128,
            136,
            141,
            149,
            154,
            162,
            167,
            175,
            183,
            188,
            196,
        ]
        computed_k = [n - 1 for n in digit_event_indices(197, 10)]
        self.assertEqual(computed_k, expected_k)

    def test_decimal_return_partition_matches_every_empirical_gap(self) -> None:
        max_n = 10_000
        pairs = list(pell_pairs(max_n))
        event_n = digit_event_indices(max_n, 10)
        empirical = Counter()
        theoretical = Counter()
        for left, right in zip(event_n, event_n[1:]):
            gap = right - left
            empirical[gap] += 1
            _, x, y = pairs[left]
            theoretical[decimal_theoretical_return_gap(x, y)] += 1
        self.assertEqual(empirical, theoretical)
        self.assertEqual(set(empirical), {3, 5, 8})

    def test_q209_exact_near_period_and_shift_counts(self) -> None:
        x, y = pell_pair(209)
        self.assertEqual(
            x,
            int(
                "500136142311784039763796745509803271115804028742221969"
                "29138047356118934742162401"
            ),
        )
        self.assertEqual(
            y,
            int(
                "353649657745142669936206018626914095229994662108096448"
                "88703383775853726309480049"
            ),
        )
        power = 10**80
        self.assertGreater(sign_a_plus_b_sqrt2(x - power, y), 0)
        self.assertLess(sign_a_plus_b_sqrt2(x, y - power), 0)

        event_set = set(digit_event_indices(100_000, 10))
        comparable = range(2, 100_000 - 209 + 1)
        mismatches = sum(
            (n in event_set) != (n + 209 in event_set)
            for n in comparable
        )
        source_events = sum(n in event_set for n in comparable)
        preserved = sum(
            n in event_set and n + 209 in event_set for n in comparable
        )
        self.assertEqual((mismatches, source_events, preserved), (23, 15021, 15009))


class PeriodOneTests(unittest.TestCase):
    def test_initial_convergents_and_recurrence_data(self) -> None:
        self.assertEqual(
            list(period_one_convergents(4, a0=2, a=3)),
            [
                (0, 2, 1),
                (1, 7, 3),
                (2, 23, 10),
                (3, 76, 33),
                (4, 251, 109),
            ],
        )

    def test_general_quadratic_sign(self) -> None:
        self.assertEqual(sign_a_plus_b_sqrt_d(3, -1, 5), 1)
        self.assertEqual(sign_a_plus_b_sqrt_d(2, -1, 5), -1)
        self.assertEqual(sign_a_plus_b_sqrt_d(-3, 1, 5), -1)
        self.assertEqual(sign_a_plus_b_sqrt_d(-2, 1, 5), 1)

    def test_small_correction_cutoff_is_detected_exactly(self) -> None:
        terms = list(period_one_terms(20, a0=100, a=1))
        self.assertFalse(period_one_corrections_below_one(terms[0], 100, 1))
        self.assertTrue(period_one_corrections_below_one(terms[-1], 100, 1))

    def test_exact_grid_including_ratio_at_least_base(self) -> None:
        parameters = [
            (1, 1),
            (2, 1),
            (10, 1),
            (1, 2),
            (3, 2),
            (12, 6),
        ]
        result = verify_period_one_grid(
            parameters=parameters,
            bases=list(range(2, 21)),
            max_n=150,
        )
        self.assertEqual(result.mismatch_count, 0)
        self.assertGreater(result.checked_cases, 10_000)
        self.assertGreater(result.xi_at_least_base_cases, 0)

        self.assertTrue(period_one_xi_at_least_base(10, 1, 2))
        values = {
            base_digits(p, 2) - base_digits(q, 2)
            for n, p, q in period_one_convergents(150, 10, 1)
            if n >= 20
        }
        self.assertEqual(values, {3, 4})

    def test_deterministic_random_composite_bases(self) -> None:
        rng = random.Random(20260724)
        composite_bases = [
            value
            for value in range(4, 101)
            if any(value % divisor == 0 for divisor in range(2, value))
        ]
        bases = rng.sample(composite_bases, 20)
        parameters = [
            (rng.randint(1, 20), rng.randint(1, 8)) for _ in range(20)
        ]
        result = verify_period_one_grid(parameters, bases, max_n=100)
        self.assertEqual(result.mismatch_count, 0)
        self.assertGreater(result.checked_cases, 30_000)


if __name__ == "__main__":
    unittest.main()

"""Exact arithmetic for Pell digit-length discrepancy experiments.

The authoritative event tests in this module use integers only.  In
particular, ``ideal_rotation_event`` does not evaluate a logarithm: it
compares the quadratic irrational

    s_n = (X_n + Y_n * sqrt(2)) / 2

with consecutive powers of the requested base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Tuple


def _validate_base(base: int) -> None:
    if not isinstance(base, int) or isinstance(base, bool) or base < 2:
        raise ValueError("base must be an integer at least 2")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def pell_pairs(max_n: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(n, X_n, Y_n)`` for ``0 <= n <= max_n``.

    Multiplication by ``1 + sqrt(2)`` gives the exact update

    ``X_{n+1} = X_n + 2 Y_n`` and ``Y_{n+1} = X_n + Y_n``.
    """

    if not isinstance(max_n, int) or isinstance(max_n, bool) or max_n < 0:
        raise ValueError("max_n must be a nonnegative integer")

    x, y = 1, 0
    yield 0, x, y
    for n in range(1, max_n + 1):
        x, y = x + 2 * y, x + y
        yield n, x, y


def pell_pair(n: int) -> Tuple[int, int]:
    """Return ``(X_n, Y_n)`` using binary exponentiation."""

    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("n must be a nonnegative integer")

    # Accumulator and current power in Z[sqrt(2)].
    ax, ay = 1, 0
    px, py = 1, 1
    exponent = n
    while exponent:
        if exponent & 1:
            ax, ay = ax * px + 2 * ay * py, ax * py + ay * px
        px, py = px * px + 2 * py * py, 2 * px * py
        exponent >>= 1
    return ax, ay


def floor_log_base(value: int, base: int) -> int:
    """Return the unique ``m`` satisfying ``base**m <= value < base**(m+1)``."""

    _validate_positive_integer(value, "value")
    _validate_base(base)

    exponent = 0
    quotient = value
    while quotient >= base:
        quotient //= base
        exponent += 1
    return exponent


def base_digits(value: int, base: int) -> int:
    """Return the number of base-``base`` digits of ``value``."""

    return floor_log_base(value, base) + 1


def is_positive_base_power(value: int, base: int) -> Optional[int]:
    """Return ``m >= 1`` when ``value == base**m``; otherwise return ``None``."""

    _validate_positive_integer(value, "value")
    _validate_base(base)

    exponent = 0
    quotient = value
    while quotient % base == 0:
        quotient //= base
        exponent += 1
    if quotient == 1 and exponent >= 1:
        return exponent
    return None


def sign_a_plus_b_sqrt2(a: int, b: int) -> int:
    """Return the exact sign of ``a + b*sqrt(2)``.

    Squaring is used only when the rational and irrational summands have
    opposite signs.  Equality cannot occur for nonzero integer ``a`` and
    ``b``, but the equality branches are retained defensively.
    """

    if a == 0:
        return (b > 0) - (b < 0)
    if b == 0:
        return (a > 0) - (a < 0)
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1

    rational_square = a * a
    irrational_square = 2 * b * b
    if rational_square == irrational_square:
        return 0
    if a > 0:  # a - |b| sqrt(2)
        return 1 if rational_square > irrational_square else -1
    # b sqrt(2) - |a|
    return 1 if irrational_square > rational_square else -1


def _compare_s_to_power(x: int, y: int, power: int) -> int:
    """Compare ``s=(x+y*sqrt(2))/2`` with an integer ``power``."""

    return sign_a_plus_b_sqrt2(x - 2 * power, y)


def _compare_s_to_sqrt2_power(x: int, y: int, power: int) -> int:
    """Compare ``s=(x+y*sqrt(2))/2`` with ``sqrt(2)*power``."""

    return sign_a_plus_b_sqrt2(x, y - 2 * power)


def _floor_power_for_s(x: int, y: int, base: int) -> Tuple[int, int]:
    """Return ``(m, base**m)`` with ``base**m < s < base**(m+1)``.

    The caller supplies a Pell pair with index at least two.  Starting from
    the integer ``x`` is safe because ``|x-s| < 1/2``; exact comparisons
    perform the final adjustment.
    """

    exponent = floor_log_base(x, base)
    power = base**exponent

    while _compare_s_to_power(x, y, power) < 0:
        if exponent == 0:
            raise AssertionError("s_n should exceed 1 for n >= 2")
        exponent -= 1
        power //= base

    while _compare_s_to_power(x, y, power * base) > 0:
        exponent += 1
        power *= base

    return exponent, power


def actual_digit_event(x: int, y: int, base: int) -> bool:
    """Return whether ``X`` has more base-``base`` digits than ``Y``."""

    return base_digits(x, base) > base_digits(y, base)


def digit_event_indices(max_n: int, base: int) -> List[int]:
    """Return all event indices ``n`` through ``max_n`` exactly.

    Digit thresholds are advanced monotonically.  This avoids recomputing a
    base logarithm from scratch for every increasingly large Pell term.
    """

    if max_n < 2:
        return []
    _validate_base(base)

    x_digits = 1
    y_digits = 1
    next_x_power = base
    next_y_power = base
    events: List[int] = []

    for n, x, y in pell_pairs(max_n):
        if n < 1:
            continue
        while x >= next_x_power:
            x_digits += 1
            next_x_power *= base
        while y >= next_y_power:
            y_digits += 1
            next_y_power *= base
        if n >= 2 and x_digits > y_digits:
            events.append(n)
    return events


def ideal_rotation_event(x: int, y: int, base: int) -> bool:
    """Test ideal rotation membership with exact quadratic comparisons.

    This is equivalent to asking whether the base-``base`` significand of
    ``alpha**n / 2`` lies in ``[1, sqrt(2))``.
    """

    _validate_base(base)
    _, power = _floor_power_for_s(x, y, base)
    return _compare_s_to_sqrt2_power(x, y, power) < 0


def boundary_correction(
    n: int, x: int, y: int, base: int
) -> Tuple[Optional[str], Optional[int]]:
    """Return the applicable boundary correction and its exponent.

    ``("plus", m)`` means even ``n`` and ``X_n = base**m``.
    ``("minus", m)`` means odd ``n`` and ``Y_n = base**m``.
    """

    if n < 2:
        raise ValueError("the event set starts at n = 2")
    if n % 2 == 0:
        exponent = is_positive_base_power(x, base)
        return ("plus", exponent) if exponent is not None else (None, None)
    exponent = is_positive_base_power(y, base)
    return ("minus", exponent) if exponent is not None else (None, None)


def predicted_digit_event(n: int, x: int, y: int, base: int) -> bool:
    """Return the event predicted by the exact boundary theorem."""

    ideal = ideal_rotation_event(x, y, base)
    correction, _ = boundary_correction(n, x, y, base)
    return (ideal or correction == "plus") and correction != "minus"


def decimal_theoretical_return_gap(
    x: int, y: int, power: Optional[int] = None
) -> int:
    """Return the exact theoretical next gap ``3``, ``5``, or ``8``.

    The supplied Pell pair must be a decimal event.  If ``B`` is the power
    of ten immediately below ``alpha**n/2``, the three return subintervals
    are tested without logarithms:

    ``gap 3`` iff ``alpha**(n+3) < 20*sqrt(2)*B``;
    ``gap 5`` iff ``alpha**(n+5) >= 200*B``;
    otherwise the gap is ``8``.
    """

    if power is None:
        _, power = _floor_power_for_s(x, y, 10)
    else:
        _validate_positive_integer(power, "power")
        if (
            _compare_s_to_power(x, y, power) <= 0
            or _compare_s_to_power(x, y, 10 * power) >= 0
        ):
            raise ValueError("power does not bracket alpha**n/2")
    if _compare_s_to_sqrt2_power(x, y, power) >= 0:
        raise ValueError("the supplied Pell pair is not a decimal event")

    # Multiply x+y*sqrt(2) by alpha**3 = 7+5*sqrt(2).
    x3, y3 = 7 * x + 10 * y, 5 * x + 7 * y
    if sign_a_plus_b_sqrt2(x3, y3 - 20 * power) < 0:
        return 3

    # Multiply by alpha**5 = 41+29*sqrt(2).
    x5, y5 = 41 * x + 58 * y, 29 * x + 41 * y
    if sign_a_plus_b_sqrt2(x5 - 200 * power, y5) >= 0:
        return 5
    return 8


@dataclass(frozen=True)
class VerificationResult:
    """Serializable result of an exact base/index verification sweep."""

    max_base: int
    max_n: int
    checked_events: int
    mismatch_count: int
    mismatches: List[Dict[str, object]]
    boundary_count: int
    boundaries: List[Dict[str, object]]
    even_base_boundary_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def verify_rectangle(max_base: int, max_n: int) -> VerificationResult:
    """Verify the boundary theorem for ``2 <= b <= max_base`` and
    ``2 <= n <= max_n``.

    All event decisions are exact.  The returned boundary list is a
    machine-readable record of every correction found in the rectangle.
    """

    if max_base < 2:
        raise ValueError("max_base must be at least 2")
    if max_n < 2:
        raise ValueError("max_n must be at least 2")

    pairs = list(pell_pairs(max_n))
    mismatches: List[Dict[str, object]] = []
    boundaries: List[Dict[str, object]] = []
    even_base_boundary_count = 0

    for base in range(2, max_base + 1):
        for n in range(2, max_n + 1):
            _, x, y = pairs[n]
            actual = actual_digit_event(x, y, base)
            ideal = ideal_rotation_event(x, y, base)
            correction, exponent = boundary_correction(n, x, y, base)
            predicted = (ideal or correction == "plus") and correction != "minus"

            if correction is not None:
                record: Dict[str, object] = {
                    "base": base,
                    "n": n,
                    "oeis_index": n - 1,
                    "kind": correction,
                    "exponent": exponent,
                    "value": str(x if correction == "plus" else y),
                    "ideal_event": ideal,
                    "actual_event": actual,
                }
                boundaries.append(record)
                if base % 2 == 0:
                    even_base_boundary_count += 1

            if actual != predicted:
                mismatches.append(
                    {
                        "base": base,
                        "n": n,
                        "x": str(x),
                        "y": str(y),
                        "actual_event": actual,
                        "ideal_event": ideal,
                        "correction": correction,
                        "predicted_event": predicted,
                    }
                )

    return VerificationResult(
        max_base=max_base,
        max_n=max_n,
        checked_events=(max_base - 1) * (max_n - 1),
        mismatch_count=len(mismatches),
        mismatches=mismatches,
        boundary_count=len(boundaries),
        boundaries=boundaries,
        even_base_boundary_count=even_base_boundary_count,
    )

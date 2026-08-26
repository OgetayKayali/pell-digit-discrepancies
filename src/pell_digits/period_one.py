"""Exact arithmetic for the period-one family ``[a0; overline(a)]``.

The dominant terms live in ``Q(sqrt(a**2 + 4))``.  Every comparison with
an integer base-power threshold is reduced to the sign of
``r + s*sqrt(a**2 + 4)`` and is therefore decided with integer arithmetic.
No logarithm or floating-point approximation is authoritative here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .core import base_digits, floor_log_base, is_positive_base_power


def _validate_parameter(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def sign_a_plus_b_sqrt_d(rational: int, radical: int, d: int) -> int:
    """Return the exact sign of ``rational + radical*sqrt(d)``.

    ``d`` need only be positive.  In this module it is ``a**2 + 4``, which
    is nonsquare for every positive integer ``a``.
    """

    _validate_parameter(d, "d")
    if rational == 0:
        return (radical > 0) - (radical < 0)
    if radical == 0:
        return (rational > 0) - (rational < 0)
    if rational > 0 and radical > 0:
        return 1
    if rational < 0 and radical < 0:
        return -1

    rational_square = rational * rational
    radical_square = d * radical * radical
    if rational_square == radical_square:
        return 0
    if rational > 0:
        return 1 if rational_square > radical_square else -1
    return 1 if radical_square > rational_square else -1


@dataclass(frozen=True)
class PeriodOneTerm:
    """One convergent and the companion data needed for exact comparisons."""

    n: int
    p: int
    q: int
    u_n: int
    u_next: int
    v_n: int
    v_next: int


def period_one_terms(
    max_n: int, a0: int, a: int
) -> Iterator[PeriodOneTerm]:
    """Yield exact convergent data for ``[a0; overline(a)]``.

    Here ``U_0=0, U_1=1`` and ``V_0=2, V_1=a`` satisfy the same recurrence
    ``G_{n+1}=a*G_n+G_{n-1}``.  The convergent is
    ``q_n=U_{n+1}``, ``p_n=a0*U_{n+1}+U_n``.
    """

    if not isinstance(max_n, int) or isinstance(max_n, bool) or max_n < 0:
        raise ValueError("max_n must be a nonnegative integer")
    _validate_parameter(a0, "a0")
    _validate_parameter(a, "a")

    u_n, u_next = 0, 1
    v_n, v_next = 2, a
    for n in range(max_n + 1):
        yield PeriodOneTerm(
            n=n,
            p=a0 * u_next + u_n,
            q=u_next,
            u_n=u_n,
            u_next=u_next,
            v_n=v_n,
            v_next=v_next,
        )
        u_n, u_next = u_next, a * u_next + u_n
        v_n, v_next = v_next, a * v_next + v_n


def period_one_convergents(
    max_n: int, a0: int, a: int
) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(n, p_n, q_n)`` for ``[a0; overline(a)]``."""

    for term in period_one_terms(max_n, a0, a):
        yield term.n, term.p, term.q


def _compare_denominator_dominant_to_integer(
    term: PeriodOneTerm, a: int, integer: int
) -> int:
    """Compare ``rho**(n+1)/sqrt(D)`` with an integer exactly."""

    d = a * a + 4
    # Multiplication by 2*sqrt(D)>0 leaves the sign unchanged.
    return sign_a_plus_b_sqrt_d(
        term.v_next, term.u_next - 2 * integer, d
    )


def _compare_numerator_dominant_to_integer(
    term: PeriodOneTerm, a0: int, a: int, integer: int
) -> int:
    """Compare ``(a0*rho**(n+1)+rho**n)/sqrt(D)`` with an integer."""

    d = a * a + 4
    return sign_a_plus_b_sqrt_d(
        a0 * term.v_next + term.v_n,
        a0 * term.u_next + term.u_n - 2 * integer,
        d,
    )


def _dominant_floor_power(
    term: PeriodOneTerm, a0: int, a: int, base: int, numerator: bool
) -> Tuple[int, int]:
    """Return ``(j, base**j)`` immediately below a dominant term.

    This routine is used only once the dominant term is at least one.
    """

    actual = term.p if numerator else term.q
    compare = (
        lambda integer: _compare_numerator_dominant_to_integer(
            term, a0, a, integer
        )
        if numerator
        else _compare_denominator_dominant_to_integer(term, a, integer)
    )
    if compare(1) < 0:
        raise ValueError("dominant term is below 1 at this index")

    exponent = floor_log_base(actual, base)
    power = base**exponent
    while compare(power) < 0:
        if exponent == 0:
            raise AssertionError("dominant term should be at least 1")
        exponent -= 1
        power //= base
    while compare(power * base) >= 0:
        exponent += 1
        power *= base
    return exponent, power


def period_one_ideal_discrepancy(
    term: PeriodOneTerm, a0: int, a: int, base: int
) -> int:
    """Return the dominant-term digit discrepancy exactly."""

    numerator_exponent, _ = _dominant_floor_power(
        term, a0, a, base, numerator=True
    )
    denominator_exponent, _ = _dominant_floor_power(
        term, a0, a, base, numerator=False
    )
    return numerator_exponent - denominator_exponent


def period_one_corrections_below_one(
    term: PeriodOneTerm, a0: int, a: int
) -> bool:
    """Return whether both conjugate corrections have absolute value < 1."""

    denominator_low = _compare_denominator_dominant_to_integer(
        term, a, term.q - 1
    )
    denominator_high = _compare_denominator_dominant_to_integer(
        term, a, term.q + 1
    )
    numerator_low = _compare_numerator_dominant_to_integer(
        term, a0, a, term.p - 1
    )
    numerator_high = _compare_numerator_dominant_to_integer(
        term, a0, a, term.p + 1
    )
    return (
        denominator_low > 0
        and denominator_high < 0
        and numerator_low > 0
        and numerator_high < 0
    )


def _positive_boundary(
    term: PeriodOneTerm, a0: int, a: int, base: int, numerator: bool
) -> Optional[int]:
    """Return the positive base-power correction exponent, if present."""

    actual = term.p if numerator else term.q
    exponent = is_positive_base_power(actual, base)
    if exponent is None:
        return None
    dominant_comparison = (
        _compare_numerator_dominant_to_integer(term, a0, a, actual)
        if numerator
        else _compare_denominator_dominant_to_integer(term, a, actual)
    )
    return exponent if dominant_comparison < 0 else None


def period_one_predicted_discrepancy(
    term: PeriodOneTerm, a0: int, a: int, base: int
) -> Tuple[int, Optional[int], Optional[int]]:
    """Return theorem prediction and numerator/denominator boundary exponents.

    The caller must first establish that both corrections have magnitude
    below one and that the two dominant terms are at least one.
    """

    ideal = period_one_ideal_discrepancy(term, a0, a, base)
    numerator_boundary = _positive_boundary(
        term, a0, a, base, numerator=True
    )
    denominator_boundary = _positive_boundary(
        term, a0, a, base, numerator=False
    )
    predicted = (
        ideal
        + (numerator_boundary is not None)
        - (denominator_boundary is not None)
    )
    return predicted, numerator_boundary, denominator_boundary


def period_one_xi_at_least_base(a0: int, a: int, base: int) -> bool:
    """Return whether ``[a0; overline(a)] >= base`` exactly."""

    _validate_parameter(a0, "a0")
    _validate_parameter(a, "a")
    if not isinstance(base, int) or isinstance(base, bool) or base < 2:
        raise ValueError("base must be an integer at least 2")
    # xi=(2*a0-a+sqrt(a**2+4))/2.
    return (
        sign_a_plus_b_sqrt_d(
            2 * a0 - a - 2 * base, 1, a * a + 4
        )
        >= 0
    )


@dataclass(frozen=True)
class PeriodOneVerificationResult:
    """Serializable exact verification result for a parameter grid."""

    parameters: List[Tuple[int, int]]
    bases: List[int]
    max_n: int
    checked_cases: int
    excluded_small_cases: int
    xi_at_least_base_cases: int
    mismatch_count: int
    mismatches: List[Dict[str, object]]
    numerator_boundary_count: int
    denominator_boundary_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def verify_period_one_grid(
    parameters: Sequence[Tuple[int, int]],
    bases: Sequence[int],
    max_n: int,
) -> PeriodOneVerificationResult:
    """Verify the eventual boundary formula on a finite exact grid.

    Small cases are excluded precisely when a dominant term is below one,
    a conjugate correction is not yet below one, or one of the actual
    terms is smaller than the base.  The last condition keeps the theorem's
    boundary convention ``base**j`` with ``j >= 1``.
    """

    if not isinstance(max_n, int) or isinstance(max_n, bool) or max_n < 0:
        raise ValueError("max_n must be a nonnegative integer")
    normalized_parameters = list(parameters)
    normalized_bases = list(bases)
    if not normalized_parameters:
        raise ValueError("parameters must not be empty")
    if not normalized_bases:
        raise ValueError("bases must not be empty")

    for a0, a in normalized_parameters:
        _validate_parameter(a0, "a0")
        _validate_parameter(a, "a")
    for base in normalized_bases:
        if not isinstance(base, int) or isinstance(base, bool) or base < 2:
            raise ValueError("every base must be an integer at least 2")

    checked_cases = 0
    excluded_small_cases = 0
    xi_at_least_base_cases = 0
    numerator_boundary_count = 0
    denominator_boundary_count = 0
    mismatches: List[Dict[str, object]] = []

    for a0, a in normalized_parameters:
        terms = list(period_one_terms(max_n, a0, a))
        for base in normalized_bases:
            large_ratio = period_one_xi_at_least_base(a0, a, base)
            for term in terms:
                if (
                    term.p < base
                    or term.q < base
                    or not period_one_corrections_below_one(term, a0, a)
                ):
                    excluded_small_cases += 1
                    continue

                predicted, numerator_boundary, denominator_boundary = (
                    period_one_predicted_discrepancy(
                        term, a0, a, base
                    )
                )
                actual = base_digits(term.p, base) - base_digits(
                    term.q, base
                )
                checked_cases += 1
                xi_at_least_base_cases += int(large_ratio)
                numerator_boundary_count += int(
                    numerator_boundary is not None
                )
                denominator_boundary_count += int(
                    denominator_boundary is not None
                )
                if actual != predicted:
                    mismatches.append(
                        {
                            "a0": a0,
                            "a": a,
                            "base": base,
                            "n": term.n,
                            "p": str(term.p),
                            "q": str(term.q),
                            "actual": actual,
                            "predicted": predicted,
                            "numerator_boundary_exponent": (
                                numerator_boundary
                            ),
                            "denominator_boundary_exponent": (
                                denominator_boundary
                            ),
                        }
                    )

    return PeriodOneVerificationResult(
        parameters=normalized_parameters,
        bases=normalized_bases,
        max_n=max_n,
        checked_cases=checked_cases,
        excluded_small_cases=excluded_small_cases,
        xi_at_least_base_cases=xi_at_least_base_cases,
        mismatch_count=len(mismatches),
        mismatches=mismatches,
        numerator_boundary_count=numerator_boundary_count,
        denominator_boundary_count=denominator_boundary_count,
    )

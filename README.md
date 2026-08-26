# Pell digit-discrepancy verification

This repository contains the exact-arithmetic source code, tests, and
machine-readable data used to verify computational claims about digit-length
discrepancies in Pell convergents and period-one quadratic continued
fractions.

It intentionally does not contain manuscript sources, submission materials,
proof-development notes, compiled documents, or internal research files.

## Contents

- `src/pell_digits/core.py` implements Pell pairs, base-power tests, exact
  comparisons in `Z[sqrt(2)]`, boundary corrections, decimal events, gaps,
  and shift statistics.
- `src/pell_digits/period_one.py` implements the corresponding exact checks
  for period-one quadratic continued fractions.
- `scripts/verify_pell.py` regenerates the verification report.
- `tests/test_core.py` exercises the exact formulas, endpoint cases, known
  sequence prefix, decimal return partition, 209-shift, and period-one cases.
- `data/pell_verification_b100_n2000.json` is the generated verification
  dataset.

The implementation uses only the Python standard library. Floating-point
arithmetic is used only for reported decimal approximations; membership,
boundary, endpoint, and quadratic-sign decisions use exact integer
arithmetic.

## Run the tests

Python 3.10 or later is required.

```console
python -m unittest discover -s tests -v
```

The test suite contains 19 tests.

## Regenerate the verification data

```console
python scripts/verify_pell.py --max-base 100 --max-n 2000 --decimal-max-n 100000 --output data/pell_verification_b100_n2000.json
```

The generated report covers 197,901 exact base-index comparisons, decimal
events through index 100,000, the 209-shift comparison, and a finite
period-one parameter sweep. The timestamp field is expected to change when
the report is regenerated; the mathematical records should remain the same.


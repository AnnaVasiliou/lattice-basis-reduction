# lattice-basis-reduction

Reduces an integer lattice basis using the LLL (Lenstra-Lenstra-Lovász) algorithm with exact rational arithmetic.

```python
from lattice_basis_reduction import lll_reduce

basis = [[1, 1, 1], [-1, 0, 2], [3, 4, 5]]
reduced = lll_reduce(basis)
print(reduced)
```

## Why this exists

LLL reduction turns a lattice basis into a nearly-orthogonal one with short vectors, which is a standard preprocessing step for integer relation finding and lattice-based cryptanalysis. This library implements it using only the Python standard library — no NumPy, no SciPy — so it runs in environments with no third-party packages installed.

The trade-off: exact integer arithmetic means the algorithm is always correct, but it is slower than floating-point implementations. The internal Gram-Schmidt uses `(numerator, denominator)` pairs rather than `Fraction` objects to avoid object allocation overhead in the inner loop, though both are mathematically equivalent.

## Edge cases

The input must be a list of integer vectors. Non-integer or ragged inputs raise `ValueError`. The `delta` parameter is given as a percentage (25–100, representing 1/4 to 1); values outside this range raise `ValueError` because the algorithm's termination guarantee requires `1/4 ≤ δ ≤ 1`.

The implementation recompute the Gram-Schmidt orthogonalisation from scratch after each basis modification. This is correct but not the most efficient approach — for bases with many vectors, expect cubic-or-worse performance rather than the theoretical optimum.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.


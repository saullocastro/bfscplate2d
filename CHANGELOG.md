# Changelog

The dates of the released versions are those of their tagged commits.

## Unreleased

### New: follower pressure

- `update_fext_follower` and `update_KCfollower`, with the module constant
  `KCfollower_SPARSE_SIZE = 1056` and the element attribute
  `init_k_KCfollower`: the force vector and the unsymmetric load stiffness
  matrix of a pressure normal to the deformed mid-surface, with the area
  vector truncated to first order (`quadratic=0`) or complete
  (`quadratic=1`), and the dead part optional (`reference`). The kernels
  are written by `deriving_equations/derive_bfscplate2d_follower_pressure.py`
  and kept in sync by `regenerate.py`. New `tests/test_follower_pressure.py`.
- The Notes of `update_KCfollower` state the conditions under which the
  load stiffness matrix of the first-order area vector is symmetric, for
  the element matrix and for the assembled matrix.

### Bug fixes

- `BFSCPlate2D.__init__` sets `init_k_KCNL`, which a second call of the
  constructor left at its previous value. New test that the constructor
  sets every `init_k_*` to zero.

### Documentation

- D1: functional docstrings of the modules, of `BFSCPlate2D` and of every
  public function.
- D2: Notes and References of the theory: Hermite interpolation of the
  BFSC element, classical laminated plate theory with the von Karman
  strains, the element matrices, the internal forces and the quadrature.
- D3: Sphinx documentation in `doc/`, with a user guide (the element,
  assembling a model, the analyses, the follower pressure) and usage
  examples `ex_*.rst` with the code of the tests.
- Solving with structsolve: the callables of `structsolve.Analysis` for an
  element loop, the boundary conditions, the follower pressure and the
  convergence criterion, in `analyses.rst` and
  `ex_structsolve_nonlinear.rst`.

### Tests

- `structsolve>=0.8.5`, with its module `structsolve.convergence`, is a
  requirement of the tests, listed in
  `requirements.txt` and not in `install_requires`.
- `tests/test_static_axial_load_nonlinear.py` solves the non-linear static
  analysis with `structsolve.Analysis` instead of hand-written
  Newton-Raphson iterations, with the same reference values.
- New `tests/test_follower_pressure_structsolve.py`: the follower pressure
  on a cantilever with `structsolve.Analysis`, against the reference values
  of panels.
- `structsolve.lb` and `structsolve.freq` cross-check the solutions of
  `tests/test_linear_buckling_iso_SSSS.py` and
  `tests/test_natural_frequency.py`.

## 0.4.1 (2026-10-22)

- Build flags for speed: OpenMP removed, since no module uses `prange`;
  `-O3 -fno-math-errno` for GCC and Clang and `/O2` for MSVC; Cython line
  tracing only when `CYTHON_TRACE` is requested, with the C++ files
  regenerated when the trace mode changes; `initializedcheck=False`.

## 0.4.0 (2026-09-21)

- Fixed the tangent stiffness matrix and the internal force vector of the
  non-linear analyses, such that `KC0 + KCNL + KG` is the derivative of
  `fint`; the kernels are regenerated after the fixes of the derivations.
- Fixed and expanded non-linear tests, with the new
  `tests/test_tangent_consistency.py`.
- Python 3.13 and 3.14 support; updated versions of the GitHub actions.

## 0.3.1 (2024-09-25)

- Fixed the compatibility with integer numbers.
- New GitHub actions and deployment script.
- `tests/test_natural_frequency.py` uses `subset_by_index` of
  `scipy.linalg.eigh`.

## 0.3.0 (2023-04-12)

- Removed the dependency on the NumPy C API, improving the compatibility.
- Fixed an issue with the integer types `long` and `long long`.
- Updated workflows; the old continuous integration removed.

## 0.2.1 (2021-11-10)

- License updated to the 3-clause BSD license.

## 0.2.0 (2021-11-01)

- Geometrically non-linear analyses: `update_KCNL` and `update_fint`.

## 0.1.9 (2021-09-27)

- `update_KG_cte_N`, the geometric stiffness matrix of constant membrane
  forces `Nxx`, `Nyy` and `Nxy`, with a new test.
- Newer Python versions, faster tests and automated releases.

## 0.1.7 (2020-08-21)

- First tagged version, after the commit `REL: Version 0.1.0` of the same
  day: the BFSC plate element `BFSCPlate2D`, with `update_KC0`,
  `update_KG` and `update_M`, and the Gauss-Legendre points and weights of
  `bfscplate2d.quadrature`.

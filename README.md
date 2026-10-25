Github Actions status:
[![pytest](https://github.com/saullocastro/bfscplate2d/actions/workflows/pytest.yml/badge.svg)](https://github.com/saullocastro/bfscplate2d/actions/workflows/pytest.yml)
[![Deploy](https://github.com/saullocastro/bfscplate2d/actions/workflows/pythonpublish.yml/badge.svg)](https://github.com/saullocastro/bfscplate2d/actions/workflows/pythonpublish.yml)

Coverage status:
[![codecov](https://github.com/saullocastro/bfscplate2d/actions/workflows/coverage.yml/badge.svg)](https://github.com/saullocastro/bfscplate2d/actions/workflows/coverage.yml)
[![Codecov Status](https://codecov.io/gh/saullocastro/bfscplate2d/branch/main/graph/badge.svg?token=W7W7WZV59A)](https://codecov.io/gh/saullocastro/bfscplate2d)

Implementation of the BFSC plate finite element in 2D
=====================================================

- BFSC plate element, `BFSCPlate2D`: a four-node rectangular element of the
  classical laminated plate theory, with the nonlinear strains of von Karman
- Linear statics: [KC0]{u} = {f}
- Linear buckling: ([KC0] + lambda[KG]){u} = 0, with [KG] of the membrane
  forces of a static solution or of constant membrane forces
- Natural frequencies: ([KC0] - omega^2[M]){u} = 0
- Geometrically nonlinear statics: internal force vector and tangent
  stiffness matrix [KC0] + [KCNL] + [KG], its exact Jacobian
- Follower pressure, normal to the deformed plate: force vector and
  unsymmetric load stiffness matrix
- The contributions of each element are written in sparse arrays; the model,
  its boundary conditions and the solution of its equations are written by
  the user, with NumPy and SciPy

The ABD matrices of the laminates are calculated with my other repository
[composites](https://github.com/saullocastro/composites), and the tests and
usage examples solve the nonlinear analyses with
[structsolve](https://github.com/saullocastro/structsolve).


Citing this library
===================

Saullo G. P. Castro (2026). Implementation of the BFSC plate finite element in 2D (Version 0.5.0). https://github.com/saullocastro/bfscplate2d.


Documentation
=============

The documentation is available on: https://saullocastro.github.io/bfscplate2d.


History
=======

See [CHANGELOG.md](CHANGELOG.md) for the details of each version.

* version 0.5.0 (2026-10-26)
    - The shape functions renamed from `N` to `S`, as in bfsccylinder: the
      attributes `Nu`, ..., `Nw_y` and the methods `update_Nu`, ...,
      `update_Nw_y` of `BFSCPlate2D` are `Su`, ..., `Sw_y` and
      `update_Su`, ..., `update_Sw_y`
    - Follower pressure: `update_fext_follower` and `update_KCfollower`, the
      force vector and the unsymmetric load stiffness matrix of a pressure
      normal to the deformed mid-surface
    - BUG: `BFSCPlate2D.__init__` sets `init_k_KCNL`, which a second call of
      the constructor left at its previous value
    - Sphinx documentation, with a user guide, usage examples and the
      solution of the analyses with `structsolve.Analysis`
    - The tests solve the nonlinear analyses with `structsolve.Analysis` and
      cross-check the buckling loads and natural frequencies with
      `structsolve.lb` and `structsolve.freq`, requiring `structsolve>=0.9.0`
    - Metadata in `pyproject.toml`; Python 3.10 to 3.14
* version 0.4.1 (2026-10-22)
    - Build flags for speed, without OpenMP
* version 0.4.0 (2026-09-21)
    - Fixed the tangent stiffness matrix and the internal force vector of the
      nonlinear analyses, such that `KC0 + KCNL + KG` is the derivative of
      `fint`
    - Python 3.13 and 3.14 support
* version 0.3.1 (2024-09-25)
    - Fixed the compatibility with integer numbers
* version 0.3.0 (2023-04-12)
    - Removed the dependency on the NumPy C API
* version 0.2.1 (2021-11-10)
    - 3-Clause BSD license
* version 0.2.0 (2021-11-01)
    - Geometrically nonlinear analyses: `update_KCNL` and `update_fint`
* version 0.1.9 (2021-09-27)
    - `update_KG_cte_N`, the geometric stiffness matrix of constant membrane
      forces
* version 0.1.7 (2020-08-21)
    - First tagged version: `BFSCPlate2D`, with `update_KC0`, `update_KG` and
      `update_M`


License
=======

Distributed under the 3-Clause BSD license
(https://raw.github.com/saullocastro/bfscplate2d/main/LICENSE).

Contact: S.G.P.Castro@tudelft.nl

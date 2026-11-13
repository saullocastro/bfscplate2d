Analyses
========

.. currentmodule:: bfscplate2d.bfscplate2d

The matrices and vectors of :doc:`element_matrices` and
:doc:`follower_pressure`, assembled and restricted to the unknown degrees of
freedom as in :doc:`assembling`, give the equations of four analyses:

.. list-table::
    :header-rows: 1

    * - Analysis
      - Equations
      - Functions
    * - Linear static
      - `\boldsymbol{K}_{C0}\boldsymbol{u} = \boldsymbol{F}_{ext}`
      - :func:`.update_KC0`
    * - Linear buckling
      - `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G)\boldsymbol{\phi} =
        \boldsymbol{0}`
      - :func:`.update_KC0`, :func:`.update_KG` or
        :func:`.update_KG_cte_N`
    * - Natural frequencies
      - `(\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M})\boldsymbol{\phi} =
        \boldsymbol{0}`
      - :func:`.update_KC0`, :func:`.update_M`
    * - Nonlinear static
      - `\boldsymbol{F}_{int}(\boldsymbol{u}) = \lambda \boldsymbol{F}_{ext}`
      - :func:`.update_fint`, :func:`.update_KC0`, :func:`.update_KCNL`,
        :func:`.update_KG`

The library solves none of them: the systems are solved with SciPy or with
`structsolve <https://saullocastro.github.io/structsolve/>`_, see `Solving
with structsolve`_.


Linear static analysis
----------------------

For small displacements the internal forces of Eq. :eq:`em-fint` reduce to
`\boldsymbol{K}_{C0}\boldsymbol{u}`, and the equilibrium is

.. math::
    :label: an-static

    \boldsymbol{K}_{C0} \boldsymbol{u} = \boldsymbol{F}_{ext}

with `\boldsymbol{K}_{C0}` symmetric, positive definite once the rigid-body
motions are constrained, e.g. solved with ``scipy.sparse.linalg.spsolve``.
The membrane and bending problems are coupled only through
`\boldsymbol{B}`: for `\boldsymbol{B} = \boldsymbol{0}` in-plane loads give
no deflection and transverse loads no membrane forces.


Linear buckling analysis
------------------------

Under in-plane loads `\lambda \boldsymbol{F}_{ext}` a flat plate with
`\boldsymbol{B} = \boldsymbol{0}` has the linear pre-buckling state
`\lambda \boldsymbol{u}_0`, `\boldsymbol{K}_{C0}\boldsymbol{u}_0 =
\boldsymbol{F}_{ext}`, without deflection. Its slopes are null, hence
`\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} = \boldsymbol{0}`, and the
tangent stiffness matrix of Eq. :eq:`em-tangent` along this path is
`\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G(\boldsymbol{u}_0)`, since
`\boldsymbol{K}_G` is linear in the displacements. A bifurcation occurs
where it becomes singular:

.. math::
    :label: an-buckling

    (\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G) \boldsymbol{\phi} =
        \boldsymbol{0}

with the load multipliers `\lambda`, the critical loads being `\lambda
\boldsymbol{F}_{ext}`, and the buckling modes `\boldsymbol{\phi}`. A negative
`\lambda` is the critical load of the reversed load. `\boldsymbol{K}_G` is
computed in one of two ways, see :doc:`element_matrices`:

- :func:`.update_KG` with the solution `\boldsymbol{u}_0` of the linear
  static analysis of the reference load: the membrane forces are those of the
  plate with its boundary conditions and the actual distribution of the load;
- :func:`.update_KG_cte_N` with membrane forces given for each element,
  positive in tension, without a static analysis: `\lambda` multiplies
  these forces, e.g. ``Nxx = -1`` in every element gives `\lambda` equal to
  the critical compressive force per unit length.

The second is exact for a pre-buckling state of uniform membrane forces; the
first represents non-uniform states, e.g. from nodal forces or constrained
in-plane displacements, whose load multipliers therefore differ slightly
from those of the uniform state.

`\boldsymbol{K}_{C0}` is positive definite and `\boldsymbol{K}_G` symmetric
and indefinite, such that Eq. :eq:`an-buckling` is a symmetric generalised
eigenvalue problem. With `\boldsymbol{K}_{C0}` as the positive definite
matrix of the pencil it is written `\boldsymbol{K}_G \boldsymbol{\phi} = \mu
\boldsymbol{K}_{C0} \boldsymbol{\phi}`, `\mu = -1/\lambda`, the form that
the tests solve with ``scipy.sparse.linalg.eigsh`` (``A = KG``, ``M =
KC0``, shift-invert mode ``'cayley'``), recovering `\lambda = -1/\mu`. With
a follower pressure the problem becomes
Eq. :eq:`fp-buckling`, unsymmetric in general.

The classical buckling coefficients verify this formulation: for a simply
supported plate of sides `a` and `b` under a uniform compression
`N_{xx}` [TimoshenkoGere1961]_

.. math::
    :label: an-kc

    N_{xx,cr} = k_c \frac{\pi^2 D}{b^2}, \qquad
    k_c = \left( \frac{m b}{a} + \frac{a}{m b} \right)^2

with `D` of Eq. :eq:`pt-D-isotropic` and `m` the number of half-waves along
`x`, `k_c = 4` for a square plate, checked in
``tests/test_linear_buckling_iso_SSSS.py``.

**Limits.** The linear buckling analysis assumes a pre-buckling state
without rotations. For `\boldsymbol{B} \neq \boldsymbol{0}`, e.g.
unsymmetric laminates, or for loads with a transverse part, the
pre-buckling state deflects, `\boldsymbol{K}_{CNL}` and the second-order
terms are not null, and Eq. :eq:`an-buckling` is an approximation of the
nonlinear response, to be checked with the nonlinear static analysis. The
critical load does not describe the post-buckling behaviour, nor the
sensitivity to imperfections.


Natural frequencies
-------------------

The kinetic energy of Eq. :eq:`em-M` and the linear strain energy give the
equations of free vibration `\boldsymbol{M}\ddot{\boldsymbol{u}} +
\boldsymbol{K}_{C0}\boldsymbol{u} = \boldsymbol{0}`, whose harmonic
solutions `\boldsymbol{u} = \boldsymbol{\phi} \sin \omega_n t` satisfy

.. math::
    :label: an-frequency

    (\boldsymbol{K}_{C0} - \omega_n^2 \boldsymbol{M}) \boldsymbol{\phi} =
        \boldsymbol{0}

with `\omega_n` the natural angular frequencies, in radians per unit time,
`f_n = \omega_n/(2\pi)`, and `\boldsymbol{\phi}` the modes. Both matrices
are symmetric and positive definite once the rigid-body motions are
constrained, e.g. solved with ``scipy.linalg.eigh`` or with
``scipy.sparse.linalg.eigsh``. :func:`.update_M` needs ``h`` and ``rho``
of each element and no integration points. For a simply supported isotropic
plate, without the rotary inertia, the mode `w = \sin(m\pi x/a)\sin(n\pi
y/b)` gives `\omega_{mn} = \pi^2 (m^2/a^2 + n^2/b^2) \sqrt{D/(\rho h)}`,
checked in ``tests/test_natural_frequency.py``. The geometric stiffness
matrix of a static pre-stress, added to `\boldsymbol{K}_{C0}`, gives the
frequencies of the pre-stressed plate, which vanish at the buckling load.


Nonlinear static analysis
-------------------------

For deflections of the order of the thickness or larger, the membrane
stretching of the von Kármán strains carries part of the load and the
equilibrium

.. math::
    :label: an-residual

    \boldsymbol{R}(\boldsymbol{u}, \lambda) =
        \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_{ext}
        = \boldsymbol{0}

is solved for the displacements at increasing values of the load factor
`\lambda`. In the Newton-Raphson method each iteration solves

.. math::
    :label: an-newton

    \boldsymbol{K}_T \Delta\boldsymbol{u} = -\boldsymbol{R}, \qquad
    \boldsymbol{K}_T = \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}}
        = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u})
        + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u})

with the tangent stiffness matrix of :func:`.update_KC0`,
:func:`.update_KCNL` and :func:`.update_KG`, all computed with the current
displacements and the same integration points as `\boldsymbol{F}_{int}` of
:func:`.update_fint`. Since `\boldsymbol{K}_T` is the exact Jacobian of
`\boldsymbol{F}_{int}`, see :doc:`element_matrices`, the iterations converge
quadratically near the solution, as asserted in
``tests/test_static_axial_load_nonlinear.py``. The value arrays of KCNL and
KG are zeroed before each assembly, since the functions add to them. A
follower pressure adds its force vector to the residual and its load
stiffness matrix to `\boldsymbol{K}_T`, Eq. :eq:`fp-residual`.

Under load control the iterations fail at a limit point, where
`\boldsymbol{K}_T` becomes singular; beyond it, path-following
(arc-length) methods are needed. A perfect plate under in-plane
compression stays flat on the trivial path: the nonlinear analysis of
buckling and post-buckling needs a small perturbation, e.g. a small
transverse load or an initial deflection, that selects the buckled path.
Below the buckling load, the response to a small transverse load
`\boldsymbol{F}_t` superposed on the in-plane load `P` tends to the
linearised solution `(\boldsymbol{K}_{C0} + \boldsymbol{K}_G(P))
\boldsymbol{u} = \boldsymbol{F}_t`, with `\boldsymbol{K}_G` of the in-plane
load alone, amplified as the load approaches the buckling load.


.. _solving-with-structsolve:

Solving with structsolve
------------------------

The solvers of `structsolve <https://saullocastro.github.io/structsolve/>`_
receive the matrices and vectors of the model assembled by the user. It is
used by the tests and not required by bfscplate2d, see :doc:`install`. The
linear buckling and natural frequency problems are solved by
``structsolve.lb(Kuu, KGuu)``, which returns the load multipliers
`\lambda`, and ``structsolve.freq(Kuu, Muu)``, which returns `-\omega_n^2`.

The nonlinear static analysis is solved by ``structsolve.Analysis``, with
four callables that loop over the elements, call the functions of the
element and assemble:

- ``calc_fext(inc=1., silent=False)``, the external forces
  `\boldsymbol{F}_{ext}` times the load factor ``inc``;
- ``calc_fint(c, silent=False)``, the internal forces of
  :func:`.update_fint` at the displacements ``c``;
- ``calc_kC(c=None, NLgeom=False, silent=False)``, `\boldsymbol{K}_{C0}` with
  ``NLgeom=False``, and `\boldsymbol{K}_{C0} +
  \boldsymbol{K}_{CNL}(\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c})`
  with ``NLgeom=True``;
- ``calc_kG(c=None, NLgeom=False, silent=False)``,
  `\boldsymbol{K}_G(\boldsymbol{c})` of :func:`.update_KG`, called with
  ``NLgeom=True``;

such that ``calc_kC + calc_kG`` is `\boldsymbol{K}_T` and the residual of
structsolve is Eq. :eq:`an-residual`, with the corrections of
Eq. :eq:`an-newton`, the convention of structsolve. The load
factor goes from 0 to 1 in increments, the first of size
``Analysis.initialInc`` and the next ones growing up to ``Analysis.maxInc``,
each starting from a predictor extrapolated from the previous increment, the
linear solution of `\boldsymbol{K}_{C0}` in the first one;
``Analysis.static`` returns the load factors and the solutions of the
converged increments.

structsolve imposes no boundary condition itself. The constrained degrees
of freedom are removed either by the partition of :doc:`assembling`, the
callables receiving ``c = u[bu]`` and returning vectors and matrices
restricted to ``bu``, or by returning the vectors and matrices of the whole
model with the rows and columns of the constrained degrees of freedom zeroed
in all the matrices, which the solvers remove.

The convergence criterion is, by default, Crisfield's: the residual and the
reference forces, the external and the internal forces, are scaled with the
diagonal of the linear stiffness matrix ``calc_kC()`` over the free degrees
of freedom, and the iterations stop when their ratio is below
``Analysis.relTOL = 1e-9``. The scaled norm does not depend on the units of
the degrees of freedom, which here mix displacements, slopes and `w_{,xy}`,
see :doc:`element`; ``Analysis.convergence = 'relative'`` selects the
Euclidean norm instead.

A follower pressure `\lambda p` enters through the optional keyword argument
``inc``, the load factor, which structsolve passes to the callables that
declare it: ``calc_fext`` gives the dead part `\lambda
\boldsymbol{F}_p(\boldsymbol{0})`, ``calc_fint`` subtracts the part of
:func:`.update_fext_follower` with ``reference=0`` for the pressure
``inc*p``, and ``calc_kC`` adds the load stiffness matrix of
:func:`.update_KCfollower` for ``inc*p``, such that the residual and the
tangent are those of Eq. :eq:`fp-residual`. The tangent stiffness matrix is
then unsymmetric in general, which the solvers of structsolve support with
LU factorisations. For the arc-length methods, ``Analysis.NL_method``,
``calc_fext`` also declares the configuration ``c`` and returns the follower
force vector `\boldsymbol{F}_p(\boldsymbol{c})` of ``p``, minus the
derivative of the residual with respect to the load factor.


Choosing an analysis and its parameters
---------------------------------------

- **Linear static** for deflections small compared with the thickness;
  for larger ones the membrane stiffening, absent from the linear solution,
  reduces the deflections under transverse loads, and the nonlinear analysis
  is needed.
- **Linear buckling** for the critical loads of flat plates under in-plane
  loads; :func:`.update_KG_cte_N` when the pre-buckling membrane forces are
  known and uniform per element, :func:`.update_KG` with a static solution
  otherwise.
- **Nonlinear static** for large deflections, post-buckling with a
  perturbation, follower pressures, and to check a linear buckling analysis
  of a plate whose pre-buckling state deflects.
- **Integration points**: ``nint = 4`` integrates KC0 and the constant-force
  KG exactly, ``nint = 5`` the KG of a static solution, ``nint = 7`` the
  internal forces and KCNL, see :doc:`element_matrices`; the same points
  for every function of one analysis.
- **Mesh**: the accuracy of buckling loads and frequencies depends on the
  number of elements per half-wave of the modes of interest, to be checked
  by refining the mesh. Boundary conditions constraining only the nodal
  values, without the derivatives along the edge, converge with the mesh,
  see :doc:`assembling`.

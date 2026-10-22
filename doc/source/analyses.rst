Analyses
========

.. currentmodule:: bfscplate2d.bfscplate2d

The matrices and vectors of :doc:`assembling`, restricted to the unknown
degrees of freedom ``bu``, give the systems of the analyses below, solved
with SciPy as in the tests, or with structsolve, see `Solving with
structsolve`_. The theory of each matrix is in the Notes of its function.


Linear static analysis
----------------------

.. math::

    [K_{C0}]\{u\} = \{f\}

with `[K_{C0}]` of :func:`.update_KC0`, which needs ``lex``, ``ley`` and
``ABD`` of each element, and the force vector `\{f\}` of :doc:`assembling`.
The tests solve it with ``scipy.sparse.linalg.spsolve``, see
:doc:`ex_linear_static`.


Linear buckling analysis
------------------------

.. math::

    ([K_{C0}] + \lambda [K_G])\{\phi\} = \{0\}

The critical loads are `\lambda` times the load that gives `[K_G]`, and
`\{\phi\}` are the buckling modes. `[K_G]` is computed in one of two ways:

- :func:`.update_KG`, from the displacements `\{u_0\}` of the linear static
  solution of a reference load `\{f\}`, such that the membrane forces are
  those of the plate with its boundary conditions. Since `[K_G]` is linear
  in `\{u_0\}`, `\lambda` multiplies `\{f\}`;
- :func:`.update_KG_cte_N`, from membrane forces ``Nxx``, ``Nyy`` and
  ``Nxy`` given for each element, positive in tension, without a static
  analysis: `\lambda` multiplies these membrane forces, e.g. ``Nxx = -1``
  gives the critical compressive force per unit length.

The tests compute the lowest load multipliers with
``scipy.sparse.linalg.eigsh``, writing the problem as `[K_G]\{\phi\} = \mu
[K_{C0}]\{\phi\}`, with `[K_{C0}]` positive definite once the boundary
conditions are applied, and `\lambda = -1/\mu`::

    eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigenvalues, which='SM',
                              M=Kuu, tol=1e-6, sigma=1., mode='cayley')
    eigvals = -1./eigvals

see :doc:`ex_linear_buckling`.


Natural frequencies
-------------------

.. math::

    ([K_{C0}] - \omega_n^2 [M])\{\phi\} = \{0\}

with the mass matrix `[M]` of :func:`.update_M`, which needs ``lex``,
``ley``, ``h`` and ``rho`` of each element and no integration points, and
`\omega_n` the natural angular frequencies, in radians per unit time. The
test of :doc:`ex_natural_frequency` solves the problem with the dense
``scipy.linalg.eigh`` and takes `\omega_n` as the square roots of its
eigenvalues.


Nonlinear static analysis
-------------------------

For large deflections, with the nonlinear strains of von Karman, the
equilibrium of the internal forces of :func:`.update_fint` with the
external forces is solved for the displacements, e.g. with the
Newton-Raphson method:

.. math::

    \{R(u)\} = \{f_{int}(u)\} - \lambda \{f\}, \qquad
    [K_T(u)] = [K_{C0}] + [K_{CNL}(u)] + [K_G(u)], \qquad
    [K_T]\{\Delta u\} = -\{R\}

with the load factor `\lambda` increased in steps and the tangent stiffness
matrix `[K_T]` of :func:`.update_KC0`, :func:`.update_KCNL` and
:func:`.update_KG`, all computed with the current displacements and the
same integration points as `\{f_{int}\}`. `[K_T]` is the exact Jacobian of
`\{f_{int}\}`, such that the Newton-Raphson iterations converge
quadratically; ``tests/test_tangent_consistency.py`` checks it with Taylor
tests and finite differences. The internal forces always include the
nonlinear terms, and for small displacements `\{f_{int}\} \approx [K_{C0}]
\{u\}`.

The value arrays of `[K_{CNL}]` and `[K_G]` are zeroed before each
assembly, since the functions add to them, see :doc:`assembling`. The test
of :doc:`ex_nonlinear_static` solves the iterations with structsolve, see
`Solving with structsolve`_ and :doc:`ex_structsolve_nonlinear`. A follower
pressure adds its force vector to the residual and its load stiffness
matrix to `[K_T]`, see :doc:`follower_pressure`.


.. _solving-with-structsolve:

Solving with structsolve
------------------------

The solvers of `structsolve <https://saullocastro.github.io/structsolve/>`_
receive the matrices and vectors of the model, assembled by the user. It is
used by the tests and the examples, and it is not a requirement of
bfscplate2d, see :doc:`install`. The linear buckling and natural frequency
problems above are solved by ``structsolve.lb(Kuu, KGuu)`` and
``structsolve.freq(Kuu, Muu)``, which return the load multipliers
`\lambda` and `-\omega_n^2`, as cross-checked in
``tests/test_linear_buckling_iso_SSSS.py`` and
``tests/test_natural_frequency.py``.

The nonlinear static analysis is solved by ``structsolve.Analysis``, with
four callables that loop over the elements, call the functions of the
element and assemble, see :doc:`ex_structsolve_nonlinear`:

- ``calc_fext(inc=1., silent=False)``, the external forces `\{f\}` times
  the load factor ``inc``;
- ``calc_fint(c, silent=False)``, the internal forces of
  :func:`.update_fint` at the displacements ``c``;
- ``calc_kC(c=None, NLgeom=False, silent=False)``, `[K_{C0}]` with
  ``NLgeom=False``, and `[K_{C0}] + [K_{CNL}(c)]` with ``NLgeom=True``;
- ``calc_kG(c=None, NLgeom=False, silent=False)``, `[K_G(c)]` of
  :func:`.update_KG`, called with ``NLgeom=True``;

such that ``calc_kC + calc_kG`` is the tangent stiffness matrix `[K_T]` and
the residual of structsolve, `\lambda \{f\} - \{f_{int}(c)\}`, is the one
above with the opposite sign. The value arrays are zeroed in each call. The
load factor goes from 0 to 1 in increments, the first one of size
``Analysis.initialInc`` and the next ones growing up to
``Analysis.maxInc``, each starting from a predictor extrapolated from the
previous increment, the linear solution of `[K_{C0}]` in the first one;
``Analysis.static`` returns the load factors and the solutions of the
converged increments.

structsolve imposes no boundary condition itself. The constrained degrees
of freedom are removed in one of two ways:

- the partition of the free degrees of freedom, as in the tests: the
  callables receive ``c = u[bu]`` and return the vectors restricted to
  ``bu`` and the matrices to ``bu`` by ``bu``, and the solutions are
  expanded with ``u[bu] = c``;
- null rows and columns: the callables return the vectors and matrices of
  the whole model, with the rows and columns of the constrained degrees of
  freedom zeroed in all the matrices. The solvers remove them and return
  vectors with the size of the model, zero at these degrees of freedom.

The convergence criterion is, by default, Crisfield's: the residual and the
reference forces, the external forces and the internal forces, are scaled
with the diagonal of the linear stiffness matrix ``calc_kC()`` over the
free degrees of freedom, and the iterations stop when their ratio is below
``Analysis.relTOL = 1e-9``. The scaled norm does not depend on the units of
the degrees of freedom, which here mix displacements, their first
derivatives and `w_{,xy}`, and leaves out the reactions at the constrained
ones; ``Analysis.convergence = 'relative'`` selects the Euclidean norm
instead. With a consistent tangent the criterion is reached in a few
iterations.

A follower pressure `\lambda p`, see :doc:`follower_pressure`, enters
through the optional keyword argument ``inc``, the load factor, which
structsolve passes to the callables that declare it. The external forces
are the dead part `\{F_p(0)\}` of ``p``, ``calc_fint`` subtracts the part
that depends on the displacements, ``reference=0``, and ``calc_kC`` adds the
load stiffness matrix of `\lambda p`, as in
``tests/test_follower_pressure_structsolve.py``, see
:doc:`ex_structsolve_nonlinear`:

.. literalinclude:: ../../tests/test_follower_pressure_structsolve.py
    :pyobject: FollowerCallables

The tangent stiffness matrix is then unsymmetric in general, which the
solvers of structsolve support with LU factorizations. For the arc-length methods, ``Analysis.NL_method``, ``calc_fext`` also
declares the configuration ``c`` and returns the follower force vector
`\{F_p(c)\}` of ``p``, the derivative of the residual with respect to the
load factor.

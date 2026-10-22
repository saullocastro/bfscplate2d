Analyses
========

.. currentmodule:: bfscplate2d.bfscplate2d

The matrices and vectors of :doc:`assembling`, restricted to the unknown
degrees of freedom ``bu``, give the systems of the analyses below, solved
with SciPy as in the tests. The theory of each matrix is in the Notes of
its function.


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
of :doc:`ex_nonlinear_static` writes the iterations explicitly, with the
residual normalised by the norm of the external forces as the convergence
criterion. A follower pressure adds its force vector to the residual and its
load stiffness matrix to `[K_T]`, see :doc:`follower_pressure`.

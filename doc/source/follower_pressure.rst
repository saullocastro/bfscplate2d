Follower pressure
=================

.. currentmodule:: bfscplate2d.bfscplate2d

A follower pressure stays normal to the deformed plate and acts on its
deformed area, e.g. a gas pressure on a plate with large deflections, as
opposed to a dead pressure, which keeps the direction `z` and the
undeformed area. Its force vector depends on the displacements. Two
functions give it:

- :func:`.update_fext_follower` adds the force vector `\{F_p(u)\}` of one
  element to a global force vector;
- :func:`.update_KCfollower` writes the load stiffness matrix
  `[K_{Cfollower}] = -\partial \{F_p\}/\partial \{u\}` of one element in
  sparse arrays, at the position ``init_k_KCfollower``, with
  ``KCfollower_SPARSE_SIZE`` entries, see :doc:`assembling`.

The theory, from the area vector of the deformed mid-surface, is given in
the Notes of these functions and of :mod:`bfscplate2d.bfscplate2d`.


Pressure at the integration points
----------------------------------

The pressure ``p`` is given to both functions at the integration points of
each element, as a C-contiguous array of shape ``(nint, nint)``, ``p[i,
j]`` at ``xi = points[i]`` and ``eta = points[j]``, per unit area of the
undeformed plate and as a function of the undeformed coordinates. The
method ``Mesh.pressure`` of ``tests/test_follower_pressure.py`` evaluates a
function of `x` and `y` at these points from the position of node 1 of the
element, see :doc:`ex_follower_pressure`. The pressure is multiplied by the
load factor before it is given to the functions.


Sign
----

A positive ``p`` pushes along `+z`, like a positive force on `w`, in the
undeformed state. At `\{u\} = \{0\}` the force vector is the dead pressure
load, `\int_A p \{N_w\}^T dA`, the consistent load vector of a dead
pressure.


Truncation of the area vector
-----------------------------

``quadratic=0``, the default, truncates the area vector to the first order
in the displacement gradients, the truncation consistent with the moderate
rotations of the von Karman strains. `\{F_p\}` is then affine in the
displacements and `[K_{Cfollower}]` constant, computed once for a fixed
pressure. ``quadratic=1`` uses the complete area vector, e.g. to check that
a result does not depend on the truncation; `[K_{Cfollower}]` then depends
on the displacements and is computed at each iteration.

``reference=0`` in :func:`.update_fext_follower` leaves out the dead part,
giving `\{F_p(u)\} - \{F_p(0)\}`, which vanishes at `\{u\} = \{0\}`.


Residual and tangent stiffness matrix
-------------------------------------

With the pressure `\lambda p` of a load factor `\lambda`, the residual of
the nonlinear static analysis of :doc:`analyses` and its Jacobian are

.. math::

    \{R(u)\} = \{f_{int}(u)\} - \lambda \{F_p(u)\}, \qquad
    [K_T] = [K_{C0}] + [K_{CNL}(u)] + [K_G(u)] + \lambda [K_{Cfollower}]

with `\{F_p\}` and `[K_{Cfollower}]` of the same ``p`` and ``quadratic``.
Since the functions receive the pressure already multiplied by `\lambda`,
the test of :doc:`ex_follower_pressure` gives them ``lbd*func(x, y)``,
recomputes `\{F_p(u)\}` at every iteration and adds the load stiffness
matrix of the current pressure to the tangent. A solver whose external
force vector is fixed, `\lambda \{f\}`, takes the dead part `\{f\} =
\{F_p(0)\}` as external forces and the part ``reference=0``,
`\lambda(\{F_p(u)\} - \{F_p(0)\})`, subtracted from the internal forces.

A linear buckling analysis under a follower pressure `\lambda p` adds the
load stiffness matrix of ``p`` to `[K_G]` of the linear static solution of
``p``:

.. math::

    ([K_{C0}] + \lambda ([K_G] + [K_{Cfollower}]))\{\phi\} = \{0\}


Unsymmetric load stiffness matrix
---------------------------------

`[K_{Cfollower}]` is unsymmetric in general, a non-conservative load. The
assembled matrix restricted to the unknown degrees of freedom is symmetric
when the pressure is uniform over the whole plate and, on each edge, `w`
or the displacement normal to the edge is constrained, e.g. simply
supported or clamped edges and symmetry planes; the matrix of one element,
a free edge, a pressure on part of the plate or a varying pressure give an
unsymmetric matrix. The conditions are derived in the Notes of
:func:`.update_KCfollower` and checked by ``test_symmetry`` of
``tests/test_follower_pressure.py``. The matrix is written in full, not as
an upper triangle, and must not be symmetrized: the linear systems and the
eigenvalue problems are solved with the solvers of unsymmetric matrices,
e.g. ``scipy.sparse.linalg.spsolve``, as in the tests, and not with those
of symmetric matrices such as ``eigsh``.

Follower pressure
=================

.. currentmodule:: bfscplate2d.bfscplate2d

A follower pressure stays normal to the deformed plate and acts on its
deformed area, e.g. a gas pressure on a plate with large deflections, as
opposed to a dead pressure, which keeps the direction `z` and the
undeformed area. Its force vector depends on the displacements. Two
functions give it:

- :func:`.update_fext_follower` adds the force vector `\boldsymbol{F}_p(\boldsymbol{u})` of one
  element to a global force vector;
- :func:`.update_KCfollower` writes the load stiffness matrix
  `\boldsymbol{K}_F = -\partial \boldsymbol{F}_p/\partial \boldsymbol{u}` of one element in
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
undeformed state. At `\boldsymbol{u} = \boldsymbol{0}` the force vector is the dead pressure
load, `\int_A p (\boldsymbol{S}^w)^\top dA`, the consistent load vector of a dead
pressure.


Truncation of the area vector
-----------------------------

``quadratic=0``, the default, truncates the area vector to the first order
in the displacement gradients, the truncation consistent with the moderate
rotations of the von Karman strains. `\boldsymbol{F}_p` is then affine in the
displacements and `\boldsymbol{K}_F` constant, computed once for a fixed
pressure. ``quadratic=1`` uses the complete area vector, e.g. to check that
a result does not depend on the truncation; `\boldsymbol{K}_F` then depends
on the displacements and is computed at each iteration.

``reference=0`` in :func:`.update_fext_follower` leaves out the dead part,
giving `\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0})`, which vanishes at `\boldsymbol{u} = \boldsymbol{0}`.


Residual and tangent stiffness matrix
-------------------------------------

With the pressure `\lambda p` of a load factor `\lambda`, the residual of
the nonlinear static analysis of :doc:`analyses` and its Jacobian are

.. math::

    \boldsymbol{R}(\boldsymbol{u}) = \boldsymbol{F}_{int}(\boldsymbol{u}) - \lambda \boldsymbol{F}_p(\boldsymbol{u}), \qquad
    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{u}) + \boldsymbol{K}_{GNL}(\boldsymbol{u}) + \boldsymbol{K}_G(\boldsymbol{u}) + \boldsymbol{K}_F,
    \qquad \boldsymbol{K}_F = \lambda \boldsymbol{K}_{F_{ref}}

with `\boldsymbol{F}_p` and `\boldsymbol{K}_{F_{ref}}` of the same ``p`` and ``quadratic``.
Since the functions receive the pressure already multiplied by `\lambda`,
the test of :doc:`ex_follower_pressure` gives them ``lbd*func(x, y)``,
recomputes `\boldsymbol{F}_p(\boldsymbol{u})` at every iteration and adds the load stiffness
matrix of the current pressure to the tangent. A solver whose external
force vector is fixed, `\lambda \boldsymbol{F}_{ext}`, takes the dead part `\boldsymbol{F}_{ext} =
\boldsymbol{F}_p(\boldsymbol{0})` as external forces and the part ``reference=0``,
`\lambda(\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0}))`, subtracted from the internal forces.

A linear buckling analysis under a follower pressure `\lambda p` adds the
load stiffness matrix `\boldsymbol{K}_{F_{ref}}` of ``p`` to `\boldsymbol{K}_G` of the linear static solution of
``p``:

.. math::

    (\boldsymbol{K}_{C0} + \lambda (\boldsymbol{K}_G + \boldsymbol{K}_{F_{ref}}))\boldsymbol{\phi} = \boldsymbol{0}


Unsymmetric load stiffness matrix
---------------------------------

`\boldsymbol{K}_F` is unsymmetric in general, a non-conservative load. For
the first-order area vector, ``quadratic=0``, it is symmetric, and the
follower pressure conservative, if and only if the pressure is uniform
over the loaded region and, at each point of the boundary of the loaded
region, `w = 0` or the in-plane displacement normal to the boundary is
zero, e.g. simply supported or clamped edges and symmetry planes. The
matrix of one element, free or partially restrained edges, a pressure on
a patch whose edges lie inside the plate or a varying pressure give an
unsymmetric matrix; in the assembled matrix, restricted to the unknown
degrees of freedom, the contributions of the edges shared by two elements
cancel. The conditions are derived in the Notes of
:func:`.update_KCfollower` and checked by ``test_symmetry`` of
``tests/test_follower_pressure.py``. The matrix is written in full, not as
an upper triangle, and must not be symmetrized: the linear systems and the
eigenvalue problems are solved with the solvers of unsymmetric matrices,
e.g. ``scipy.sparse.linalg.spsolve``, as in the tests, and not with those
of symmetric matrices such as ``eigsh``.

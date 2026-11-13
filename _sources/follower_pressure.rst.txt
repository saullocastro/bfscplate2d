Follower pressure
=================

.. currentmodule:: bfscplate2d.bfscplate2d

A pressure of a gas or a liquid stays normal to the deformed plate and acts
on its deformed area, as opposed to a dead pressure, which keeps the
direction `z` and the undeformed area. For large deflections the
difference matters: the force vector of a follower pressure depends on the
displacements, and its derivative, the load stiffness matrix, enters the
tangent stiffness matrix [Hibbitt1979]_ [SchweizerhofRamm1984]_. This
chapter derives the two quantities computed by the library:

- :func:`.update_fext_follower`, the force vector
  `\boldsymbol{F}_p(\boldsymbol{u})` of one element, added to a global
  force vector;
- :func:`.update_KCfollower`, the load stiffness matrix `\boldsymbol{K}_F =
  -\partial \boldsymbol{F}_p/\partial \boldsymbol{u}_e` of one element,
  written in sparse arrays, see :doc:`assembling`.


Virtual work and area vector
----------------------------

The pressure `p(x, y)` is given per unit area of the undeformed plate, as
a function of the undeformed coordinates. With `\boldsymbol{X} =
\boldsymbol{X}_0 + \boldsymbol{u}` the deformed position of the reference
surface, `\boldsymbol{u} = \{u, v, w\}^\top`, its virtual work is

.. math::
    :label: fp-virtual-work

    \delta W_p = \int_A p \, \boldsymbol{a} \cdot \delta\boldsymbol{u} \, dx
        \, dy, \qquad
    \boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}, \qquad
    \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\ v_{,x} \\ w_{,x}
        \end{Bmatrix}, \qquad
    \boldsymbol{X}_{,y} = \begin{Bmatrix} u_{,y} \\ 1 + v_{,y} \\ w_{,y}
        \end{Bmatrix}

The area vector `\boldsymbol{a}` is normal to the deformed surface, and its
length is the ratio between the deformed and the undeformed areas. The cross
product is bilinear in the displacement gradients; split by orders,
`\boldsymbol{a} = \boldsymbol{a}_0 + \boldsymbol{a}_1 + \boldsymbol{a}_2`,

.. math::
    :label: fp-area-vector

    \boldsymbol{a}_0 = \begin{Bmatrix} 0 \\ 0 \\ 1 \end{Bmatrix}, \qquad
    \boldsymbol{a}_1 = \begin{Bmatrix} -w_{,x} \\ -w_{,y} \\
        u_{,x} + v_{,y} \end{Bmatrix}, \qquad
    \boldsymbol{a}_2 = \begin{Bmatrix} v_{,x} w_{,y} - w_{,x} v_{,y} \\
        w_{,x} u_{,y} - u_{,x} w_{,y} \\
        u_{,x} v_{,y} - v_{,x} u_{,y} \end{Bmatrix}

`\boldsymbol{a}_0` gives the dead pressure; `\boldsymbol{a}_1` contains the
rotation of the normal, `(-w_{,x}, -w_{,y})`, and the change of area
`u_{,x} + v_{,y}`.


Consistent truncation
---------------------

With the moderate rotations of the von Kármán strains of
:doc:`plate_theory`, `u_{,x} \sim \epsilon` and `w_{,x} \sim
\epsilon^{1/2}`, with `\epsilon` the order of the strains, such that `\delta
u \sim \epsilon^{1/2} \delta w`. `\boldsymbol{a}_0 + \boldsymbol{a}_1` then
keeps every term of `p \, \boldsymbol{a} \cdot \delta\boldsymbol{u}` up to
the order `\epsilon \, p \, \delta w`, while `\boldsymbol{a}_2` adds terms
of the order `\epsilon^2 p \, \delta w`, of the order of the terms that the
strains neglect. The first-order area vector, ``quadratic=0`` (the
default), is therefore the truncation consistent with the kinematics: the
force vector is affine in the displacements and the load stiffness matrix
constant, computed once for a given pressure. ``quadratic=1`` uses the
complete area vector, e.g. to check the sensitivity of a result to the
truncation; the load stiffness matrix then depends on the displacements and
is recomputed at each iteration.


Force vector
------------

Discretised with the rows of Eq. :eq:`el-rows` and the quadrature of
:doc:`element_matrices`, Eq. :eq:`fp-virtual-work` gives the element
force vector of :func:`.update_fext_follower`

.. math::
    :label: fp-force

    \boldsymbol{F}_p = \int_A p \left( (\boldsymbol{S}^u)^\top a_x +
        (\boldsymbol{S}^v)^\top a_y + (\boldsymbol{S}^w)^\top a_z \right) dA,
    \qquad
    \boldsymbol{a} = r \, \boldsymbol{a}_0 + \boldsymbol{a}_1 + q \,
        \boldsymbol{a}_2

with `q = 1` for ``quadratic`` not 0 and `r = 1` for ``reference`` not 0
(the default), otherwise 0. At `\boldsymbol{u} = \boldsymbol{0}`,
`\boldsymbol{F}_p(\boldsymbol{0}) = \int_A p (\boldsymbol{S}^w)^\top dA` is
the consistent load vector of the dead pressure. ``reference=0`` leaves this
term out and gives `\boldsymbol{F}_p(\boldsymbol{u}) -
\boldsymbol{F}_p(\boldsymbol{0})`, the part that vanishes in the undeformed
state.

The pressure is given to the functions at the integration points of each
element, a C-contiguous array ``p`` of shape ``(nint, nint)``, ``p[i, j]``
at ``xi = points[i]`` and ``eta = points[j]``, already multiplied by the
load factor if there is one. A positive ``p`` pushes along `+z` in the
undeformed state, like a positive force on `w`.


Load stiffness matrix
---------------------

The derivative of Eq. :eq:`fp-force` gives the load stiffness matrix of
:func:`.update_KCfollower`

.. math::
    :label: fp-KF

    \boldsymbol{K}_F = -\frac{\partial \boldsymbol{F}_p}{\partial
        \boldsymbol{u}_e} = -\int_A p \left( (\boldsymbol{S}^u)^\top
        \frac{\partial a_x}{\partial \boldsymbol{u}_e} + (\boldsymbol{S}^v)^\top
        \frac{\partial a_y}{\partial \boldsymbol{u}_e} + (\boldsymbol{S}^w)^\top
        \frac{\partial a_z}{\partial \boldsymbol{u}_e} \right) dA

For the first-order area vector, `\partial \boldsymbol{a}_1/\partial
\boldsymbol{u}_e` has the constant rows `-\boldsymbol{S}^w_{,x}`,
`-\boldsymbol{S}^w_{,y}` and `\boldsymbol{S}^u_{,x} +
\boldsymbol{S}^v_{,y}`, and

.. math::
    :label: fp-KF-linear

    \boldsymbol{K}_F = \int_A p \left( (\boldsymbol{S}^u)^\top
        \boldsymbol{S}^w_{,x} + (\boldsymbol{S}^v)^\top \boldsymbol{S}^w_{,y}
        - (\boldsymbol{S}^w)^\top \boldsymbol{S}^u_{,x}
        - (\boldsymbol{S}^w)^\top \boldsymbol{S}^v_{,y} \right) dA

independent of the displacements, with entries in the blocks `[u, w]`,
`[v, w]`, `[w, u]` and `[w, v]`. The complete area vector adds `\partial
\boldsymbol{a}_2/\partial \boldsymbol{u}_e`, linear in the displacement
gradients, to these blocks and to the blocks `[u, v]` and `[v, u]`. The
function writes all six blocks, ``KCfollower_SPARSE_SIZE`` entries per
element, the values of `[u, v]` and `[v, u]` being zero for
``quadratic=0``.


Symmetry and conservativeness
-----------------------------

A load derived from a potential has a symmetric load stiffness matrix. For
the first-order area vector, `\boldsymbol{\delta}^\top \boldsymbol{K}_F
\boldsymbol{\Delta} = -P(\delta, \Delta)` for two displacement fields
`\delta` and `\Delta`, with the bilinear form `P(\delta, \Delta) = \int_A p
\, \delta\boldsymbol{u} \cdot \boldsymbol{a}_1(\Delta\boldsymbol{u}) \,
dA`. Its skew part collects into derivatives, `p \, \partial_x (\Delta u \,
\delta w - \delta u \, \Delta w)` and the same along `y`, and an integration
by parts gives

.. math::
    :label: fp-skew

    P(\delta, \Delta) - P(\Delta, \delta) =
    - \oint p \left[ (\delta u \, \Delta w - \delta w \, \Delta u) n_x
      + (\delta v \, \Delta w - \delta w \, \Delta v) n_y \right] ds
    + \int_A \left[ p_{,x} (\delta u \, \Delta w - \delta w \, \Delta u)
      + p_{,y} (\delta v \, \Delta w - \delta w \, \Delta v) \right] dA

with `\{n_x, n_y\}` the outward normal of the boundary of the loaded region.
The load stiffness matrix is therefore symmetric, and the follower pressure
conservative, if and only if

- `p` is uniform over the loaded region, and
- at each point of the boundary of the loaded region either `w = 0` or the
  in-plane displacement normal to the boundary vanishes, `u = 0` on the
  edges `x` = constant and `v = 0` on the edges `y` = constant.

Simply supported and clamped edges and symmetry planes meet the second
condition; the load potential is then the pressure times the change of the
enclosed volume. Free or partially restrained edges, a pressure on a patch
whose edges lie inside the plate and a pressure that varies over the plate
give an unsymmetric matrix, the result of [Hibbitt1979]_ and
[SchweizerhofRamm1984]_ for closed surfaces and for loaded surfaces with a
fixed boundary.

The boundary of one element is the boundary of its loaded region, where the
second condition does not hold: the element matrix is unsymmetric. In the
assembled matrix the contour integrals of the edges shared by two elements
cancel when `p` is continuous, since `u`, `v` and `w` are continuous, see
:doc:`element`, and the conditions apply to the boundary of the loaded
region of the plate. The matrix is written in full, not as a triangle, and
must not be symmetrised: the linear systems and eigenvalue problems that
contain it need solvers of unsymmetric matrices. These conditions are those
of the first-order area vector; for ``quadratic=1`` the matrix depends on
the displacements.


Residual and tangent stiffness matrix
-------------------------------------

With the pressure `\lambda p` of a load factor `\lambda`, the residual of
the nonlinear static analysis of :doc:`analyses` and its Jacobian are

.. math::
    :label: fp-residual

    \boldsymbol{R}(\boldsymbol{u}) = \boldsymbol{F}_{int}(\boldsymbol{u}) -
        \lambda \boldsymbol{F}_p(\boldsymbol{u}), \qquad
    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} +
        \boldsymbol{K}_{GNL} + \boldsymbol{K}_G + \boldsymbol{K}_F, \qquad
    \boldsymbol{K}_F = \lambda \boldsymbol{K}_{F_{ref}}

with `\boldsymbol{F}_p` and `\boldsymbol{K}_{F_{ref}}` of the same ``p``
and ``quadratic``. Since the functions receive the pressure already
multiplied by `\lambda`, the force vector and the load stiffness matrix of
`\lambda p` are computed by giving them ``lbd*p``. Writing
`\boldsymbol{F}_p(\boldsymbol{u}) = \boldsymbol{F}_p(\boldsymbol{0}) +
(\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0}))`, a
solver whose external force vector is fixed, `\lambda
\boldsymbol{F}_{ext}`, takes the dead part `\boldsymbol{F}_{ext} =
\boldsymbol{F}_p(\boldsymbol{0})` as external forces and the part of
``reference=0``, times `\lambda`, subtracted from the internal forces, see
:doc:`analyses`.

A linear buckling analysis under a follower pressure `\lambda p` adds the
load stiffness matrix of ``p`` to the geometric stiffness matrix of the
linear static solution of ``p``:

.. math::
    :label: fp-buckling

    \left( \boldsymbol{K}_{C0} + \lambda (\boldsymbol{K}_G +
        \boldsymbol{K}_{F_{ref}}) \right) \boldsymbol{\phi} = \boldsymbol{0}

an unsymmetric eigenvalue problem unless the conditions above hold.


Numerical integration
---------------------

For a pressure constant over the element the integrands of Eqs.
:eq:`fp-force` and :eq:`fp-KF` are polynomials of degree at most 6 in `\xi`
and in `\eta` for ``quadratic=0`` and 8 for ``quadratic=1``, integrated
exactly with ``nint >= 4`` and ``nint >= 5``.

The force vector, the load stiffness matrix, its symmetry conditions and the
consistency of the tangent stiffness matrix are verified in
``tests/test_follower_pressure.py``.

Element matrices and internal forces
====================================

.. currentmodule:: bfscplate2d.bfscplate2d

With the interpolation of :doc:`element`, the strain energy of
Eq. :eq:`pt-strain-energy` becomes a function of the 40 element degrees of
freedom `\boldsymbol{u}_e`. Its gradient is the internal force vector, and
its Hessian the tangent stiffness matrix, which the library splits into the
matrices that the linear and nonlinear analyses of :doc:`analyses` need.
This chapter derives them, as computed by the functions of
:mod:`bfscplate2d.bfscplate2d`, and the numerical integration that the
functions use.


Discrete strain energy and internal forces
------------------------------------------

Substituting Eq. :eq:`el-strain-operators` in Eq. :eq:`pt-strain-energy`,
with the membrane forces and moments

.. math::
    :label: em-resultants

    \boldsymbol{N} = \boldsymbol{A} \left( \boldsymbol{B}_m + \frac{1}{2}
        \boldsymbol{B}_{NL} \right) \boldsymbol{u}_e
        + \boldsymbol{B} \boldsymbol{B}_b \boldsymbol{u}_e, \qquad
    \boldsymbol{M} = \boldsymbol{B} \left( \boldsymbol{B}_m + \frac{1}{2}
        \boldsymbol{B}_{NL} \right) \boldsymbol{u}_e
        + \boldsymbol{D} \boldsymbol{B}_b \boldsymbol{u}_e

the first variation `\delta U = \delta\boldsymbol{u}_e^\top
\boldsymbol{F}_{int}` gives the internal force vector of
:func:`.update_fint`

.. math::
    :label: em-fint

    \boldsymbol{F}_{int} = \frac{\partial U}{\partial \boldsymbol{u}_e}
        = \int_A \left( (\boldsymbol{B}_m + \boldsymbol{B}_{NL})^\top
        \boldsymbol{N} + \boldsymbol{B}_b^\top \boldsymbol{M} \right) dA

which always includes the nonlinear terms. For small displacements
`\boldsymbol{B}_{NL} \to \boldsymbol{0}` and `\boldsymbol{F}_{int} \approx
\boldsymbol{K}_{C0} \boldsymbol{u}_e`, the linear relation below.


Tangent stiffness matrix
------------------------

The derivative of Eq. :eq:`em-fint` has two parts: the derivative of
`\boldsymbol{N}` and `\boldsymbol{M}`, with `\partial
\boldsymbol{\epsilon}/\partial \boldsymbol{u}_e = \boldsymbol{B}_m +
\boldsymbol{B}_{NL}`, and the derivative of `\boldsymbol{B}_{NL}^\top` at
constant `\boldsymbol{N}`. Writing
`\boldsymbol{B}_{NL}^\top \boldsymbol{N} = (\boldsymbol{S}^w_{,x})^\top (N_{xx}
w_{,x} + N_{xy} w_{,y}) + (\boldsymbol{S}^w_{,y})^\top (N_{xy} w_{,x} +
N_{yy} w_{,y})`, the second part is

.. math::
    :label: em-geometric-term

    \frac{\partial (\boldsymbol{B}_{NL}^\top \boldsymbol{N})}{\partial
        \boldsymbol{u}_e} \bigg|_{\boldsymbol{N}} =
    \boldsymbol{G}^\top \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy}
        \end{bmatrix} \boldsymbol{G}

and the tangent stiffness matrix is

.. math::
    :label: em-tangent

    \frac{\partial \boldsymbol{F}_{int}}{\partial \boldsymbol{u}_e}
    = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}
      + \boldsymbol{K}_G

with the terms below. It is symmetric, the Hessian of `U`.

**Linear constitutive stiffness** `\boldsymbol{K}_{C0}`,
:func:`.update_KC0`, the terms without `\boldsymbol{B}_{NL}`:

.. math::
    :label: em-KC0

    \boldsymbol{K}_{C0} = \int_A \left( \boldsymbol{B}_m^\top \boldsymbol{A}
        \boldsymbol{B}_m + \boldsymbol{B}_m^\top \boldsymbol{B}
        \boldsymbol{B}_b + \boldsymbol{B}_b^\top \boldsymbol{B}
        \boldsymbol{B}_m + \boldsymbol{B}_b^\top \boldsymbol{D}
        \boldsymbol{B}_b \right) dA

the membrane stiffness, the membrane-bending coupling, null for
`\boldsymbol{B} = \boldsymbol{0}`, and the bending stiffness; independent of
the displacements.

**Nonlinear constitutive stiffness** `\boldsymbol{K}_{CNL} =
\boldsymbol{K}_{C0L} + \boldsymbol{K}_{CL0} + \boldsymbol{K}_{CLL}`, the
terms with `\boldsymbol{B}_{NL}`:

.. math::
    :label: em-KCNL

    \boldsymbol{K}_{C0L} = \int_A \left( \boldsymbol{B}_m^\top \boldsymbol{A}
        + \boldsymbol{B}_b^\top \boldsymbol{B} \right) \boldsymbol{B}_{NL}
        \, dA, \qquad
    \boldsymbol{K}_{CL0} = \boldsymbol{K}_{C0L}^\top, \qquad
    \boldsymbol{K}_{CLL} = \int_A \boldsymbol{B}_{NL}^\top \boldsymbol{A}
        \boldsymbol{B}_{NL} \, dA

the first two linear, the third quadratic in the slopes.

**Geometric stiffness**, Eq. :eq:`em-geometric-term`, split according to
the two parts of `\boldsymbol{N}` in Eq. :eq:`em-resultants`:

.. math::
    :label: em-KG

    \boldsymbol{K}_G = \int_A \boldsymbol{G}^\top \begin{bmatrix} N_{xx} &
        N_{xy} \\ N_{xy} & N_{yy} \end{bmatrix} \boldsymbol{G} \, dA,
    \qquad
    \{N_{xx}, N_{yy}, N_{xy}\}^\top = \boldsymbol{A} \boldsymbol{B}_m
        \boldsymbol{u}_e + \boldsymbol{B} \boldsymbol{B}_b \boldsymbol{u}_e

the matrix of the membrane forces of the linear strains, linear in
`\boldsymbol{u}_e`, and `\boldsymbol{K}_{GNL} = \boldsymbol{K}_{GLL}`, the
same integral with the membrane forces of the nonlinear strains
`\boldsymbol{A} \{w_{,x}^2/2, w_{,y}^2/2, w_{,x} w_{,y}\}^\top`, quadratic
in the slopes.

The functions group the terms as follows:

.. list-table::
    :header-rows: 1

    * - Function
      - Matrix
      - Depends on `\boldsymbol{u}_e`
      - Pairs of degrees of freedom written
    * - :func:`.update_KC0`
      - `\boldsymbol{K}_{C0}`
      - no
      - all, 40 by 40
    * - :func:`.update_KCNL`
      - `\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`
        (KC0L + KCL0 + KCLL + KGLL)
      - linear and quadratic in the slopes, null for null slopes
      - in-plane with out-of-plane, both orders, and out-of-plane with
        out-of-plane
    * - :func:`.update_KG`
      - `\boldsymbol{K}_G` of Eq. :eq:`em-KG`
      - linear
      - out-of-plane with out-of-plane
    * - :func:`.update_KG_cte_N`
      - `\boldsymbol{K}_G` of given ``Nxx``, ``Nyy``, ``Nxy``, uniform over
        the element
      - no
      - out-of-plane with out-of-plane

`\boldsymbol{K}_{CNL}` and `\boldsymbol{K}_{GNL}` are in one function
because both vanish in the undeformed state and are needed only by the
nonlinear analysis, whereas `\boldsymbol{K}_G` is also the matrix of the
linear buckling analysis. Since `\boldsymbol{B}_{NL}` and `\boldsymbol{G}`
have entries only in the columns of `w` to `w_{,xy}`, no nonlinear term
couples two in-plane degrees of freedom.

Because the internal forces and the three stiffness functions use the same
operators and the same integration points, KC0 + KCNL + KG is the exact
Jacobian of the internal forces computed with those points, whatever their
number: Newton-Raphson iterations with this tangent converge quadratically,
see :doc:`analyses`. This is checked by Taylor tests and finite differences
in ``tests/test_tangent_consistency.py``.


Geometric stiffness of a linear buckling analysis
--------------------------------------------------

At the solution `\boldsymbol{u}_0` of a linear static analysis, the
membrane forces of Eq. :eq:`em-KG` are those of the pre-buckling state of
the plate, with its boundary conditions and the distribution of the load,
and :func:`.update_KG` computed with `\boldsymbol{u}_0` is proportional to
the load. :func:`.update_KG_cte_N` gives the same matrix for membrane forces
prescribed for each element, positive in tension, without a static
analysis. The two coincide when the static solution has uniform membrane
forces; they differ when the edges, the supports or the application of the
load make the pre-buckling state non-uniform, e.g. nodal forces at an edge
or constrained in-plane displacements, see :doc:`analyses`.


Mass matrix
-----------

The kinetic energy of the displacements of Eq. :eq:`pt-displacements`, for
a density ``rho`` uniform through the thickness ``h`` and the reference
surface at mid-thickness, gives the consistent mass matrix of
:func:`.update_M`

.. math::
    :label: em-M

    \boldsymbol{M} = \int_A \rho \left( h \left( (\boldsymbol{S}^u)^\top
        \boldsymbol{S}^u + (\boldsymbol{S}^v)^\top \boldsymbol{S}^v
        + (\boldsymbol{S}^w)^\top \boldsymbol{S}^w \right)
        + \frac{h^3}{12} \left( (\boldsymbol{S}^w_{,x})^\top
        \boldsymbol{S}^w_{,x} + (\boldsymbol{S}^w_{,y})^\top
        \boldsymbol{S}^w_{,y} \right) \right) dA

The terms with `h^3/12` are the rotary inertia of the normal, small for thin
plates. The terms with `\int \rho z \, dz`, which couple `u` and `v` with
the slopes of `w`, vanish for the reference surface at mid-thickness and
are not included: ``M`` has no entries between in-plane and out-of-plane
degrees of freedom. For a laminate of plies of different densities, ``rho``
can be the mean density, which gives the mass per unit area `\rho h`
exactly, but not the rotary inertia `\int \rho z^2 dz`. The integrands are
polynomials, integrated in closed form: :func:`.update_M` takes no
integration points. The matrix is independent of ``ABD`` and of the
displacements.


Numerical integration
---------------------

All the functions except :func:`.update_M` integrate over the element with
the products of the one-dimensional Gauss-Legendre points and weights of
:func:`bfscplate2d.quadrature.get_points_weights`, ``nint`` along each of
`\xi` and `\eta`:

.. math::
    :label: em-quadrature

    \int_A f \, dA = \frac{lex \, ley}{4} \int_{-1}^{1} \int_{-1}^{1} f \,
        d\xi \, d\eta \approx \frac{lex \, ley}{4} \sum_{i=1}^{nint}
        \sum_{j=1}^{nint} weights_i \, weights_j \, f(points_i, points_j)

with ``0.25*lex*ley*weight`` of the code the factor of each point. The
points are the roots of the Legendre polynomial of degree ``nint``, and the
rule integrates exactly the polynomials of degree up to `2 \, nint - 1` in
each variable. Since the interpolation functions are bicubic, the integrand
of each matrix is a polynomial whose degree, counted from those of the
operators (`\boldsymbol{B}_m`, `\boldsymbol{B}_b`, `\boldsymbol{G}` and the
membrane forces of the linear strains of degree at most 3 in each variable,
`\boldsymbol{B}_{NL}` of degree at most 6), gives the number of points for
an exact integration with a constant ``ABD``:

.. list-table::
    :header-rows: 1

    * - Matrix or vector
      - Maximum degree in `\xi` or `\eta`
      - ``nint`` for exact integration
    * - KC0, :func:`.update_KG_cte_N`
      - 6
      - 4
    * - KG of :func:`.update_KG`
      - 9
      - 5
    * - KCNL, internal forces
      - 12
      - 7
    * - follower pressure, first-order area vector, see
        :doc:`follower_pressure`
      - 6
      - 4
    * - follower pressure, complete area vector
      - 8
      - 5

With fewer points the matrices are approximate but remain consistent with
each other, as stated above. The same ``points`` and ``weights`` are passed
to every function of an analysis that combines them, e.g. ``nint = 7``,
which integrates exactly every term of a nonlinear static analysis.

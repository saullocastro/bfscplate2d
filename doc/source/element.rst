The BFSC plate element
======================

.. currentmodule:: bfscplate2d.bfscplate2d

:class:`.BFSCPlate2D` approximates the plate theory of :doc:`plate_theory`
over a rectangle with four nodes. Its name comes from the element of
Bogner, Fox and Schmit (BFS) [Bogner1965]_, for the deflection `w`, whose
interpolation the BFSC element extends to the in-plane displacements `u`
and `v` [CastroJansen2021]_. This chapter derives its interpolation
functions, its degrees of freedom and the operators that give the strains;
:doc:`element_matrices` uses them to build the matrices.


Why the interpolation of `w` must be C\ :sup:`1`
------------------------------------------------

The curvatures of Eq. :eq:`pt-strains` contain the second derivatives of
`w`. The strain energy of an approximation of `w` built element by element
is finite, and the approximation is conforming, only when `w` and its first
derivatives `w_{,x}` and `w_{,y}` are continuous across the edges between
elements, i.e. when `w` is of class C\ :sup:`1`; within an element the
second derivatives need only be integrable. Interpolations with the nodal
values of `w` alone, as in the membrane elements, are C\ :sup:`0`: the slope
normal to an edge jumps from one element to the next.

On a mesh of rectangles with sides parallel to the axes, C\ :sup:`1`
continuity is obtained with products of one-dimensional cubic Hermite
polynomials, which interpolate a function from its values and first
derivatives at the two ends of an interval [Bogner1965]_. The products of
the four Hermite polynomials along `x` with the four along `y` are 16
functions, which span all the bicubic polynomials, and need four degrees of
freedom of `w` per node: `w`, `w_{,x}`, `w_{,y}` and the cross derivative
`w_{,xy}`, the nodal value multiplying the product of the two slope
functions. This is the BFS element. The construction uses the constant
mapping between the natural and the physical coordinates of a rectangle
with sides parallel to `x` and `y`, below, to which the element is
restricted.


Geometry and natural coordinates
--------------------------------

The element has the sides ``lex`` along `x` and ``ley`` along `y`. Its
natural coordinates `\xi` and `\eta` (``xi`` and ``eta``) go from `-1` to
`1`:

.. math::
    :label: el-mapping

    x = x_1 + \frac{lex}{2} (\xi + 1), \qquad
    y = y_1 + \frac{ley}{2} (\eta + 1), \qquad
    \frac{\partial}{\partial x} = \frac{2}{lex} \frac{\partial}{\partial \xi},
    \qquad
    \frac{\partial}{\partial y} = \frac{2}{ley} \frac{\partial}{\partial \eta},
    \qquad
    dA = \frac{lex \, ley}{4} \, d\xi \, d\eta

with `(x_1, y_1)` the position of node 1. The nodes are numbered
counterclockwise from the corner with the smallest `x` and `y`, as the
CQUAD4 element of Nastran:

.. list-table::
    :header-rows: 1

    * - Node
      - `(\xi_i, \eta_i)`
      - Position
      - Attribute of its first degree of freedom
    * - 1
      - `(-1, -1)`
      - `(x_1, y_1)`
      - ``c1``
    * - 2
      - `(+1, -1)`
      - `(x_1 + lex, y_1)`
      - ``c2``
    * - 3
      - `(+1, +1)`
      - `(x_1 + lex, y_1 + ley)`
      - ``c3``
    * - 4
      - `(-1, +1)`
      - `(x_1, y_1 + ley)`
      - ``c4``

The element knows neither `x_1` nor `y_1`: its matrices depend on ``lex``,
``ley``, ``ABD`` (and ``h``, ``rho``) and on its degrees of freedom, not on
its position.


Hermite polynomials and interpolation functions
-----------------------------------------------

For a node at `\xi_i = \pm 1`, the two cubic Hermite polynomials of that
end of the interval `[-1, 1]` are

.. math::
    :label: el-hermite-1d

    f_i(\xi) = \frac{1}{4} (\xi + \xi_i)^2 (2 - \xi \xi_i), \qquad
    g_i(\xi) = \frac{1}{4} \xi_i (\xi + \xi_i)^2 (\xi \xi_i - 1)

with `f_i(\xi_i) = 1`, `f_i'(\xi_i) = 0`, `g_i(\xi_i) = 0`, `g_i'(\xi_i) =
1`, and `f_i`, `f_i'`, `g_i`, `g_i'` all null at the other end, `\xi =
-\xi_i`: `f_i` carries the value and `g_i` the derivative with respect to
`\xi` at the node. The interpolation functions of node `i` are their
products, with the factors `lex/2` and `ley/2` that turn a derivative with
respect to `\xi` or `\eta` into one with respect to `x` or `y`:

.. math::
    :label: el-shape-functions

    H_i = f_i(\xi) f_i(\eta), \qquad
    H_{x,i} = \frac{lex}{2} g_i(\xi) f_i(\eta), \qquad
    H_{y,i} = \frac{ley}{2} f_i(\xi) g_i(\eta), \qquad
    H_{xy,i} = \frac{lex \, ley}{4} g_i(\xi) g_i(\eta)

Each function is 1 for the nodal degree of freedom it multiplies, e.g.
`\partial H_{x,i}/\partial x = 1` and `\partial^2 H_{xy,i}/\partial x
\partial y = 1` at node `i`, and 0 for every other nodal degree of freedom of
the element. Expanded, Eq. :eq:`el-shape-functions` is the form given in
the Notes of :mod:`bfscplate2d.bfscplate2d` [OchoaReddy1992]_, e.g. `H_i =
(\xi + \xi_i)^2 (\xi \xi_i - 2) (\eta + \eta_i)^2 (\eta \eta_i - 2)/16`. The
functions are bicubic, of degree 3 in `\xi` and in `\eta`, which sets the
number of integration points of each matrix, see :doc:`element_matrices`.


Degrees of freedom and interpolation
------------------------------------

In the BFSC element the in-plane displacements are interpolated with the
same functions as `w`, without the cross derivatives [CastroJansen2021]_:

.. math::
    :label: el-interpolation

    u = \sum_{i=1}^{4} \left( H_i u_i + H_{x,i} u_{,x i} + H_{y,i} u_{,y i}
        \right), \qquad
    v = \sum_{i=1}^{4} \left( H_i v_i + H_{x,i} v_{,x i} + H_{y,i} v_{,y i}
        \right),

    w = \sum_{i=1}^{4} \left( H_i w_i + H_{x,i} w_{,x i} + H_{y,i} w_{,y i}
        + H_{xy,i} w_{,xy i} \right)

Each node has therefore ``DOF = 10`` degrees of freedom, at the positions
``c`` to ``c + 9`` of the global vectors and matrices, ``c`` being the
attribute ``c1``, ``c2``, ``c3`` or ``c4`` of the node:

==========  ===================  ===================
Position    Degree of freedom    Dimension
==========  ===================  ===================
``c + 0``   `u`                  length
``c + 1``   `u_{,x}`             dimensionless
``c + 2``   `u_{,y}`             dimensionless
``c + 3``   `v`                  length
``c + 4``   `v_{,x}`             dimensionless
``c + 5``   `v_{,y}`             dimensionless
``c + 6``   `w`                  length
``c + 7``   `w_{,x}`             dimensionless
``c + 8``   `w_{,y}`             dimensionless
``c + 9``   `w_{,xy}`            one over length
==========  ===================  ===================

The 40 degrees of freedom of the element, the vector `\boldsymbol{u}_e`
(``ue`` in the code, the attribute ``u`` of the element), are those of the
nodes 1, 2, 3 and 4 in sequence, 10 per node in the order of the table. In
matrix form

.. math::
    :label: el-rows

    u = \boldsymbol{S}^u \boldsymbol{u}_e, \qquad
    v = \boldsymbol{S}^v \boldsymbol{u}_e, \qquad
    w = \boldsymbol{S}^w \boldsymbol{u}_e

with the rows of 40 entries ``Su``, ``Sv`` and ``Sw``, computed at a point
`(\xi, \eta)` by :meth:`.BFSCPlate2D.update_Su`,
:meth:`.BFSCPlate2D.update_Sv` and :meth:`.BFSCPlate2D.update_Sw`.

Every function of Eq. :eq:`el-shape-functions` has a null cross
derivative at the four nodes, except `H_{xy,i}` at node `i`. Without
`H_{xy,i}`, the interpolation of `u` and `v` therefore gives `u_{,xy} =
v_{,xy} = 0` at the nodes: the 12 functions of `u` (and of `v`) contain the
linear polynomials, hence the states of constant membrane strain, and e.g.
`x^2` and `y^3`, but not `xy`, whose cross derivative is 1 everywhere.

The degrees of freedom mix lengths, dimensionless slopes and the cross
derivative of dimension one over length. Norms of vectors of degrees of
freedom, e.g. in a convergence criterion, depend on the units unless they
are scaled, see :doc:`analyses`.


Derivatives and strain operators
--------------------------------

With the constant mapping of Eq. :eq:`el-mapping`, the derivatives of the
rows are

.. math::
    :label: el-row-derivatives

    \boldsymbol{S}^u_{,x} = \frac{2}{lex}
        \frac{\partial \boldsymbol{S}^u}{\partial \xi}, \quad
    \boldsymbol{S}^u_{,y} = \frac{2}{ley}
        \frac{\partial \boldsymbol{S}^u}{\partial \eta}, \quad
    \boldsymbol{S}^w_{,xx} = \frac{4}{lex^2}
        \frac{\partial^2 \boldsymbol{S}^w}{\partial \xi^2}, \quad
    \boldsymbol{S}^w_{,xy} = \frac{4}{lex \, ley}
        \frac{\partial^2 \boldsymbol{S}^w}{\partial \xi \partial \eta}, \quad
    \ldots

the rows ``Su_x``, ``Su_y``, ``Sv_x``, ``Sv_y``, ``Sw_x`` and ``Sw_y`` of
:meth:`.BFSCPlate2D.update_Su_x` to :meth:`.BFSCPlate2D.update_Sw_y`.
Substituted in Eq. :eq:`pt-strains`, they give the strains in terms of the
element degrees of freedom:

.. math::
    :label: el-strain-operators

    \boldsymbol{\epsilon} = \left( \boldsymbol{B}_m + \frac{1}{2}
        \boldsymbol{B}_{NL} \right) \boldsymbol{u}_e, \qquad
    \delta\boldsymbol{\epsilon} = \left( \boldsymbol{B}_m +
        \boldsymbol{B}_{NL} \right) \delta\boldsymbol{u}_e, \qquad
    \boldsymbol{\kappa} = \boldsymbol{B}_b \boldsymbol{u}_e

with the operators, named as in the code:

.. list-table::
    :header-rows: 1

    * - Operator
      - Code
      - Rows
      - Gives
    * - `\boldsymbol{B}_m`
      - ``Bm``, :meth:`.BFSCPlate2D.update_Bm`
      - `\boldsymbol{S}^u_{,x}`, `\boldsymbol{S}^v_{,y}`,
        `\boldsymbol{S}^u_{,y} + \boldsymbol{S}^v_{,x}`
      - linear membrane strains
    * - `\boldsymbol{B}_b`
      - ``Bb``, :meth:`.BFSCPlate2D.update_Bb`
      - `-\boldsymbol{S}^w_{,xx}`, `-\boldsymbol{S}^w_{,yy}`,
        `-2\boldsymbol{S}^w_{,xy}`
      - curvatures
    * - `\boldsymbol{G}`
      - ``G``
      - `\boldsymbol{S}^w_{,x}`, `\boldsymbol{S}^w_{,y}`
      - slopes `w_{,x}`, `w_{,y}` (``w_x``, ``w_y``)
    * - `\boldsymbol{B}_{NL}`
      - ``BmL``
      - `w_{,x}\boldsymbol{S}^w_{,x}`, `w_{,y}\boldsymbol{S}^w_{,y}`,
        `w_{,x}\boldsymbol{S}^w_{,y} + w_{,y}\boldsymbol{S}^w_{,x}`
      - nonlinear membrane strains

Since the nonlinear strains are quadratic in `\boldsymbol{u}_e`,
`\boldsymbol{B}_{NL}\boldsymbol{u}_e` is twice the nonlinear strains,
whereas their variation is `\boldsymbol{B}_{NL}\delta\boldsymbol{u}_e`,
hence the factors `1/2` and `1` of Eq. :eq:`el-strain-operators`.
`\boldsymbol{B}_b`, `\boldsymbol{G}` and `\boldsymbol{B}_{NL}` have nonzero
entries only in the 16 columns of `w`, `w_{,x}`, `w_{,y}`, `w_{,xy}`, and
`\boldsymbol{B}_m` only in the 24 columns of `u` to `v_{,y}`. In the
functions of the module the nonzero entries are local variables named after
the operator, the row and the column, both counted from 1: ``Bm1_01`` is the
entry of `\boldsymbol{B}_m` in the row 1 and the column 1 (`u` of node 1),
``Bb3_40`` the entry of `\boldsymbol{B}_b` in the row 3 and the column 40
(`w_{,xy}` of node 4).

The ``update_*`` functions of the module evaluate the operators themselves
and do not use the rows stored in the element. The methods of the element
serve the user, e.g. to integrate a distributed load with ``Sw`` or ``Su``,
see :doc:`assembling`, or to recover from a solution the linear membrane
strains ``Bm @ ue`` and the curvatures ``Bb @ ue`` at a point, with ``ue``
the 40 degrees of freedom of the element, and from them the membrane forces
and moments of Eq. :eq:`pt-abd`.


Continuity between elements
---------------------------

On the edge `\xi = \xi_i` of an element, `f_j(\xi_i)` is 1 for the two
nodes `j` of that edge and 0 for the other two, and `g_j(\xi_i) = 0` for
all nodes. Along the edge, Eq. :eq:`el-interpolation` reduces to

.. math::
    :label: el-edge

    w = \sum_{j \in \text{edge}} \left( f_j(\eta) w_j
        + \frac{ley}{2} g_j(\eta) w_{,y j} \right), \qquad
    w_{,x} = \sum_{j \in \text{edge}} \left( f_j(\eta) w_{,x j}
        + \frac{ley}{2} g_j(\eta) w_{,xy j} \right)

and likewise for an edge `\eta` = constant with `x` and `y` exchanged. `w`
along an edge, and its slope normal to the edge, depend only on the degrees
of freedom of the two nodes of the edge, shared by the two neighbouring
elements: `w`, `w_{,x}` and `w_{,y}` are continuous, and the element is
conforming for the classical plate theory. By the same argument `u`, `v`
and their first derivatives are continuous; along an edge `x` = constant,
`u_{,x}` is interpolated from the nodal `u_{,x}` with `f_j(\eta)` only,
since `u_{,xy}` is not a degree of freedom. The second derivatives, i.e. the
curvatures and the moments, are in general discontinuous between elements.

Eq. :eq:`el-edge` also shows which constraints make a displacement vanish
along a whole edge, see :doc:`assembling`: `w = 0` along an edge `x` =
constant needs `w = 0` **and** `w_{,y} = 0` at its nodes, and a zero normal
slope `w_{,x} = 0` needs `w_{,x} = 0` and `w_{,xy} = 0`.


Attributes of the element
-------------------------

Before the functions of the module are called, the attributes below are set
for each element, see :class:`.BFSCPlate2D` for their definitions:

- ``c1``, ``c2``, ``c3``, ``c4``: positions of the first degree of freedom
  of each node in the global vectors;
- ``lex``, ``ley``: lengths of the sides;
- ``ABD``: the ``(6, 6)`` constitutive matrix of Eq. :eq:`pt-abd`,
  constant over the element;
- ``h`` and ``rho``: thickness and density, read only by :func:`.update_M`;
- ``init_k_KC0``, ``init_k_KCNL``, ``init_k_KG``, ``init_k_M``,
  ``init_k_KCfollower``: positions of the element in the sparse arrays of
  each matrix, see :doc:`assembling`.

``n1``, ``n2``, ``n3`` and ``n4`` are identification numbers for the user,
not read by the functions. A laminate that varies over the plate is
represented by a different ``ABD`` in each element.

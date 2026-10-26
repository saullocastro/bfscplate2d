The element and its degrees of freedom
======================================

.. currentmodule:: bfscplate2d.bfscplate2d

:class:`.BFSCPlate2D` is a four-node rectangle of sides ``lex`` along `x`
and ``ley`` along `y`, parallel to the axes, of the classical laminated
plate theory: the deflection `w` and the in-plane displacements `u` and
`v` are interpolated with products of cubic Hermite polynomials, whose
nodal values are the degrees of freedom below. The kinematics, the
interpolation functions and the integration of each matrix are given in the
Notes of :mod:`bfscplate2d.bfscplate2d`.


Node ordering
-------------

The nodes are numbered counterclockwise from the corner with the smallest
`x` and `y`, as the CQUAD4 element of Nastran:

.. list-table::
    :header-rows: 1

    * - Node
      - Natural coordinates
      - Position
    * - 1
      - `\xi = -1, \eta = -1`
      - `(x_1, y_1)`
    * - 2
      - `\xi = +1, \eta = -1`
      - `(x_1 + lex, y_1)`
    * - 3
      - `\xi = +1, \eta = +1`
      - `(x_1 + lex, y_1 + ley)`
    * - 4
      - `\xi = -1, \eta = +1`
      - `(x_1, y_1 + ley)`

with `x = x_1 + lex (\xi + 1)/2` and `y = y_1 + ley (\eta + 1)/2`. The
element knows neither `x_1` nor `y_1`: its matrices depend on ``lex``,
``ley``, its properties and its degrees of freedom, not on its position.


Degrees of freedom of a node
----------------------------

Each node has ``DOF = 10`` degrees of freedom, at the positions ``c`` to
``c + 9`` of the global vectors and matrices, with ``c`` the attribute
``c1``, ``c2``, ``c3`` or ``c4`` of the element for the nodes 1 to 4:

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

`u`, `v` and `w` are the displacements of the reference surface along `x`,
`y` and `z`. The cross derivative `w_{,xy}` is a degree of freedom of `w`
only, as in the element of Bogner, Fox and Schmit; `u` and `v` have no cross
derivative. With ``c = DOF*i`` for the node of index ``i``, the usual choice
of the tests, the slice ``u[6::DOF]`` of a global vector gives `w` at all
the nodes, ``u[0::DOF]`` gives `u`, and so on, which is how the boundary
conditions and the loads are applied, see :doc:`assembling`.

The 40 degrees of freedom of the element, e.g. its attribute ``u``, are
those of the nodes 1, 2, 3 and 4 in sequence, 10 per node in the order of
the table.


Continuity between elements
---------------------------

Along an edge, `u`, `v`, `w` and their first derivatives depend only on the
degrees of freedom of the two nodes of that edge, which makes them
continuous between neighbouring elements: the element is conforming for
the classical plate theory. Along an edge `x` = constant, for instance, `w`
is interpolated from `w` and `w_{,y}` of the two nodes, and the normal
slope `w_{,x}` from `w_{,x}` and `w_{,xy}`. A displacement vanishes along a
whole edge only when both its value and its derivative along the edge are
constrained at the nodes of the edge, see :doc:`assembling`.


Properties of the element
-------------------------

Before the functions of the module are called, the attributes below are set
for each element, see :class:`.BFSCPlate2D` for their definitions:

- ``c1``, ``c2``, ``c3``, ``c4``: positions of the nodes in the global
  vectors;
- ``lex``, ``ley``: lengths of the sides;
- ``ABD``: the ``(6, 6)`` constitutive matrix of the laminate, e.g. the
  ``ABD`` attribute of :func:`composites.isotropic_plate` or
  :func:`composites.laminated_plate`, constant over the element;
- ``h`` and ``rho``: thickness and density, used only by the mass matrix of
  :func:`.update_M`;
- ``init_k_KC0``, ``init_k_KCNL``, ``init_k_KG``, ``init_k_M``,
  ``init_k_KCfollower``: positions of the element in the sparse arrays of
  each matrix, see :doc:`assembling`.

``n1``, ``n2``, ``n3`` and ``n4`` are identification numbers for the user,
not read by the functions. The methods :meth:`.BFSCPlate2D.update_Su` to
:meth:`.BFSCPlate2D.update_Sw_y`, :meth:`.BFSCPlate2D.update_Bm` and
:meth:`.BFSCPlate2D.update_Bb` evaluate the interpolation functions and the
strain operators at a point `(\xi, \eta)`, e.g. to integrate a load or to
recover the strains; the ``update_*`` functions of the module do not use
them.


Integration points
------------------

All the functions except :func:`.update_M` integrate over the element with
``nint`` by ``nint`` Gauss-Legendre points, the products of the
one-dimensional points and weights of
:func:`bfscplate2d.quadrature.get_points_weights`, given to each function.
The number of points that integrates each matrix exactly is given in the
Notes of :mod:`bfscplate2d.bfscplate2d`.

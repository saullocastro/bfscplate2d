Assembly, loads and boundary conditions
=======================================

.. currentmodule:: bfscplate2d.bfscplate2d

``bfscplate2d`` computes the contributions of one element. The mesh, the
global matrices and vectors, the loads and the boundary conditions belong to
the model, written by the user with NumPy and SciPy. This chapter states the
conventions that connect the element of :doc:`element` to a model.


Global degrees of freedom
-------------------------

A model with ``num_nodes`` nodes has ``N = DOF*num_nodes`` degrees of
freedom. The attributes ``c1`` to ``c4`` of each element give the position
of the first degree of freedom of its nodes in the global vectors; the usual
choice is ``c = DOF*i`` for the node of index ``i``, with which the slice
``u[6::DOF]`` of a global vector gives `w` at all the nodes, ``u[0::DOF]``
gives `u`, ``u[7::DOF]`` gives `w_{,x}`, and so on, in the order of the
table of :doc:`element`.

A rectangular plate of sides `a` and `b` is divided into rectangles of
sides ``lex`` and ``ley``; with ``nx`` by ``ny`` nodes on a regular grid,
``lex = a/(nx - 1)`` and ``ley = b/(ny - 1)``. Each element takes its four
nodes counterclockwise from the corner of smallest `x` and `y`, the order
of :doc:`element`; a node shared by several elements has the same ``c`` in
all of them, which makes the assembled fields continuous.


Sparse arrays
-------------

Each element matrix is written as triplets, the row index, the column index
and the value of each entry, in three 1-D arrays, the coordinate format of
``scipy.sparse.coo_matrix``. The functions write the entries of one element
at the positions ``init_k`` to ``init_k + SPARSE_SIZE - 1``, with ``init_k``
the attribute of the element for that matrix and ``SPARSE_SIZE`` the number
of entries per element. The sizes follow from the pairs of degrees of
freedom that each matrix couples, with 24 in-plane degrees of freedom (`u`
to `v_{,y}` of the four nodes) and 16 out-of-plane ones (`w` to `w_{,xy}`)
per element:

.. list-table::
    :header-rows: 1

    * - Function
      - Position attribute
      - Entries per element
      - Pairs written
    * - :func:`.update_KC0`
      - ``init_k_KC0``
      - ``KC0_SPARSE_SIZE = 1600``
      - all, `40^2`
    * - :func:`.update_KCNL`
      - ``init_k_KCNL``
      - ``KCNL_SPARSE_SIZE = 1024``
      - in-plane with out-of-plane in both orders and out-of-plane with
        out-of-plane, `2 \cdot 24 \cdot 16 + 16^2`
    * - :func:`.update_KG`, :func:`.update_KG_cte_N`
      - ``init_k_KG``
      - ``KG_SPARSE_SIZE = 256``
      - out-of-plane with out-of-plane, `16^2`
    * - :func:`.update_M`
      - ``init_k_M``
      - ``M_SPARSE_SIZE = 544``
      - among `u`, `u_{,x}`, `u_{,y}`, among `v`, `v_{,x}`, `v_{,y}` and
        among the out-of-plane ones, `2 \cdot 12^2 + 16^2`
    * - :func:`.update_KCfollower`
      - ``init_k_KCfollower``
      - ``KCfollower_SPARSE_SIZE = 1056``
      - the blocks `[u, v]`, `[v, u]` (`12^2` each) and `[u, w]`, `[v, w]`,
        `[w, u]`, `[w, v]` (`12 \cdot 16` each)

For ``num_elements`` elements the arrays have ``SPARSE_SIZE*num_elements``
entries, of type ``INT`` for the indices and ``DOUBLE`` for the values, and
the element of index ``k`` gets ``init_k = k*SPARSE_SIZE``. The global
matrix is the sum of the element matrices,
`\boldsymbol{K} = \sum_e \boldsymbol{L}_e^\top \boldsymbol{K}_e
\boldsymbol{L}_e`, with `\boldsymbol{L}_e` the Boolean matrix that extracts
`\boldsymbol{u}_e` from the global vector; ``coo_matrix`` performs this sum
by adding the values at repeated positions::

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

The functions write all the pairs, not only a triangle, such that the
symmetric matrices come out symmetric after the sum and the unsymmetric
load stiffness matrix of :doc:`follower_pressure` is complete. The indices
are overwritten, but the values are **added** to the existing ones: value
arrays reused for a new configuration, e.g. those of KCNL and KG at each
iteration of a nonlinear analysis, are zeroed first; the indices do not
change for a fixed mesh. ``INT``, ``DOUBLE``, ``DOF`` and the sizes are
given by the ``bfscplate2d`` namespace.

The vectors have no position attributes: :func:`.update_fint` and
:func:`.update_fext_follower` add the vector of one element to a global
vector of size ``N`` at the positions of its nodes, and every function that
depends on the displacements reads them from a global vector ``u`` of size
``N``.


Loads
-----

The external force vector is the virtual work of the loads,
`\delta W = \delta\boldsymbol{u}^\top \boldsymbol{F}_{ext}`. A concentrated
force or moment is a value at one degree of freedom of a node, e.g. a
transverse force at the position ``c + 6``; the conjugate of `w_{,x}` is a
moment, that of `w_{,xy}` a generalised force of dimension force times
length squared. Distributed loads give the
consistent load vectors, integrated with the rows of :doc:`element` and the
quadrature of :doc:`element_matrices`:

- a dead pressure `P`, `\boldsymbol{F}_{ext} = \int_A P
  (\boldsymbol{S}^w)^\top dA`, with :meth:`.BFSCPlate2D.update_Sw` at the
  integration points; this is also :func:`.update_fext_follower` at
  `\boldsymbol{u} = \boldsymbol{0}`;
- an in-plane force per unit length `\bar{N}_{xx}` on an edge `x` =
  constant, `\int \bar{N}_{xx} (\boldsymbol{S}^u)^\top dy`, with
  :meth:`.BFSCPlate2D.update_Su` at `\xi = \pm 1`;
- a follower pressure, :func:`.update_fext_follower`, see
  :doc:`follower_pressure`.

The consistent load of a uniform edge load on an edge `x` = constant has,
besides the forces at `u`, entries at `u_{,y}` of the two nodes at the ends
of the edge (those of the interior nodes cancel between neighbouring
elements). Nodal forces at `u` alone, e.g. with half the force at the
corner nodes, are not equivalent, and the membrane forces of the resulting
static solution are not exactly uniform, see :doc:`analyses`.


Boundary conditions
-------------------

A constrained degree of freedom is set to zero by partitioning the global
system into the unknown degrees of freedom ``bu`` and the constrained ones
``bk = ~bu``, two Boolean arrays of size ``N``:

.. math::
    :label: as-partition

    \boldsymbol{K}_{uu} \boldsymbol{u}_u = \boldsymbol{F}_u

with the matrices restricted to ``bu`` by ``bu`` and the vectors to ``bu``,
e.g. ``KC0[bu, :][:, bu]``. The solution is expanded to the size of the
model, with zeros at the constrained degrees of freedom, ``u[bu] = uu``,
before it is given to the functions that depend on the displacements.

Since a degree of freedom is a nodal value, the condition it imposes along
an edge follows from Eq. :eq:`el-edge`:

.. list-table::
    :header-rows: 1

    * - Condition along an edge
      - Edge `x` = constant (positions)
      - Edge `y` = constant (positions)
    * - `w = 0`
      - `w`, `w_{,y}` (6, 8)
      - `w`, `w_{,x}` (6, 7)
    * - normal slope zero, with `w = 0` a clamped edge
      - `w_{,x}`, `w_{,xy}` (7, 9)
      - `w_{,y}`, `w_{,xy}` (8, 9)
    * - `u = 0`
      - `u`, `u_{,y}` (0, 2)
      - `u`, `u_{,x}` (0, 1)
    * - `v = 0`
      - `v`, `v_{,y}` (3, 5)
      - `v`, `v_{,x}` (3, 4)

each value with its derivative along the edge. Constraining the value alone,
e.g. `w` at the nodes of a simply supported edge, makes it vanish at the
nodes only: between them it follows the free derivative along the edge,
and the condition is reached as the mesh is refined. The rigid-body motions
must be constrained, in-plane and out-of-plane, for ``KC0[bu, :][:, bu]`` to
be nonsingular. Prescribed nonzero displacements are a partition with
``u[bk]`` given and the right-hand side `\boldsymbol{F}_u -
\boldsymbol{K}_{uk}\boldsymbol{u}_k`.

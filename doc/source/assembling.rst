Assembling a model
==================

.. currentmodule:: bfscplate2d.bfscplate2d

``bfscplate2d`` computes the contributions of one element; the mesh, the
global matrices, the boundary conditions and the loads are written by the
user with NumPy and SciPy, as in the tests of the repository, see
:doc:`ex_linear_static`.


Mesh
----

A rectangular plate of sides `a` and `b` is divided into ``nx - 1`` by
``ny - 1`` elements of equal sides ``lex = a/(nx - 1)`` and ``ley = b/(ny -
1)``. The tests number the nodes with `x` varying slowest, ``nids_mesh =
nids.reshape(nx, ny)``, and take the four nodes of each element from the
corners of this array, in the order of :doc:`element`::

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

with ``nid_pos`` mapping each node identification number to its index,
such that ``c1 = DOF*nid_pos[n1]``, and so on, and the model has ``N =
DOF*nx*ny`` degrees of freedom.


Element loop and sparse arrays
------------------------------

Each function ``update_K*`` and :func:`.update_M` writes the entries of one
element in three 1-D arrays, the row indices, the column indices and the
values, at the positions ``init_k`` to ``init_k + SPARSE_SIZE - 1``, with
``init_k`` the attribute of the element for that matrix and ``SPARSE_SIZE``
the number of entries of one element:

.. list-table::
    :header-rows: 1

    * - Function
      - Position attribute
      - Entries per element
    * - :func:`.update_KC0`
      - ``init_k_KC0``
      - ``KC0_SPARSE_SIZE``
    * - :func:`.update_KCNL`
      - ``init_k_KCNL``
      - ``KCNL_SPARSE_SIZE``
    * - :func:`.update_KG`, :func:`.update_KG_cte_N`
      - ``init_k_KG``
      - ``KG_SPARSE_SIZE``
    * - :func:`.update_M`
      - ``init_k_M``
      - ``M_SPARSE_SIZE``
    * - :func:`.update_KCfollower`
      - ``init_k_KCfollower``
      - ``KCfollower_SPARSE_SIZE``

For ``num_elements`` elements the arrays have ``SPARSE_SIZE*num_elements``
entries, of type ``INT`` for the indices and ``DOUBLE`` for the values, and
the element of index ``k`` gets ``init_k = k*SPARSE_SIZE``, e.g. by
increasing a counter by ``SPARSE_SIZE`` in the element loop::

    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSCPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        ...
        plate.init_k_KC0 = init_k_KC0
        update_KC0(plate, points, weights, KC0r, KC0c, KC0v)
        init_k_KC0 += KC0_SPARSE_SIZE

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

``scipy.sparse.coo_matrix`` sums the values at repeated positions, the
entries of the nodes shared by several elements. ``INT``, ``DOUBLE``,
``DOF`` and the sizes are given by the ``bfscplate2d`` namespace. The
element writes all the pairs of degrees of freedom of each matrix, not only
an upper triangle; the matrices that are symmetric come out symmetric
after the sum.

The functions overwrite the indices but **add** the values to the existing
ones. Arrays reused for a new configuration, e.g. KCNL and KG at each
iteration of a nonlinear analysis, are zeroed first, ``KCNLv *= 0``; the
indices do not change for a fixed mesh.

The vectors are assembled without position attributes:
:func:`.update_fint` and :func:`.update_fext_follower` add the vector of
one element directly to a global vector of size ``N``, at the positions of
its nodes, and the functions that depend on the displacements read them
from a global vector ``u`` of size ``N``.


Loads
-----

A concentrated load is a value of the global force vector at a degree of
freedom of a node, e.g. ``f[6::DOF][check] = fmid`` for a transverse force
at the nodes selected by the boolean array ``check``. Distributed loads are
integrated with the interpolation functions of the element, the
consistent load vector: a pressure `P` with the rows of
:meth:`.BFSCPlate2D.update_Nw` over the integration points of the element,
see :doc:`ex_linear_static`, and a force per unit length on an edge with
:meth:`.BFSCPlate2D.update_Nu` at `\xi = \pm 1`, see
:doc:`ex_linear_buckling`. The force vector of a pressure that follows the
deformation is given by :func:`.update_fext_follower`, see
:doc:`follower_pressure`.


Boundary conditions
-------------------

The tests constrain degrees of freedom by partitioning the global system: a
boolean array ``bk`` marks the constrained degrees of freedom, selected with
the slices of :doc:`element` and the coordinates of the nodes, and ``bu =
~bk`` the unknown ones::

    bk = np.zeros(N, dtype=bool)
    check = isclose(x, 0) | isclose(x, a) | isclose(y, 0) | isclose(y, b)
    bk[6::DOF] = check            # w = 0 at the nodes of the four edges
    bk[0::DOF] = isclose(x, 0)    # u = 0 at x = 0
    bk[3::DOF] = isclose(y, 0)    # v = 0 at y = 0
    bu = ~bk

    Kuu = KC0[bu, :][:, bu]
    u = np.zeros(N)
    u[bu] = spsolve(Kuu, f[bu])

The solution of the reduced system is expanded to the size of the model,
with zeros at the constrained degrees of freedom, before it is given to the
functions that depend on the displacements, e.g. :func:`.update_KG`.

Since `w` along an edge is interpolated from `w` and its derivative along
the edge at the nodes, see :doc:`element`, `w` vanishes along the whole
edge `x` = constant when `w` and `w_{,y}` are constrained at its nodes,
positions ``6`` and ``8``, and along an edge `y` = constant with `w` and
`w_{,x}`, positions ``6`` and ``7``. A clamped edge `x` = constant also
constrains the normal slope `w_{,x}` and its derivative along the edge
`w_{,xy}`, positions ``7`` and ``9``. The function ``boundary_flags`` of
``tests/test_follower_pressure.py`` builds ``bk`` in this way, for the
value and the derivative along the edge of each constrained displacement,
see :doc:`ex_follower_pressure`.

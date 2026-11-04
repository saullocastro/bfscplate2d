r"""Linear static analysis of a plate under a central point load

A rectangular isotropic plate of ``a = 3`` by ``b = 7``, thickness
``h = 0.005``, on a mesh of 29 by 29 nodes, simply supported, with `w`
and its derivative along the edge constrained at the nodes of the four
edges, and `u` and `v` at every node, under a unit
transverse force at the central node. The stiffness matrix is assembled
with :func:`bfscplate2d.bfscplate2d.update_KC0`. The test checks that the
force vector sums to the applied force, and the deflection at the centre
against the series solution of Timoshenko and Woinowsky-Krieger (1959),
Eq. (147), with a relative tolerance of ``1e-3``.

Units: N, m, Pa.

Notes
-----
The deflection at the centre of the simply supported rectangular plate of
sides `a` and `b \ge a` under a load `P` at its centre is Eq. (147) of [1]_

.. math::

    w_{max} = \frac{P a^2}{2 \pi^3 D} \sum_{m = 1, 3, ...} \frac{1}{m^3}
        \left( \tanh \alpha_m - \frac{\alpha_m}{\cosh^2 \alpha_m} \right)
        = \alpha \frac{P a^2}{D}, \qquad
    \alpha_m = \frac{m \pi b}{2 a}, \qquad
    D = \frac{E h^3}{12 (1 - \nu^2)}

For `b/a = 7/3` the series gives `\alpha = 0.016785`, with the terms up to
`m = 1001`, between the values 0.01651 for `b/a = 2` and 0.01690 for `b/a =
3` of Table 23 of [1]_; for the ratios of the table the series agrees with
it within 0.2 %. For the plate of the test, `w_{max} = 6.5986 \times
10^{-5}`.

Along an edge `x` = constant the interpolated `w` depends only on `w` and
`w_{,y}` at the two nodes of the edge, so `w = 0` on the whole edge needs
both nodal values constrained (`w` and `w_{,x}` on an edge `y` =
constant). The deflection under the point load converges from below with
the mesh, with relative errors of -0.92 %, -0.23 % and -0.057 % with 5 by
9, 9 by 17 and 17 by 33 nodes, -0.056 % with the 29 by 29 nodes of the
test and -0.006 % with 49 by 97 nodes, which justifies the tolerance. With
`w` alone constrained at the nodes of the edges the errors differ by less
than 0.07 % of the deflection.

References
----------
.. [1] Timoshenko, S., and Woinowsky-Krieger, S., Theory of Plates and
   Shells, 2nd ed., McGraw-Hill, New York, 1959.

"""
import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from composites import isotropic_plate

from bfscplate2d import (BFSCPlate2D, update_KC0, DOF, KC0_SPARSE_SIZE, DOUBLE,
        INT)
from bfscplate2d.quadrature import get_points_weights

def test_static_point_load(plot=False):
    # number of nodes
    nx = 29
    ny = 29
    points, weights = get_points_weights(nint=4)

    # geometry
    a = 3
    b = 7

    # material properties
    E = 200e9
    nu = 0.3
    h = 0.005
    lam = isotropic_plate(thickness=h, E=E, nu=nu)

    xtmp = np.linspace(0, a, nx)
    ytmp = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xtmp, ytmp)

    # getting nodes
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
    x = ncoords[:, 0]
    y = ncoords[:, 1]
    nids = 1 + np.arange(ncoords.shape[0])
    nid_pos = dict(zip(nids, np.arange(len(nids))))

    nids_mesh = nids.reshape(nx, ny)

    n1s = nids_mesh[:-1, :-1].flatten()
    n2s = nids_mesh[1:, :-1].flatten()
    n3s = nids_mesh[1:, 1:].flatten()
    n4s = nids_mesh[:-1, 1:].flatten()

    num_elements = len(n1s)
    print('num_elements', num_elements)

    N = DOF*nx*ny
    Kr = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kc = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    Kv = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0

    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSCPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        plate.c2 = DOF*nid_pos[n2]
        plate.c3 = DOF*nid_pos[n3]
        plate.c4 = DOF*nid_pos[n4]
        plate.ABD = lam.ABD
        plate.lex = a/(nx - 1)
        plate.ley = b/(ny - 1)
        plate.init_k_KC0 = init_k_KC0
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        init_k_KC0 += KC0_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    bk = np.zeros(KC0.shape[0], dtype=bool)

    # simply supported: w and its derivative along the edge
    edges_x = isclose(x, 0) | isclose(x, a)
    edges_y = isclose(y, 0) | isclose(y, b)
    bk[6::DOF] = edges_x | edges_y
    bk[8::DOF] = edges_x
    bk[7::DOF] = edges_y

    # eliminating all u,v displacements
    bk[0::DOF] = True
    bk[3::DOF] = True

    bu = ~bk # same as np.logical_not, defining unknown DOFs

    # external force vector for point load at center
    f = np.zeros(KC0.shape[0])
    fmid = 1.

    # force at center node
    check = np.isclose(x, a/2) & np.isclose(y, b/2)
    f[6::DOF][check] = fmid
    assert f.sum() == fmid

    # sub-matrices corresponding to unknown DOFs
    Kuu = KC0[bu, :][:, bu]
    fu = f[bu]

    # solving
    uu = spsolve(Kuu, fu)
    u = np.zeros(KC0.shape[0], dtype=float)
    u[bu] = uu

    w = u[6::DOF].reshape(nx, ny).T
    print('wmax', w.max())
    print('wmin', w.min())
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    # reference deflection of the Notes, b/a = 7/3
    D = E*h**3/(12*(1 - nu**2))
    w_ref = 0.016785*fmid*a**2/D
    assert isclose(w.max(), u[6::DOF][check][0])
    assert isclose(w.max(), w_ref, rtol=1e-3)

if __name__ == '__main__':
    test_static_point_load(plot=True)

r"""Linear static analysis of a plate under a uniform pressure

A rectangular isotropic plate of ``a = 3`` by ``b = 7``, thickness
``h = 0.005``, simply supported, with `w` and its derivative along the
edge constrained at the nodes of the four edges, and `u` and `v` at the
central node, under the pressure ``P = 10``, on a mesh of 5 by 7 nodes. The
stiffness matrix is assembled with
:func:`bfscplate2d.bfscplate2d.update_KC0` and the consistent force vector
of the pressure is integrated element by element with the rows of
:meth:`.BFSCPlate2D.update_Sw`. The maximum deflection is checked against
the Navier solution of Timoshenko and Woinowsky-Krieger (1959), with a
relative tolerance of ``1e-4``.

Units: N, m, Pa.

Notes
-----
The consistent force vector of the pressure `P` is

.. math::

    \boldsymbol{F}_{ext} = \int_A P \, (\boldsymbol{S}^w)^\top dA

computed element by element with ``nint = 4`` points along `\xi` and
`\eta`, exact for the bicubic rows `\boldsymbol{S}^w`. Along an edge `x` =
constant the interpolated `w` depends only on `w` and `w_{,y}` at the two
nodes of the edge, so `w = 0` on the whole edge needs both nodal values
constrained (`w` and `w_{,x}` on an edge `y` = constant); the normal slope
and `w_{,xy}` stay free, which gives the simply supported plate. The
reference value is the Navier double-series solution of the simply
supported rectangular plate under a uniform pressure [1]_,

.. math::

    w_{max} = \frac{16 P}{\pi^6 D} \sum_{m = 1, 3, ...}
        \sum_{n = 1, 3, ...}
        \frac{\sin(m \pi/2) \sin(n \pi/2)}
        {m n \left( m^2/a^2 + n^2/b^2 \right)^2}, \qquad
    D = \frac{E h^3}{12 (1 - \nu^2)}

which gives `w_{max} = 3.937198 \times 10^{-3}` with the terms up to
`m = n = 199`. The mesh of the test is 0.0024 % above it, and 0.0001 % with
9 by 13 nodes. With `w` alone constrained at the nodes of the edges, `w`
between the nodes of the edges reaches 1.9 % of `w_{max}` and `w_{max}`
is 0.089 % above the Navier solution with the mesh of the test, 0.0025 %
with 9 by 13 nodes.

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

def test_static_pressure(plot=False):
    # number of nodes
    nx = 5
    ny = 7
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
    # point supports (middle)
    check = isclose(x, a/2) & isclose(y, b/2)
    bk[0::DOF] = check
    bk[3::DOF] = check

    bu = ~bk

    # external force vector applying consistent pressure
    f = np.zeros(KC0.shape[0], dtype=float)
    P = 10 # Pa
    nint = 4
    points, weights = get_points_weights(nint)
    for plate in plates:
        lex = plate.lex
        ley = plate.ley
        indices = []
        c1 = plate.c1
        c2 = plate.c2
        c3 = plate.c3
        c4 = plate.c4
        cs = [c1, c2, c3, c4]
        for ci in cs:
            for i in range(DOF):
                indices.append(ci + i)
        fe = np.zeros(4*DOF, dtype=float)
        for i in range(nint):
            xi = points[i]
            weight_xi = weights[i]
            for j in range(nint):
                eta = points[j]
                weight_eta = weights[j]
                weight = weight_xi*weight_eta
                plate.update_Sw(xi, eta)
                Sw = np.asarray(plate.Sw)
                fe += (lex*ley)/4*weight*Sw*P
        f[indices] += fe

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
    # Navier solution of the Notes
    w_ref = 3.937198e-3
    assert np.isclose(w.max(), w_ref, rtol=1e-4)

if __name__ == '__main__':
    test_static_pressure(plot=True)

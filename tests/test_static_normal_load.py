r"""Linear static analysis of a plate with a built-in and a free edge

A square isotropic plate of ``a = b = 0.5``, thickness ``h = 0.001``, on a
mesh of 7 by 5 nodes, built in at `x = 0`, free at `x = a` and simply
supported at `y = 0` and `y = b`, under the uniform pressure ``q = 1``. The
stiffness matrix is assembled with :func:`bfscplate2d.bfscplate2d.update_KC0`
and the consistent force vector of the pressure is integrated element by
element with the rows of :meth:`.BFSCPlate2D.update_Sw`. The built-in edge
has `u`, `v`, `w`, `w_{,x}`, `w_{,y}` and `w_{,xy}` constrained at its
nodes, the simply supported edges `w` and `w_{,x}`. The maximum deflection,
at the middle of the free edge, is checked against the Lévy solution of
Timoshenko and Woinowsky-Krieger (1959), Art. 46, Table 39, with a relative
tolerance of ``1e-3``.

Units: N, m, Pa.

Notes
-----
The reference value is the deflection of the uniformly loaded rectangular
plate with two opposite edges simply supported, the third edge free and the
fourth built in, of Art. 46 of [1]_. In the coordinates of the test, with
`b` the span between the simply supported edges, `\beta_m = m \pi a/b` and
`\bar{x} = m \pi x/b`, the Lévy series of Eqs. (d) to (f) of [1]_ is

.. math::

    w = \frac{q b^4}{D} \sum_{m = 1, 3, ...} \left( \frac{4}{\pi^5 m^5}
        + A_m \cosh \bar{x} + B_m \bar{x} \sinh \bar{x}
        + C_m \sinh \bar{x} + D_m \bar{x} \cosh \bar{x} \right)
        \sin \frac{m \pi y}{b}, \qquad
    D = \frac{E h^3}{12 (1 - \nu^2)}

with the constants of Eqs. (g) and (h) of [1]_, which satisfy `w =
w_{,x} = 0` at `x = 0` and the zero bending moment and Kirchhoff shear
force at `x = a`. At the middle of the free edge, `x = a`, `y = b/2`, for
`a/b = 1` and `\nu = 0.3`, the series gives

.. math::

    w_{max} = 0.0112359 \frac{q b^4}{D}

converged to 7 digits with the terms up to `m = 31`. Table 39 of [1]_
lists `0.0113 q b^4/D`, the first term of the series, `0.011295`, rounded;
the converged value, 0.6 % lower, is the reference.

Along an edge the element interpolates `w` from the nodal `w` and the
nodal derivative along the edge, and the normal slope from the nodal
normal slope and `w_{,xy}`: `w = 0` along a whole edge needs both nodal
values constrained, and the zero normal slope of the built-in edge
`w_{,x} = 0` and `w_{,xy} = 0`. With these constraints the
deflection converges with the mesh to the series: relative errors of
-0.07 %, -0.01 % and +0.014 % with 3 by 3, 5 by 5 and the 7 by 5 nodes of
the test, and below `10^{-6}` with 25 by 25 nodes, which justifies the
tolerance. Constraining only the nodal values of `w` on the simply
supported edges gives +0.6 % with 5 by 5 nodes.

References
----------
.. [1] Timoshenko, S., and Woinowsky-Krieger, S., Theory of Plates and
   Shells, 2nd ed., McGraw-Hill, New York, 1959.

"""
import sys
sys.path.append('..')

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from composites import isotropic_plate

from bfscplate2d import (BFSCPlate2D, update_KC0, DOF, KC0_SPARSE_SIZE, DOUBLE,
        INT)
from bfscplate2d.quadrature import get_points_weights

def test_static(plot=False):
    # number of nodes
    nx = 7
    ny = 5
    points, weights = get_points_weights(nint=4)

    # geometry
    a = 0.5
    b = 0.5

    # material properties
    E = 0.7e11
    nu = 0.3
    h = 0.001
    lam = isotropic_plate(thickness=h, E=E, nu=nu)

    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    x, y = np.meshgrid(x, y)

    # getting nodes
    ncoords = np.vstack((x.T.flatten(), y.T.flatten())).T
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
        plate.n1 = n1
        plate.n2 = n2
        plate.n3 = n3
        plate.n4 = n4
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
    # built in at x = 0, simply supported at y = 0 and y = b, free at x = a
    clamped = np.isclose(ncoords[:, 0], 0.)
    supported = np.isclose(ncoords[:, 1], 0.) | np.isclose(ncoords[:, 1], b)
    print('boundary conditions', clamped.sum(), supported.sum())
    bk = np.zeros(KC0.shape[0], dtype=bool)
    bk[0::DOF] = clamped
    bk[3::DOF] = clamped
    bk[6::DOF] = clamped | supported
    bk[7::DOF] = clamped | supported
    bk[8::DOF] = clamped
    bk[9::DOF] = clamped

    bu = ~bk
    print('boundary conditions bu',  bu.sum())

    # consistent force vector of the uniform pressure
    q = 1. # Pa
    f = np.zeros(KC0.shape[0], dtype=float)
    for plate in plates:
        indices = []
        for ci in [plate.c1, plate.c2, plate.c3, plate.c4]:
            for i in range(DOF):
                indices.append(ci + i)
        fe = np.zeros(4*DOF, dtype=float)
        for i in range(len(points)):
            for j in range(len(points)):
                weight = weights[i]*weights[j]
                plate.update_Sw(points[i], points[j])
                Sw = np.asarray(plate.Sw)
                fe += (plate.lex*plate.ley)/4*weight*Sw*q
        f[indices] += fe

    Kuu = KC0[bu, :][:, bu]
    fu = f[bu]

    # solving
    uu = spsolve(Kuu, fu)
    u = np.zeros(KC0.shape[0], dtype=float)
    u[bu] = uu

    if False:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        for plate in plates:
            pos1 = nid_pos[plate.n1]
            pos2 = nid_pos[plate.n2]
            pos3 = nid_pos[plate.n3]
            pos4 = nid_pos[plate.n4]
            x1, y1 = ncoords[pos1]
            x2, y2 = ncoords[pos2]
            x3, y3 = ncoords[pos3]
            x4, y4 = ncoords[pos4]
            plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], '-')
        plt.show()

    w = u[6::DOF].reshape(nx, ny).T
    print('w min max', w.min(), w.max())
    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(x, y, w, levels=levels)
        plt.colorbar()
        plt.show()

    # reference deflection of the Notes, a/b = 1, nu = 0.3
    D = E*h**3/(12*(1 - nu**2))
    w_ref = 0.0112359*q*b**4/D
    check = np.isclose(ncoords[:, 0], a) & np.isclose(ncoords[:, 1], b/2)
    assert np.isclose(u[6::DOF][check][0], w.max())
    assert np.isclose(w.max(), w_ref, rtol=1e-3)

if __name__ == '__main__':
    test_static(plot=True)

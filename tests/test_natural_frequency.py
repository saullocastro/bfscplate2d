r"""Natural frequencies of a simply supported plate

A rectangular isotropic plate of ``a = 0.6`` by ``b = 0.2``, thickness
``h = 0.001``, density ``rho = 7.8e3``, on a mesh of 9 by 7 nodes, simply
supported, with `w` and its derivative along the edge constrained at the
nodes of the four edges, `u` at `x = 0` and `v` at `y = 0`. The stiffness and mass matrices are assembled with
:func:`bfscplate2d.bfscplate2d.update_KC0` and
:func:`bfscplate2d.bfscplate2d.update_M`, and the first four natural
angular frequencies are computed with the dense ``scipy.linalg.eigh``. The
first one is checked against the closed-form natural angular frequency of
the mode `m = n = 1` of the simply supported plate (Leissa, 1969), within
0.1 %, and the four against those of ``structsolve.freq``, within ``1e-6``.

Units: N, m, kg, Pa, rad/s.

Notes
-----
The generalized eigenvalue problem is

.. math::

    \boldsymbol{K}_{C0} \boldsymbol{\phi} = \omega_n^2 \boldsymbol{M}
    \boldsymbol{\phi}

For the simply supported plate the mode `w = \sin(m \pi x/a) \sin(n \pi
y/b)` satisfies the equation of motion `D \nabla^4 w = \rho h \omega^2 w`
of the classical plate theory, with the rotary inertia neglected [1]_, for

.. math::

    \omega_{mn} = \pi^2 \left( \frac{m^2}{a^2} + \frac{n^2}{b^2} \right)
        \sqrt{\frac{D}{\rho h}}, \qquad
    D = \frac{E h^3}{12 (1 - \nu^2)}

The expression of the test, written with the bending stiffness of a
plate of thickness `2 h` and the mass per unit area `2 \rho h`, is the same
value, `\omega_{11} = 248.535` rad/s. ``structsolve.freq`` returns the
eigenvalues `-\omega_n^2`.

Along an edge `x` = constant the interpolated `w` depends only on `w` and
`w_{,y}` at the two nodes of the edge, so `w = 0` on the whole edge needs
both nodal values constrained (`w` and `w_{,x}` on an edge `y` =
constant). With the mesh of the test the first four frequencies, the modes
`m = 1` to 4 and `n = 1`, are 0.003 %, 0.004 %, 0.033 % and 0.17 % above
`\omega_{mn}`. With `w` alone constrained at the nodes of the edges they
are 0.040 %, 0.21 %, 0.52 % and 0.89 % below it.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969.

"""
import sys
sys.path.append('..')

from scipy.sparse import coo_matrix
from scipy.linalg import eigh
import numpy as np
from composites import isotropic_plate
from structsolve import freq

from bfscplate2d import (BFSCPlate2D, update_KC0, update_M, DOF, KC0_SPARSE_SIZE,
        M_SPARSE_SIZE, DOUBLE, INT)
from bfscplate2d.quadrature import get_points_weights

def test_nat_freq(plot_mode=None):
    # number of nodes
    nx = 9 # along x
    ny = 7 # along y

    points, weights = get_points_weights(nint=4)

    # geometry
    a = 0.6
    b = 0.2

    # material properties (Aluminum)
    E = 70e9
    nu = 0.3
    rho = 7.8e3
    h = 0.001
    lam = isotropic_plate(thickness=h, E=E, nu=nu, rho=rho)

    # creating mesh
    x = np.linspace(0, a, nx)
    y = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(x, y)

    # node coordinates and position in the global matrix
    ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
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
    Mr = np.zeros(M_SPARSE_SIZE*num_elements, dtype=INT)
    Mc = np.zeros(M_SPARSE_SIZE*num_elements, dtype=INT)
    Mv = np.zeros(M_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_M = 0

    plates = []
    for n1, n2, n3, n4 in zip(n1s, n2s, n3s, n4s):
        plate = BFSCPlate2D()
        plate.c1 = DOF*nid_pos[n1]
        plate.c2 = DOF*nid_pos[n2]
        plate.c3 = DOF*nid_pos[n3]
        plate.c4 = DOF*nid_pos[n4]
        plate.ABD = lam.ABD
        plate.h = h
        plate.rho = rho
        plate.lex = a/(nx - 1)
        plate.ley = b/(ny - 1)
        plate.init_k_KC0 = init_k_KC0
        plate.init_k_M = init_k_M
        update_KC0(plate, points, weights, Kr, Kc, Kv)
        update_M(plate, Mr, Mc, Mv)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_M += M_SPARSE_SIZE
        plates.append(plate)

    KC0 = coo_matrix((Kv, (Kr, Kc)), shape=(N, N)).tocsc()
    M = coo_matrix((Mv, (Mr, Mc)), shape=(N, N)).tocsc()

    # applying boundary conditions
    # simply supported

    # locating nodes
    bk = np.zeros(KC0.shape[0], dtype=bool) # constrained DOFs, can be used to prescribe displacements

    x = ncoords[:, 0]
    y = ncoords[:, 1]

    # simply supported: w and its derivative along the edge
    edges_x = np.isclose(x, 0.) | np.isclose(x, a)
    edges_y = np.isclose(y, 0.) | np.isclose(y, b)
    bk[6::DOF] = edges_x | edges_y
    bk[8::DOF] = edges_x
    bk[7::DOF] = edges_y
    # constraining u at x = 0
    check = np.isclose(x, 0.)
    bk[0::DOF] = check
    # constraining v at y = 0
    check = np.isclose(y, 0.)
    bk[3::DOF] = check

    # unconstrained nodes
    bu = ~bk # logical_not

    Kuu = KC0[bu, :][:, bu]
    Muu = M[bu, :][:, bu]

    # solving generalized eigenvalue problem
    num_eigenvalues = 4
    print('eig solver begin')
    eigvals, eigvecsu = eigh(a=Kuu.toarray(), b=Muu.toarray(), subset_by_index=(0,
        num_eigenvalues-1))
    print('eig solver end')
    eigvecs = np.zeros((KC0.shape[0], num_eigenvalues), dtype=float)
    eigvecs[bu, :] = eigvecsu
    omegan = eigvals**0.5

    # theoretical reference
    m = 1
    n = 1
    D = 2*h**3*E/(3*(1 - nu**2))
    wmn = (m**2/a**2 + n**2/b**2)*np.sqrt(D*np.pi**4/(2*rho*h))/2
    print('Theoretical omega123', wmn)
    print(omegan)
    assert np.isclose(omegan[0], wmn, rtol=1e-3)

    # the same natural frequencies with the sparse solver of structsolve,
    # which returns lambda2 = -omegan**2
    lambda2, _ = freq(Kuu, Muu, num_eigvalues=num_eigenvalues,
                      symmetric=True, silent=True)
    assert np.allclose((-lambda2)**0.5, omegan, rtol=1e-6)

    if plot_mode is not None:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        wplot = eigvecs[6::DOF, plot_mode].reshape(nx, ny).T
        levels = np.linspace(wplot.min(), wplot.max(), 300)
        plt.contourf(xmesh, ymesh, wplot, levels=levels)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    test_nat_freq(plot_mode=0)

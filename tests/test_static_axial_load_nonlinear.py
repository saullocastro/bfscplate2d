import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, spsolve
from composites import isotropic_plate

from bfscplate2d import (BFSCPlate2D, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfscplate2d.quadrature import get_points_weights

def test_static(plot=False):
    # number of nodes
    nx = 13
    ny = 7
    points, weights = get_points_weights(nint=4)

    load = 300. # N

    # geometry
    a = 0.9
    b = 0.5

    # material properties
    E = 0.7e11
    nu = 0.3
    h = 0.001
    lam = isotropic_plate(thickness=h, E=E, nu=nu)

    xlin = np.linspace(0, a, nx)
    ylin = np.linspace(0, b, ny)
    xmesh, ymesh = np.meshgrid(xlin, ylin)

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
    KC0r = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0c = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=INT)
    KC0v = np.zeros(KC0_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
    KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
    KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
    init_k_KC0 = 0
    init_k_KCNL = 0
    init_k_KG = 0

    elements = []
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
        plate.init_k_KCNL = init_k_KCNL
        plate.init_k_KG = init_k_KG
        update_KC0(plate, points, weights, KC0r, KC0c, KC0v)
        init_k_KC0 += KC0_SPARSE_SIZE
        init_k_KCNL += KCNL_SPARSE_SIZE
        init_k_KG += KG_SPARSE_SIZE
        elements.append(plate)

    KC0 = coo_matrix((KC0v, (KC0r, KC0c)), shape=(N, N)).tocsc()

    # applying boundary conditions
    # simply supported
    bk = np.zeros(N, dtype=bool)
    checkSS = isclose(x, 0) | isclose(x, a)
    bk[3::DOF] = checkSS
    bk[6::DOF] = checkSS
    checkSS = isclose(y, 0) | isclose(y, b)
    bk[6::DOF] += checkSS
    check = isclose(x, a/2.)
    bk[0::DOF] = check
    bu = ~bk
    print('boundary conditions bu',  bu.sum())

    checkTopEdge = isclose(x, a)
    checkBottomEdge = isclose(x, 0)
    fext = np.zeros(N)
    fext[0::DOF][checkBottomEdge] = +load/ny
    assert isclose(fext.sum(), load)
    fext[0::DOF][checkTopEdge] = -load/ny
    assert isclose(fext.sum(), 0)
    check = isclose(x, a/2) & isclose(y, b/2)
    assert check.sum() == 1
    fext[6::DOF][check] = 0.01

    KC0uu = KC0[bu, :][:, bu]
    fu = fext[bu]

    # solving
    def calc_KT(u, KCNLv, KGv):
        KCNLv *= 0
        KGv *= 0
        for plate in elements:
            update_KCNL(u, plate, points, weights, KCNLr, KCNLc, KCNLv)
            update_KG(u, plate, points, weights, KGr, KGc, KGv)
        KCNL = coo_matrix((KCNLv, (KCNLr, KCNLc)), shape=(N, N)).tocsc()
        KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
        return KC0 + KCNL + KG

    def calc_fint(u, fint):
        fint *= 0
        for plate in elements:
            update_fint(u, plate, points, weights, fint)
        return fint

    # solving using Modified Newton-Raphson method
    def scaling(vec, D):
        """
            A. Peano and R. Riccioni, Automated discretisatton error
            control in finite element analysis. In Finite Elements m
            the Commercial Enviror&ent (Editei by J. 26.  Robinson),
            pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
        """
        return np.sqrt((vec*np.abs(1/D))@vec)

    #initial
    u0 = np.zeros(N) # any initial condition here

    u0[bu] = spsolve(KC0uu, fext[bu])

    # solving eigenvalue problem for linear buckling
    KGv *= 0
    for plate in elements:
        update_KG(u0, plate, points, weights, KGr, KGc, KGv)
    KG = coo_matrix((KGv, (KGr, KGc)), shape=(N, N)).tocsc()
    KGuu = KG[bu, :][:, bu]
    num_eigenvalues = 5
    eigvals, eigvecsu = eigsh(A=KGuu, k=num_eigenvalues, which='SM', M=KC0uu,
            tol=1e-6, sigma=1., mode='cayley')
    eigvals = -1./eigvals
    print('load_cr', eigvals[0]*load)

    #PREC = 1/KC0uu.diagonal().max()
    #u0[bu], info = cg(PREC*KC0uu, PREC*fext[bu], atol=1e-9)
    #if info != 0:
        #print('#   failed with cg()')
        #print('#   trying spsolve()')
        #uu = spsolve(KC0uu, fext[bu])
    count = 0
    fint = np.zeros(N)
    fint = calc_fint(u0, fint)
    Ri = fint - fext
    du = np.zeros(N)
    ui = u0.copy()
    epsilon = 1.e-4
    KT = calc_KT(u0, KCNLv, KGv)
    KTuu = KT[bu, :][:, bu]
    D = KC0uu.diagonal() # at beginning of load increment
    while True:
        print('count', count)
        duu = spsolve(KTuu, -Ri[bu])
        #PREC = 1/KTuu.diagonal().max()
        #duu, info = cg(PREC*KTuu, -PREC*Ri[bu], atol=1e-9)
        #if info != 0:
            #print('#   failed with cg()')
            #print('#   trying spsolve()')
            #duu = spsolve(KTuu, -Ri[bu])
        du[bu] = duu
        u = ui + du
        fint = calc_fint(u, fint)
        Ri = fint - fext
        crisfield_test = scaling(Ri[bu], D)/max(scaling(fext[bu], D), scaling(fint[bu], D))
        print('    crisfield_test', crisfield_test)
        if crisfield_test < epsilon:
            print('    converged')
            break
        count += 1
        KT = calc_KT(u, KCNLv, KGv)
        KTuu = KT[bu, :][:, bu]
        ui = u.copy()
        if count > 40:
            raise RuntimeError('Not converged!')

    if False:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        for plate in elements:
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
        plt.contourf(xmesh, ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    assert isclose(w.max(), 6.556443293916061e-05, rtol=1e-3)

if __name__ == '__main__':
    test_static(plot=True)

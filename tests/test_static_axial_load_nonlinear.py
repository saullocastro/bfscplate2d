"""Nonlinear static analysis of an axially compressed plate

The plate is simply supported, compressed along x and given a small
transverse load at the centre, so that it deflects out of plane instead of
sitting on the trivial flat branch.

Two regimes are checked:

``test_static``
    a transverse load small enough that the response is still the linearised
    one, so it can be compared against ``(KC0 + KG(P))**-1 f``, the exact
    answer of the eigenvalue-free buckling amplification problem. That pins
    the result to physics rather than to a recorded number. The old
    reference value for this case was 6.56e-5, an amplification of 10.4 over
    the transverse-load-only response, where ``1/(1 - P/Pcr) = 2.49`` is the
    theoretical ceiling; it was an artefact of an internal force vector that
    counted its geometric term twice.

``test_large_deflection``
    a transverse load large enough to reach w/h of about 2.6, where the
    membrane stretching of the von Karman terms actually carries the load.
    Here the plate must come out *stiffer* than the linearised prediction.

Both also assert that Newton-Raphson converges quadratically, which only
happens when KC0 + KCNL(u) + KG(u) is the exact derivative of fint. See
test_tangent_consistency.py, which checks that property directly.
"""
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

# number of nodes
nx = 13
ny = 7

# geometry
a = 0.9
b = 0.5

# material properties
E = 0.7e11
nu = 0.3
h = 0.001


class Model(object):
    """Mesh, elements, boundary conditions and the operators built on them."""


def build_model():
    points, weights = get_points_weights(nint=4)
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
    print('boundary conditions bu', bu.sum())

    m = Model()
    m.points, m.weights, m.lam = points, weights, lam
    m.ncoords, m.x, m.y = ncoords, x, y
    m.xmesh, m.ymesh = xmesh, ymesh
    m.nid_pos = nid_pos
    m.elements = elements
    m.N = N
    m.KC0 = KC0
    m.KC0uu = KC0[bu, :][:, bu]
    m.bk, m.bu = bk, bu
    m.KCNLr, m.KCNLc, m.KCNLv = KCNLr, KCNLc, KCNLv
    m.KGr, m.KGc, m.KGv = KGr, KGc, KGv
    return m


def calc_KT(m, u):
    """The tangent stiffness, KC0 + KCNL(u) + KG(u)."""
    m.KCNLv *= 0
    m.KGv *= 0
    for plate in m.elements:
        update_KCNL(u, plate, m.points, m.weights, m.KCNLr, m.KCNLc, m.KCNLv)
        update_KG(u, plate, m.points, m.weights, m.KGr, m.KGc, m.KGv)
    KCNL = coo_matrix((m.KCNLv, (m.KCNLr, m.KCNLc)),
                      shape=(m.N, m.N)).tocsc()
    KG = coo_matrix((m.KGv, (m.KGr, m.KGc)), shape=(m.N, m.N)).tocsc()
    return m.KC0 + KCNL + KG


def calc_KG(m, u):
    """The geometric stiffness alone, for the linearised comparisons."""
    m.KGv *= 0
    for plate in m.elements:
        update_KG(u, plate, m.points, m.weights, m.KGr, m.KGc, m.KGv)
    return coo_matrix((m.KGv, (m.KGr, m.KGc)), shape=(m.N, m.N)).tocsc()


def calc_fint(m, u):
    fint = np.zeros(m.N)
    for plate in m.elements:
        update_fint(u, plate, m.points, m.weights, fint)
    return fint


def forces(m, load, transverse):
    """Axial compression along x, plus a transverse load at the centre."""
    fext = np.zeros(m.N)
    checkBottomEdge = isclose(m.x, 0)
    checkTopEdge = isclose(m.x, a)
    fext[0::DOF][checkBottomEdge] = +load/ny
    assert isclose(fext.sum(), load)
    fext[0::DOF][checkTopEdge] = -load/ny
    assert isclose(fext.sum(), 0)

    ftrans = np.zeros(m.N)
    check = isclose(m.x, a/2) & isclose(m.y, b/2)
    assert check.sum() == 1
    ftrans[6::DOF][check] = transverse
    return fext, ftrans


def newton_raphson(m, fext, u0=None, increments=1, epsilon=1.e-9,
                   max_iterations=40):
    """Full Newton-Raphson. Returns the solution and the residual history.

    The tangent is rebuilt every iteration, so the residual history is the
    direct evidence of whether KT is the derivative of fint: a consistent
    tangent squares the residual each step, an inconsistent one multiplies
    it by a constant factor instead.
    """
    u = np.zeros(m.N) if u0 is None else u0.copy()
    bu = m.bu
    history = []
    for increment in range(1, increments + 1):
        f = (float(increment)/increments)*fext
        scale = np.linalg.norm(f[bu])
        history = []
        for count in range(max_iterations):
            Ri = calc_fint(m, u) - f
            residual = np.linalg.norm(Ri[bu])/scale
            history.append(residual)
            print('    increment %d  count %d  residual %.6e'
                  % (increment, count, residual))
            if residual < epsilon:
                break
            KTuu = calc_KT(m, u)[bu, :][:, bu]
            u[bu] += spsolve(KTuu, -Ri[bu])
        else:
            raise RuntimeError('Not converged!')
    return u, history


def assert_quadratic_convergence(history, epsilon=1.e-9):
    """The last Newton step must square the residual, not merely shrink it.

    Quadratic convergence is an asymptotic statement, so it is the final
    transition that carries it: the step that takes the iterate from above
    the tolerance to below it. Earlier steps are not a fair test here,
    because the residual is normalised by the whole external load, which the
    axial part satisfies exactly at the first step while the out-of-plane
    part is still far from converged.

    With an inconsistent tangent the residual falls by a roughly constant
    factor throughout, so no step ever squares it and this fails.
    """
    assert history[-1] < epsilon, 'did not reach the tolerance'
    assert len(history) <= 6, (
        'took %d iterations to converge; a consistent tangent needs a '
        'handful' % len(history))
    assert len(history) >= 2, 'no Newton step was taken'
    prev, curr = history[-2], history[-1]
    # the floor keeps the test meaningful once prev**2 drops below the
    # round-off level of the residual itself
    assert curr < max(10.*prev**2, 1.e-14), (
        'the last step took the residual %.3e -> %.3e, which is linear '
        'rather than quadratic; the tangent is not the derivative of fint'
        % (prev, curr))


def linear_buckling_load(m, load):
    """The first buckling load, for the reference amplification."""
    fext, _ = forces(m, load, 0.)
    u0 = np.zeros(m.N)
    u0[m.bu] = spsolve(m.KC0uu, fext[m.bu])
    KGuu = calc_KG(m, u0)[m.bu, :][:, m.bu]
    eigvals, eigvecsu = eigsh(A=KGuu, k=5, which='SM', M=m.KC0uu, tol=1e-6,
                              sigma=1., mode='cayley')
    eigvals = -1./eigvals
    return eigvals[0]*load, u0


def test_static(plot=False):
    """Small transverse load: the linearised buckling amplification."""
    load = 300.  # N
    transverse = 0.01  # N

    m = build_model()
    fext, ftrans = forces(m, load, transverse)

    load_cr, u_axial = linear_buckling_load(m, load)
    print('load_cr', load_cr)
    assert isclose(load_cr, 501.59022126865693, rtol=1e-3)
    assert load < load_cr

    # the exact linearised answer: the transverse load resisted by the
    # bending stiffness reduced by the geometric stiffness of the axial load
    KG = calc_KG(m, u_axial)
    u_lin = np.zeros(m.N)
    u_lin[m.bu] = spsolve((m.KC0 + KG)[m.bu, :][:, m.bu], ftrans[m.bu])
    w_lin = u_lin[6::DOF].max()

    # and the transverse load on its own, for the amplification factor
    u_bend = np.zeros(m.N)
    u_bend[m.bu] = spsolve(m.KC0uu, ftrans[m.bu])
    w_bend = u_bend[6::DOF].max()

    u, history = newton_raphson(m, fext + ftrans)
    assert_quadratic_convergence(history)

    w = u[6::DOF].reshape(nx, ny).T
    print('w min max', w.min(), w.max())
    print('w linearised            ', w_lin)
    print('amplification           ', w.max()/w_bend)
    print('amplification ceiling   ', 1./(1. - load/load_cr))

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(m.xmesh, m.ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    # w/h is about 0.011 here, so the membrane stretching is negligible and
    # the nonlinear answer must reproduce the linearised one
    assert isclose(w.max(), w_lin, rtol=1e-3)

    # a point load excites many modes, so the amplification stays below the
    # single-mode ceiling 1/(1 - P/Pcr); it must still exceed one
    amplification = w.max()/w_bend
    assert 1. < amplification < 1./(1. - load/load_cr)

    assert isclose(w.max(), 1.1104539457685102e-05, rtol=1e-3)


def test_large_deflection(plot=False):
    """Transverse load reaching w/h of about 2.6: membrane stiffening."""
    load = 300.  # N
    transverse = 5.  # N

    m = build_model()
    fext, ftrans = forces(m, load, transverse)

    load_cr, u_axial = linear_buckling_load(m, load)

    # the linearised answer the plate would give if the von Karman terms did
    # not stiffen it as it deflects
    KG = calc_KG(m, u_axial)
    u_lin = np.zeros(m.N)
    u_lin[m.bu] = spsolve((m.KC0 + KG)[m.bu, :][:, m.bu], ftrans[m.bu])
    w_lin = u_lin[6::DOF].max()

    u, history = newton_raphson(m, fext + ftrans, increments=4)
    assert_quadratic_convergence(history)

    w = u[6::DOF].reshape(nx, ny).T
    print('w min max', w.min(), w.max())
    print('w/h', w.max()/h)
    print('w linearised', w_lin)

    if plot:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        levels = np.linspace(w.min(), w.max(), 300)
        plt.contourf(m.xmesh, m.ymesh, w, levels=levels)
        plt.colorbar()
        plt.show()

    # the regime is genuinely nonlinear
    assert 2. < w.max()/h < 3.

    # and membrane stretching makes the plate stiffer than the linearised
    # prediction, by a wide margin at this deflection
    assert w.max() < 0.6*w_lin

    assert isclose(w.max(), 2.5826494310546447e-03, rtol=1e-3)


if __name__ == '__main__':
    test_static(plot=True)
    test_large_deflection(plot=True)

r"""Follower pressure: update_fext_follower and update_KCfollower

A follower pressure stays normal to the deformed mid-surface and acts on its
deformed area, with the area vector `a = X_{,x} \times X_{,y}` truncated to
first order in the displacement gradients (``quadratic=0``) or complete
(``quadratic=1``). The checks, for both truncations:

1. in the undeformed state the force vector equals the dead pressure load
   `\int p \boldsymbol{S}^w dA`, computed with the rows of
   :meth:`.BFSCPlate2D.update_Sw`;
2. the force vector against an independent integration of `\int p
   \boldsymbol{S} \cdot a dA`, with cubic Hermite polynomials written here
   and ``np.cross``;
3. directional Taylor test of the force vector against `-KCfollower d`:
   exact for the first-order area vector, for which `F_p` is affine, and a
   remainder falling as `O(h)` for the complete one;
4. ``reference=0`` gives `F_p(u) - F_p(0)`;
5. no closed-form solution of the panels validation cases (rings and
   cylinders) applies to a flat plate, hence:

   - the skew part of the load stiffness against the contour and area
     integrals of ``theory/shells/follower_pressure/follower_pressure.md``
     of panels, and its symmetry for the boundary conditions and pressures
     for which that formula vanishes;
   - a large-deflection Newton-Raphson analysis of a simply supported
     plate, in which the follower pressure gives a solution measurably
     different from the dead pressure, converging quadratically, against
     reference values of the plate_clpt_donnell model of panels, the same
     von Karman kinematics and area vector with another discretization;

6. `KC0 + KCNL + KG + KCfollower` is the Jacobian of the residual `f_{int}
   - F_p`, against finite differences.

Units: N, m, Pa.
"""
import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from composites import isotropic_plate, laminated_plate

from bfscplate2d import (BFSCPlate2D, update_KC0, update_KCNL, update_KG,
        update_fint, update_fext_follower, update_KCfollower, DOF, DOUBLE,
        INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE,
        KCfollower_SPARSE_SIZE)
from bfscplate2d.quadrature import get_points_weights

# exact for the complete area vector with a pressure of degree 2, see the
# Notes of update_fext_follower
NINT = 6
TRUNCATIONS = [0, 1]

LAMINAPROP = (142.5e9, 142.5e9/20, 0.25, 0.5*142.5e9/20, 0.5*142.5e9/20,
              0.5*142.5e9/20)
# unsymmetric and unbalanced, such that A16, A26 and B are non-zero
LAMINATE = laminated_plate(stack=[30, -45, 0, 90], laminaprop=LAMINAPROP,
                           plyt=0.125e-3)

A, B = 0.3, 0.2


def p_function(x, y):
    """Non-uniform pressure, of degree 2"""
    return 1.e4*(1 + x/A - 2*y**2/B**2)


def p_function_x(x, y):
    return 1.e4/A + 0*x*y


def p_function_y(x, y):
    return -1.e4*4*y/B**2 + 0*x


class Mesh(object):
    """Rectangular mesh of ``nx`` by ``ny`` nodes over ``a`` by ``b``"""
    def __init__(self, nx=4, ny=3, a=A, b=B, lam=LAMINATE, nint=NINT):
        self.a, self.b = a, b
        self.points, self.weights = get_points_weights(nint=nint)
        xlin = np.linspace(0, a, nx)
        ylin = np.linspace(0, b, ny)
        xmesh, ymesh = np.meshgrid(xlin, ylin)
        ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
        self.x = ncoords[:, 0]
        self.y = ncoords[:, 1]
        nids_mesh = np.arange(nx*ny).reshape(nx, ny)
        n1s = nids_mesh[:-1, :-1].flatten()
        n2s = nids_mesh[1:, :-1].flatten()
        n3s = nids_mesh[1:, 1:].flatten()
        n4s = nids_mesh[:-1, 1:].flatten()
        self.N = DOF*nx*ny
        self.elements = []
        for k, (n1, n2, n3, n4) in enumerate(zip(n1s, n2s, n3s, n4s)):
            plate = BFSCPlate2D()
            plate.n1, plate.n2, plate.n3, plate.n4 = n1, n2, n3, n4
            plate.c1 = DOF*n1
            plate.c2 = DOF*n2
            plate.c3 = DOF*n3
            plate.c4 = DOF*n4
            plate.ABD = lam.ABD
            plate.lex = a/(nx - 1)
            plate.ley = b/(ny - 1)
            plate.init_k_KC0 = k*KC0_SPARSE_SIZE
            plate.init_k_KCNL = k*KCNL_SPARSE_SIZE
            plate.init_k_KG = k*KG_SPARSE_SIZE
            plate.init_k_KCfollower = k*KCfollower_SPARSE_SIZE
            self.elements.append(plate)
        self.h = lam.h

    def origin(self, plate):
        return self.x[plate.n1], self.y[plate.n1]

    def pressure(self, plate, func):
        r"""``p[i, j]`` at ``xi = points[i]``, ``eta = points[j]``"""
        x1, y1 = self.origin(plate)
        x = x1 + plate.lex*(self.points + 1)/2
        y = y1 + plate.ley*(self.points + 1)/2
        p = func(x[:, None], y[None, :])*np.ones((x.size, y.size))
        return np.ascontiguousarray(p, dtype=DOUBLE)

    def fext_follower(self, u, func, quadratic, reference=1):
        f = np.zeros(self.N, dtype=DOUBLE)
        for plate in self.elements:
            update_fext_follower(u, plate, self.points, self.weights,
                                 self.pressure(plate, func), f, quadratic,
                                 reference)
        return f

    def _matrix(self, size, call):
        n = size*len(self.elements)
        r = np.zeros(n, dtype=INT)
        c = np.zeros(n, dtype=INT)
        v = np.zeros(n, dtype=DOUBLE)
        for plate in self.elements:
            call(plate, r, c, v)
        return coo_matrix((v, (r, c)), shape=(self.N, self.N)).tocsc()

    def KCfollower(self, u, func, quadratic):
        return self._matrix(KCfollower_SPARSE_SIZE,
            lambda plate, r, c, v: update_KCfollower(u, plate, self.points,
                self.weights, self.pressure(plate, func), r, c, v,
                quadratic))

    def KC0(self):
        return self._matrix(KC0_SPARSE_SIZE,
            lambda plate, r, c, v: update_KC0(plate, self.points,
                self.weights, r, c, v))

    def KT(self, u):
        KCNL = self._matrix(KCNL_SPARSE_SIZE,
            lambda plate, r, c, v: update_KCNL(u, plate, self.points,
                self.weights, r, c, v))
        KG = self._matrix(KG_SPARSE_SIZE,
            lambda plate, r, c, v: update_KG(u, plate, self.points,
                self.weights, r, c, v))
        return self.KC0() + KCNL + KG

    def fint(self, u):
        f = np.zeros(self.N, dtype=DOUBLE)
        for plate in self.elements:
            update_fint(u, plate, self.points, self.weights, f)
        return f

    def random_state(self, rng, scale=1.):
        """In-plane displacements of the order of the thickness, deflections
        well above it, such that the nonlinear terms carry weight"""
        u = scale*self.h*rng.standard_normal(self.N)
        u[6::DOF] *= 20
        lmin = min(self.a, self.b)
        for i in (1, 2, 4, 5, 7, 8):
            u[i::DOF] *= 1/lmin
        u[7::DOF] *= 20
        u[8::DOF] *= 20
        u[9::DOF] *= 20/lmin**2
        return u

    def positions(self, plate):
        return np.concatenate([np.arange(c, c + DOF) for c in
                               (plate.c1, plate.c2, plate.c3, plate.c4)])


# ----------------------------------------------------------------------------
# Independent interpolation: the cubic Hermite polynomials on t in [0, 1],
# t = (xi + 1)/2, written without the functions of the element

H00 = Polynomial([1, 0, -3, 2])
H10 = Polynomial([0, 1, -2, 1])
H01 = Polynomial([0, 0, 3, -2])
H11 = Polynomial([0, 0, -1, 1])
# node 1 (xi, eta) = (-1, -1), node 2 (+1, -1), node 3 (+1, +1), node 4 (-1, +1)
CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def hermite(t, L, end, order):
    """Value and slope functions of the node at ``end`` (0 or 1) of a segment
    of length ``L``, derivative ``order`` (0 or 1) with respect to x"""
    val, slope = (H00, L*H10) if end == 0 else (H01, L*H11)
    if order == 1:
        val, slope = val.deriv()/L, slope.deriv()/L
    return val(t), slope(t)


def rows(t, s, lex, ley):
    """Rows of u, v, w and of their x and y derivatives, 40 entries each, at
    the point (t, s) of the element, as a dict ``rows[field][d]`` with ``d``
    0 for the function, 1 for d/dx and 2 for d/dy"""
    out = {f: np.zeros((3, 4*DOF)) for f in 'uvw'}
    for node, (ex, ey) in enumerate(CORNERS):
        for d, (ox, oy) in enumerate([(0, 0), (1, 0), (0, 1)]):
            fx, fxs = hermite(t, lex, ex, ox)
            fy, fys = hermite(s, ley, ey, oy)
            c = node*DOF
            for f, k in (('u', 0), ('v', 3)):
                out[f][d, c + k] = fx*fy
                out[f][d, c + k + 1] = fxs*fy
                out[f][d, c + k + 2] = fx*fys
            out['w'][d, c + 6] = fx*fy
            out['w'][d, c + 7] = fxs*fy
            out['w'][d, c + 8] = fx*fys
            out['w'][d, c + 9] = fxs*fys
    return out


def area_vector(R, ue, quadratic):
    u_x, u_y = R['u'][1] @ ue, R['u'][2] @ ue
    v_x, v_y = R['v'][1] @ ue, R['v'][2] @ ue
    w_x, w_y = R['w'][1] @ ue, R['w'][2] @ ue
    if quadratic:
        return np.cross([1 + u_x, v_x, w_x], [u_y, 1 + v_y, w_y])
    return np.array([-w_x, -w_y, 1 + u_x + v_y])


def fext_follower_reference(m, u, func, quadratic, npts=8):
    r"""`\int p (\boldsymbol{S}^u a_x + \boldsymbol{S}^v a_y
    + \boldsymbol{S}^w a_z) dA` element by element"""
    pts, wts = leggauss(npts)
    out = np.zeros(m.N)
    for plate in m.elements:
        pos = m.positions(plate)
        ue = u[pos]
        x1, y1 = m.origin(plate)
        for xi, wxi in zip(pts, wts):
            t = (xi + 1)/2
            for eta, weta in zip(pts, wts):
                s = (eta + 1)/2
                R = rows(t, s, plate.lex, plate.ley)
                a = area_vector(R, ue, quadratic)
                p = func(x1 + t*plate.lex, y1 + s*plate.ley)
                dA = wxi*weta*plate.lex*plate.ley/4
                out[pos] += dA*p*(R['u'][0]*a[0] + R['v'][0]*a[1]
                                  + R['w'][0]*a[2])
    return out


# ----------------------------------------------------------------------------
# 1 to 4, verification

def test_sparse_layout():
    """KCfollower_SPARSE_SIZE entries, exactly at the positions of the
    element, every pair once, none of the diagonal blocks [u, u], [v, v] and
    [w, w]"""
    assert KCfollower_SPARSE_SIZE == 2*12*12 + 4*12*16
    m = Mesh(nx=2, ny=2)
    plate = m.elements[0]
    offset = 7
    plate.init_k_KCfollower = offset
    size = KCfollower_SPARSE_SIZE + 2*offset
    r = np.full(size, -1, dtype=INT)
    c = np.full(size, -1, dtype=INT)
    v = np.full(size, np.nan, dtype=DOUBLE)
    v[offset:offset + KCfollower_SPARSE_SIZE] = 0.
    u = m.random_state(np.random.default_rng(0))
    update_KCfollower(u, plate, m.points, m.weights,
                      m.pressure(plate, p_function), r, c, v, 1)
    sl = slice(offset, offset + KCfollower_SPARSE_SIZE)
    assert np.all(r[:offset] == -1) and np.all(r[sl.stop:] == -1)
    assert np.all(np.isnan(v[:offset])) and np.all(np.isnan(v[sl.stop:]))
    pairs = set(zip(r[sl], c[sl]))
    assert len(pairs) == KCfollower_SPARSE_SIZE
    field = {}
    for i in range(DOF):
        field[i] = 'u' if i < 3 else ('v' if i < 6 else 'w')
    assert all(field[i % DOF] != field[j % DOF] for i, j in pairs)
    assert np.all(np.isfinite(v[sl]))


@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_undeformed_state_equals_dead_load(quadratic):
    m = Mesh()
    dead = np.zeros(m.N)
    for plate in m.elements:
        p = m.pressure(plate, p_function)
        fe = np.zeros(4*DOF)
        for i, xi in enumerate(m.points):
            for j, eta in enumerate(m.points):
                plate.update_Sw(xi, eta)
                fe += (plate.lex*plate.ley/4*m.weights[i]*m.weights[j]
                       *p[i, j]*np.asarray(plate.Sw))
        dead[m.positions(plate)] += fe
    u0 = np.zeros(m.N)
    f = m.fext_follower(u0, p_function, quadratic)
    assert np.abs(dead).max() > 0
    assert np.allclose(f, dead, rtol=1e-13, atol=1e-13*np.abs(dead).max())
    # the dead load acts only on the degrees of freedom of w
    assert np.all(f[0::DOF] == 0) and np.all(f[3::DOF] == 0)
    assert np.all(m.fext_follower(u0, p_function, quadratic, reference=0) == 0)


@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_force_against_independent_quadrature(quadratic):
    m = Mesh()
    rng = np.random.default_rng(11)
    u = m.random_state(rng)
    f = m.fext_follower(u, p_function, quadratic)
    ref = fext_follower_reference(m, u, p_function, quadratic)
    assert np.allclose(f, ref, rtol=1e-12, atol=1e-12*np.abs(ref).max())
    # the follower terms matter in this state
    f0 = m.fext_follower(0*u, p_function, quadratic)
    assert np.linalg.norm(f - f0) > 1e-2*np.linalg.norm(f0)


@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_taylor_follower_force(quadratic):
    m = Mesh()
    rng = np.random.default_rng(0)
    u = m.random_state(rng)
    d = m.random_state(rng)
    F0 = m.fext_follower(u, p_function, quadratic)
    Kd = -(m.KCfollower(u, p_function, quadratic) @ d)
    assert np.linalg.norm(Kd) > 0
    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5]
    errors = [np.linalg.norm(m.fext_follower(u + h*d, p_function, quadratic)
                             - F0 - h*Kd)/np.linalg.norm(h*Kd)
              for h in steps]
    if quadratic == 0:
        # F_p is affine in u
        assert max(errors) < 1.e-9, errors
    else:
        for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
            assert err < 0.2*prev, errors
        coefficients = [err/h for h, err in zip(steps, errors)]
        assert max(coefficients)/min(coefficients) < 1.5, errors


def test_linear_load_stiffness_does_not_depend_on_u():
    m = Mesh()
    rng = np.random.default_rng(5)
    u = m.random_state(rng)
    K0 = m.KCfollower(0*u, p_function, 0).toarray()
    K1 = m.KCfollower(u, p_function, 0).toarray()
    assert np.abs(K0).max() > 0
    assert np.all(K0 == K1)
    # equal to the complete area vector at u = 0
    Kq = m.KCfollower(0*u, p_function, 1).toarray()
    assert np.allclose(Kq, K0, rtol=0, atol=1e-14*np.abs(K0).max())


@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_reference_zero(quadratic):
    m = Mesh()
    rng = np.random.default_rng(3)
    u = m.random_state(rng)
    full = m.fext_follower(u, p_function, quadratic, reference=1)
    dead = m.fext_follower(0*u, p_function, quadratic, reference=1)
    part = m.fext_follower(u, p_function, quadratic, reference=0)
    assert np.allclose(part, full - dead, rtol=1e-12,
                       atol=1e-12*np.abs(full).max())


# ----------------------------------------------------------------------------
# 5, the skew part of the load stiffness and its symmetry

def skew_reference(m, d, D, p, px, py, npts=8):
    r"""`P(d, D) - P(D, d)` of the first-order area vector, by the contour
    integral over the boundary of the plate and the area integral of the
    derivatives of `p` of follower_pressure.md of panels"""
    pts, wts = leggauss(npts)
    contour = 0.
    area = 0.
    for plate in m.elements:
        pos = m.positions(plate)
        de, De = d[pos], D[pos]
        x1, y1 = m.origin(plate)

        def fields(t, s):
            R = rows(t, s, plate.lex, plate.ley)
            return ([R[f][0] @ de for f in 'uvw'],
                    [R[f][0] @ De for f in 'uvw'])

        # edges of the element on the boundary of the plate: (t, s) on the
        # edge as a function of the Gauss coordinate, outward normal, length
        edges = []
        if isclose(y1, 0):
            edges.append((lambda g: (g, 0.), (0, -1), plate.lex))
        if isclose(x1 + plate.lex, m.a):
            edges.append((lambda g: (1., g), (1, 0), plate.ley))
        if isclose(y1 + plate.ley, m.b):
            edges.append((lambda g: (g, 1.), (0, 1), plate.lex))
        if isclose(x1, 0):
            edges.append((lambda g: (0., g), (-1, 0), plate.ley))
        for where, (nx, ny), L in edges:
            for g, wg in zip(pts, wts):
                t, s = where((g + 1)/2)
                (du, dv, dw), (Du, Dv, Dw) = fields(t, s)
                pv = p(x1 + t*plate.lex, y1 + s*plate.ley)
                contour += wg*L/2*pv*((du*Dw - dw*Du)*nx + (dv*Dw - dw*Dv)*ny)
        for xi, wxi in zip(pts, wts):
            for eta, weta in zip(pts, wts):
                t, s = (xi + 1)/2, (eta + 1)/2
                (du, dv, dw), (Du, Dv, Dw) = fields(t, s)
                x, y = x1 + t*plate.lex, y1 + s*plate.ley
                area += (wxi*weta*plate.lex*plate.ley/4
                         *(px(x, y)*(du*Dw - dw*Du) + py(x, y)*(dv*Dw - dw*Dv)))
    return -contour + area


@pytest.mark.parametrize('case', ['uniform', 'function'])
def test_skew_part_of_load_stiffness(case):
    r"""`d^T K D - D^T K d = -(P(d, D) - P(D, d))` for a free plate, all
    degrees of freedom free, since `d^T K D = -P(d, D)`"""
    if case == 'uniform':
        p = lambda x, y: -2.e4 + 0*x*y
        px = py = lambda x, y: 0*x*y
    else:
        p, px, py = p_function, p_function_x, p_function_y
    m = Mesh()
    rng = np.random.default_rng(21)
    d = m.random_state(rng)
    D = m.random_state(rng)
    K = m.KCfollower(0*d, p, 0)
    skew = d @ (K @ D) - D @ (K @ d)
    ref = -skew_reference(m, d, D, p, px, py)
    assert abs(ref) > 1e-3*abs(d @ (K @ D))
    assert isclose(skew, ref, rtol=1e-10), (skew, ref)


def boundary_flags(m, edges):
    r"""Constrained degrees of freedom: for each edge in ``edges``, a dict
    ``{'x1': 'w', ...}`` of the displacements set to zero along it, with
    their derivatives along the edge"""
    bk = np.zeros(m.N, dtype=bool)
    on = {'x1': isclose(m.x, 0), 'x2': isclose(m.x, m.a),
          'y1': isclose(m.y, 0), 'y2': isclose(m.y, m.b)}
    # value and derivative along the edge: y for the edges x = const
    along = {'x': {'u': (0, 2), 'v': (3, 5), 'w': (6, 8)},
             'y': {'u': (0, 1), 'v': (3, 4), 'w': (6, 7)}}
    for edge, fields in edges.items():
        for f in fields:
            for i in along[edge[0]][f]:
                bk[i::DOF] |= on[edge]
    return bk


def asym(K):
    K = K.toarray()
    return np.linalg.norm(K - K.T)/np.linalg.norm(K)


def test_symmetry():
    r"""Symmetric when `p` is uniform over the whole plate and, on each edge,
    `w = 0` or the displacement normal to the edge is zero, unsymmetric
    otherwise"""
    m = Mesh(nx=5, ny=4)
    u0 = np.zeros(m.N)
    uniform = lambda x, y: -1.e4 + 0*x*y
    ss = {'x1': 'w', 'x2': 'w', 'y1': 'w', 'y2': 'w'}
    normal = {'x1': 'u', 'x2': 'u', 'y1': 'v', 'y2': 'v'}
    mixed = {'x1': 'w', 'x2': 'u', 'y1': 'v', 'y2': 'w'}
    K = m.KCfollower(u0, uniform, 0)
    # the element matrix alone is unsymmetric, its edges are interior edges
    single = Mesh(nx=2, ny=2)
    assert asym(single.KCfollower(np.zeros(single.N), uniform, 0)) > 1e-3
    for edges in (ss, normal, mixed):
        bu = ~boundary_flags(m, edges)
        assert asym(K[bu, :][:, bu]) < 1e-12, edges
    # a free edge
    free = dict(ss)
    free.pop('x2')
    bu = ~boundary_flags(m, free)
    assert asym(K[bu, :][:, bu]) > 1e-3
    # a varying pressure
    bu = ~boundary_flags(m, ss)
    Kf = m.KCfollower(u0, p_function, 0)
    assert asym(Kf[bu, :][:, bu]) > 1e-3
    # a pressure on part of the plate
    patch = lambda x, y: np.where(x < m.a/2, -1.e4, 0.) + 0*y
    Kp = m.KCfollower(u0, patch, 0)
    assert asym(Kp[bu, :][:, bu]) > 1e-3
    # the complete area vector away from u = 0: with w = 0 on the whole
    # boundary and a uniform p the load has the potential -p V, V = int w
    # ((1 + u,x) (1 + v,y) - u,y v,x) dA, hence a symmetric matrix also at
    # u != 0, unsymmetric with a free edge
    rng = np.random.default_rng(2)
    u = m.random_state(rng)
    u[boundary_flags(m, ss)] = 0
    Kq = m.KCfollower(u, uniform, 1)
    K0 = m.KCfollower(0*u, uniform, 1)
    assert np.linalg.norm((Kq - K0).toarray()) > 1e-2*np.linalg.norm(K0.toarray())
    assert asym(Kq[bu, :][:, bu]) < 1e-12
    bu = ~boundary_flags(m, free)
    assert asym(Kq[bu, :][:, bu]) > 1e-3


# ----------------------------------------------------------------------------
# 6, Jacobian of the residual

@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_tangent_is_jacobian_of_residual(quadratic):
    r"""`K_T = KC0 + KCNL + KG + KCfollower` against the central differences
    of `R = f_{int} - F_p`, every entry, and a Taylor test"""
    m = Mesh(nx=3, ny=2)
    func = lambda x, y: 100*p_function(x, y)
    rng = np.random.default_rng(7)
    u = m.random_state(rng)

    def R(u):
        return m.fint(u) - m.fext_follower(u, func, quadratic)

    KT = (m.KT(u) + m.KCfollower(u, func, quadratic)).toarray()
    # steps relative to each degree of freedom, large enough to stay clear
    # of the round-off of fint, whose stiffness is ~500 times that of the
    # load here
    step = 1.e-4
    J = np.empty((m.N, m.N))
    scale = np.abs(u).copy()
    scale[scale == 0] = 1.
    for j in range(m.N):
        e = np.zeros(m.N)
        e[j] = step*scale[j]
        J[:, j] = (R(u + e) - R(u - e))/(2*e[j])
    err = np.linalg.norm(KT - J)/np.linalg.norm(J)
    assert err < 1.e-8, err
    # also small compared to the load stiffness, which is ~0.2% of KT
    Kf = m.KCfollower(u, func, quadratic).toarray()
    assert np.linalg.norm(KT - J)/np.linalg.norm(Kf) < 1.e-5
    d = m.random_state(rng)
    R0 = R(u)
    KTd = KT @ d
    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5]
    errors = [np.linalg.norm(R(u + h*d) - R0 - h*KTd)/np.linalg.norm(h*KTd)
              for h in steps]
    for h, prev, e in zip(steps[1:], errors[:-1], errors[1:]):
        assert e < 0.2*prev, errors


# ----------------------------------------------------------------------------
# 5, large deflection of a cantilever plate against panels

# plate_clpt_donnell of panels (commit f7a4275), the same von Karman strains
# and area vector discretized with m = n = 16 Bardell functions and nx = ny =
# 20 Gauss points, Newton-Raphson of structsolve with 10 increments and
# relTOL = 1e-11: w and u at (a, b/2) of the cantilever below. Refining from
# m = n = 12 changes w by 2e-4, and the BFSC mesh of the test is within 5e-4
# of these values (20 by 20 elements: 5e-5), both converging from below
PANELS = {
    'dead': dict(w=20.05040761, u=-3.06710517e-03),
    0: dict(w=20.23571166, u=-3.12566486e-03),
    1: dict(w=20.23571209, u=-3.12566500e-03),
}


def newton_raphson(m, bu, func, load, increments=5, tol=1.e-9,
                   max_iterations=30):
    r"""Solves `f_{int}(u) = \lambda F_p(u)` (``load`` 0 or 1, the truncation
    of the follower pressure) or `f_{int}(u) = \lambda F_p(0)` (``load =
    'dead'``) for `\lambda` up to 1, returning the solution and the relative
    residuals of each increment"""
    u = np.zeros(m.N)
    histories = []
    for lbd in np.linspace(0, 1, increments + 1)[1:]:
        pf = lambda x, y: lbd*func(x, y)
        history = []
        for count in range(max_iterations):
            if load == 'dead':
                F = m.fext_follower(0*u, pf, 0)
            else:
                F = m.fext_follower(u, pf, load)
            R = m.fint(u) - F
            history.append(np.linalg.norm(R[bu])/np.linalg.norm(F[bu]))
            if history[-1] < tol:
                break
            KT = m.KT(u)
            if load != 'dead':
                KT = KT + m.KCfollower(u, pf, load)
            u[bu] -= spsolve(KT[bu, :][:, bu].tocsc(), R[bu])
        else:
            raise RuntimeError('not converged: %s' % history)
        histories.append(history)
    return u, histories


def test_cantilever_large_deflection_against_panels():
    r"""Square isotropic cantilever, clamped at `x = 0`, the other edges
    free, under a uniform pressure pushing along `+z`, with the tip
    deflection about 20 times the thickness, `w/a = 0.13`. The follower
    pressure tilts with the plate and its in-plane component `-p w_{,x}`
    pulls the tip towards the clamp: the tip deflection is 0.92% larger and
    the tip displacement `u` 1.9% larger in magnitude than under the dead
    pressure, against panels within 5e-4 for `w`, i.e. the follower effect
    itself within 1%. The load stiffness is unsymmetric (free edges) and
    Newton-Raphson converges quadratically with it"""
    E, nu, h = 70.e9, 0.3, 2.e-3
    a = b = 0.3
    nel = 10
    m = Mesh(nx=nel + 1, ny=nel + 1, a=a, b=b,
             lam=isotropic_plate(thickness=h, E=E, nu=nu))
    bk = boundary_flags(m, {'x1': 'uvw'})
    bk[7::DOF] |= isclose(m.x, 0)
    bk[9::DOF] |= isclose(m.x, 0)
    bu = ~bk
    func = lambda x, y: 2.e3 + 0*x*y
    K = m.KCfollower(np.zeros(m.N), func, 0)
    assert asym(K[bu, :][:, bu]) > 1e-3
    tip = np.where(isclose(m.x, a) & isclose(m.y, b/2))[0][0]
    out = {}
    for load in ('dead', 0, 1):
        u, histories = newton_raphson(m, bu, func, load)
        out[load] = dict(w=u[DOF*tip + 6]/h, u=u[DOF*tip])
        # quadratic, e_k+1 <= 10 e_k**1.8, from 1e-3 down to the round-off of
        # the residual, about 1e-11 here
        checked = 0
        for history in histories:
            for e0, e1 in zip(history[:-1], history[1:]):
                if 1.e-10 < e0 < 1.e-3:
                    assert e1 <= max(10*e0**1.8, 1.e-10), history
                    checked += 1
        assert checked >= 3, histories
        for key in ('w', 'u'):
            assert isclose(out[load][key], PANELS[load][key], rtol=1.e-3), (
                load, key, out[load][key], PANELS[load][key])
    effect = out[0]['w']/out['dead']['w'] - 1
    effect_ref = PANELS[0]['w']/PANELS['dead']['w'] - 1
    assert effect > 9.e-3
    assert isclose(effect, effect_ref, rtol=1.e-2), (effect, effect_ref)
    assert out[0]['u']/out['dead']['u'] - 1 > 1.5e-2
    # the truncation of the area vector is immaterial, as follower_pressure.md
    # of panels predicts for the moderate rotations of the von Karman strains
    assert isclose(out[1]['w'], out[0]['w'], rtol=1.e-6)

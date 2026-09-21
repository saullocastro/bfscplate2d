"""The tangent stiffness matrix must be the derivative of the internal force

KT = KC0 + KCNL(u) + KG(u) is what every Newton-Raphson built on this element
uses as the Jacobian of fint. If the two drift apart, the analyses still
converge, but linearly instead of quadratically, with a contraction factor
set by how far KT is from the true Jacobian, which on a fine mesh is close
enough to one to exhaust the iteration cap at every load step.

The check is a directional Taylor test,

    |fint(u + h d) - fint(u) - h KT d| / |h KT d|

which falls proportionally to h for a consistent tangent and plateaus for an
inconsistent one. It runs over several random states and directions, in a
deformed state of the order of the plate thickness so that the nonlinear
terms actually carry weight, and over a symmetric, an unsymmetric and an
isotropic laminate, so that the B matrix coupling membrane and bending is
exercised rather than sitting at zero.
"""
import sys

sys.path.append(r'..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from composites import laminated_plate, isotropic_plate

from bfscplate2d import (BFSCPlate2D, update_KC0, update_KCNL, update_KG,
        update_fint, DOF, DOUBLE, INT, KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE,
        KG_SPARSE_SIZE)
from bfscplate2d.quadrature import get_points_weights

NINT = 4
N = 4*DOF

LAMINAPROP = (127.629e9, 11.3074e9, 0.300235, 6.00257e9, 6.00257e9, 6.00257e9)
PLYT = 0.00101539/8

LAMINATES = {
    # B = 0, the case every existing test covers
    'symmetric': lambda: laminated_plate(
        stack=[45, -45, 0, 90, 90, 0, -45, 45],
        laminaprop=LAMINAPROP, plyt=PLYT),
    # B != 0, so that the membrane-bending coupling terms of fint and of the
    # tangent have to agree as well
    'unsymmetric': lambda: laminated_plate(
        stack=[0, 45, 90, -45], laminaprop=LAMINAPROP, plyt=PLYT),
    'isotropic': lambda: isotropic_plate(thickness=0.001, E=0.7e11, nu=0.3),
}


def make_element(laminate):
    """One element of the plate used by the static tests, with its laminate."""
    lam = LAMINATES[laminate]()
    points, weights = get_points_weights(nint=NINT)
    plate = BFSCPlate2D()
    plate.n1, plate.n2, plate.n3, plate.n4 = 1, 2, 3, 4
    plate.c1, plate.c2, plate.c3, plate.c4 = 0, DOF, 2*DOF, 3*DOF
    plate.ABD = lam.ABD
    plate.lex = 0.9/12
    plate.ley = 0.5/6
    plate.init_k_KC0 = 0
    plate.init_k_KCNL = 0
    plate.init_k_KG = 0
    return plate, points, weights, lam


def assemble(update, plate, points, weights, size, u=None):
    r = np.zeros(size, dtype=INT)
    c = np.zeros(size, dtype=INT)
    v = np.zeros(size, dtype=DOUBLE)
    if u is None:
        update(plate, points, weights, r, c, v)
    else:
        update(u, plate, points, weights, r, c, v)
    return coo_matrix((v, (r, c)), shape=(N, N)).toarray()


def make_callables(laminate):
    plate, points, weights, lam = make_element(laminate)

    def fint(u):
        f = np.zeros(N, dtype=DOUBLE)
        update_fint(u, plate, points, weights, f)
        return f

    KC0 = assemble(update_KC0, plate, points, weights, KC0_SPARSE_SIZE)

    def KT(u):
        return (KC0
                + assemble(update_KCNL, plate, points, weights,
                           KCNL_SPARSE_SIZE, u)
                + assemble(update_KG, plate, points, weights,
                           KG_SPARSE_SIZE, u))

    return fint, KT, lam.h


@pytest.mark.parametrize('laminate', sorted(LAMINATES))
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_tangent_is_derivative_of_fint(laminate, seed):
    """Taylor test: the error must fall by about ten for each decade of h."""
    fint, KT, h_plate = make_callables(laminate)
    rng = np.random.default_rng(seed)
    u = h_plate*rng.standard_normal(N)
    d = h_plate*rng.standard_normal(N)

    f0 = fint(u)
    KTd = KT(u) @ d
    scale = np.linalg.norm(KTd)
    assert scale > 0

    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    errors = [np.linalg.norm(fint(u + h*d) - f0 - h*KTd)/(h*scale)
              for h in steps]

    # a consistent tangent leaves a second order remainder, so the error
    # falls by ten for every decade of h; an inconsistent one leaves a first
    # order remainder, so the error plateaus instead
    for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
        assert err < 0.2*prev, (
            'error did not fall by ten going to h=%.0e: %.3e -> %.3e; the '
            'tangent is not the derivative of fint' % (h, prev, err))

    # the same statement without an arbitrary scale: error/h is the constant
    # that multiplies the second derivative, so it must not drift with h.
    # An inconsistent tangent spreads this over four orders of magnitude
    coefficients = [err/h for h, err in zip(steps, errors)]
    spread = max(coefficients)/min(coefficients)
    assert spread < 2., (
        'error/h drifted by a factor %.1f over %.0e..%.0e, so the remainder '
        'is not second order' % (spread, steps[0], steps[-1]))

    # and the remainder is small in absolute terms once h is small, rather
    # than merely self-consistent
    assert errors[-1] < 1.e-6, (
        'residual %.3e at h=%.0e is too large for a consistent tangent'
        % (errors[-1], steps[-1]))


@pytest.mark.parametrize('laminate', sorted(LAMINATES))
def test_tangent_matches_finite_difference_jacobian(laminate):
    """Every entry of KT, against a central difference of fint."""
    fint, KT, h_plate = make_callables(laminate)
    rng = np.random.default_rng(7)
    u = h_plate*rng.standard_normal(N)

    step = 1.e-8
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (fint(u + e) - fint(u - e))/(2*step)

    K = KT(u)
    err = np.linalg.norm(K - J)/np.linalg.norm(J)
    assert err < 1.e-6, 'KT differs from d(fint)/du by %.3e' % err


@pytest.mark.parametrize('laminate', sorted(LAMINATES))
def test_tangent_is_symmetric(laminate):
    """A tangent that comes from a strain energy is symmetric."""
    fint, KT, h_plate = make_callables(laminate)
    rng = np.random.default_rng(11)
    u = h_plate*rng.standard_normal(N)
    K = KT(u)
    assert np.linalg.norm(K - K.T)/np.linalg.norm(K) < 1.e-12


@pytest.mark.parametrize('laminate', sorted(LAMINATES))
def test_fint_and_tangent_vanish_in_the_undeformed_state(laminate):
    """At u = 0 there is no internal force, and KT reduces to KC0."""
    plate, points, weights, lam = make_element(laminate)
    u = np.zeros(N, dtype=DOUBLE)
    f = np.zeros(N, dtype=DOUBLE)
    update_fint(u, plate, points, weights, f)
    assert np.all(f == 0)

    KC0 = assemble(update_KC0, plate, points, weights, KC0_SPARSE_SIZE)
    KCNL = assemble(update_KCNL, plate, points, weights, KCNL_SPARSE_SIZE, u)
    KG = assemble(update_KG, plate, points, weights, KG_SPARSE_SIZE, u)
    assert np.abs(KCNL).max() == 0
    assert np.abs(KG).max() == 0
    assert np.abs(KC0).max() > 0


@pytest.mark.parametrize('laminate', sorted(LAMINATES))
def test_fint_is_linear_for_small_displacements(laminate):
    """For a state well below the thickness, fint approaches KC0 @ u."""
    plate, points, weights, lam = make_element(laminate)
    rng = np.random.default_rng(3)
    u = 1.e-6*lam.h*rng.standard_normal(N)
    f = np.zeros(N, dtype=DOUBLE)
    update_fint(u, plate, points, weights, f)
    KC0 = assemble(update_KC0, plate, points, weights, KC0_SPARSE_SIZE)
    assert np.linalg.norm(f - KC0 @ u)/np.linalg.norm(KC0 @ u) < 1.e-6


if __name__ == '__main__':
    for name in sorted(LAMINATES):
        fint, KT, h_plate = make_callables(name)
        rng = np.random.default_rng(0)
        u = h_plate*rng.standard_normal(N)
        d = h_plate*rng.standard_normal(N)
        f0 = fint(u)
        KTd = KT(u) @ d
        print(name)
        prev = None
        for h in (1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7):
            err = (np.linalg.norm(fint(u + h*d) - f0 - h*KTd)
                   / np.linalg.norm(h*KTd))
            ratio = '' if prev is None else '   (x %.2f)' % (err/prev)
            print('   h %.0e   rel err %.6e%s' % (h, err, ratio))
            prev = err

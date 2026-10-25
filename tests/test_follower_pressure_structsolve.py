r"""Follower pressure solved with structsolve.Analysis

The large-deflection cantilever of ``test_follower_pressure.py`` under a
uniform follower pressure, solved with the full Newton-Raphson method of
structsolve instead of the iterations written in that test. The callables
of ``FollowerCallables`` receive the load factor ``inc``, which structsolve
passes to the callables that declare it:

- ``calc_fext`` gives the dead part `\lambda \boldsymbol{F}_p(\boldsymbol{0})` of the pressure;
- ``calc_fint`` subtracts the part that depends on the displacements,
  `\lambda (\boldsymbol{F}_p(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{0}))`, of ``update_fext_follower`` with
  ``reference=0``, such that structsolve solves the equilibrium of the
  follower pressure, `\boldsymbol{F}_{int}(\boldsymbol{u}) = \lambda
  \boldsymbol{F}_p(\boldsymbol{u})`;
- ``calc_kC`` adds the load stiffness matrix of the pressure `\lambda p`,
  unsymmetric for the free edges of the cantilever.

The tip displacements are checked against the reference values of the
plate_clpt_donnell model of panels, as in ``test_follower_pressure.py``.

Units: N, m, Pa.
"""
import sys
sys.path.append('..')

import numpy as np
from numpy import isclose
import pytest
from scipy.sparse import coo_matrix
from composites import isotropic_plate
from structsolve import Analysis

from bfscplate2d import (update_fint, update_KCNL, update_KG,
        update_fext_follower, update_KCfollower, DOF, DOUBLE, INT,
        KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, KCfollower_SPARSE_SIZE)

from test_follower_pressure import Mesh, boundary_flags, PANELS, TRUNCATIONS


class FollowerCallables(object):
    r"""The callables of structsolve.Analysis for the mesh ``m`` under the
    follower pressure ``func(x, y)`` times the load factor

    They loop over the elements with the update functions and assemble, and
    work on the unknown degrees of freedom ``bu`` only, the partition of the
    free dofs of structsolve: the solution vector of the solvers is ``c =
    u[bu]``, and the force vectors and matrices are restricted to ``bu``.
    ``quadratic`` is the truncation of the area vector.
    """
    def __init__(self, m, bu, func, quadratic):
        self.m = m
        self.bu = bu
        self.quadratic = quadratic
        self.pressures = [m.pressure(plate, func) for plate in m.elements]
        self.KC0 = m.KC0()
        N = m.N
        num_elements = len(m.elements)
        self.KCNLr = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCNLc = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCNLv = np.zeros(KCNL_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        self.KGr = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGc = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=INT)
        self.KGv = np.zeros(KG_SPARSE_SIZE*num_elements, dtype=DOUBLE)
        self.KCfr = np.zeros(KCfollower_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCfc = np.zeros(KCfollower_SPARSE_SIZE*num_elements, dtype=INT)
        self.KCfv = np.zeros(KCfollower_SPARSE_SIZE*num_elements,
                             dtype=DOUBLE)
        # the dead part of the pressure, the force vector at u = 0
        self.fp0 = np.zeros(N, dtype=DOUBLE)
        for plate, p in zip(m.elements, self.pressures):
            update_fext_follower(np.zeros(N), plate, m.points, m.weights, p,
                                 self.fp0, quadratic)

    def expand(self, c):
        """The displacements of the model from those of the unknown dofs."""
        u = np.zeros(self.m.N, dtype=DOUBLE)
        u[self.bu] = c
        return u

    def calc_fext(self, inc=1., silent=False):
        return inc*self.fp0[self.bu]

    def calc_fint(self, c, inc=1., silent=False):
        m = self.m
        u = self.expand(c)
        fint = np.zeros(m.N, dtype=DOUBLE)
        for plate, p in zip(m.elements, self.pressures):
            update_fint(u, plate, m.points, m.weights, fint)
            # minus the part of the follower pressure that depends on u
            update_fext_follower(u, plate, m.points, m.weights, -inc*p, fint,
                                 self.quadratic, 0)
        return fint[self.bu]

    def calc_kC(self, c=None, NLgeom=False, inc=1., silent=False):
        bu = self.bu
        if not NLgeom:
            return self.KC0[bu, :][:, bu]
        m = self.m
        u = self.expand(c)
        self.KCNLv *= 0
        self.KCfv *= 0
        for plate, p in zip(m.elements, self.pressures):
            update_KCNL(u, plate, m.points, m.weights, self.KCNLr,
                        self.KCNLc, self.KCNLv)
            # the load stiffness matrix of the pressure inc*p
            update_KCfollower(u, plate, m.points, m.weights, inc*p,
                              self.KCfr, self.KCfc, self.KCfv,
                              self.quadratic)
        KCNL = coo_matrix((self.KCNLv, (self.KCNLr, self.KCNLc)),
                          shape=(m.N, m.N))
        KCf = coo_matrix((self.KCfv, (self.KCfr, self.KCfc)),
                         shape=(m.N, m.N))
        return (self.KC0 + KCNL + KCf).tocsc()[bu, :][:, bu]

    def calc_kG(self, c=None, NLgeom=False, silent=False):
        m = self.m
        u = self.expand(c)
        self.KGv *= 0
        for plate in m.elements:
            update_KG(u, plate, m.points, m.weights, self.KGr, self.KGc,
                      self.KGv)
        KG = coo_matrix((self.KGv, (self.KGr, self.KGc)), shape=(m.N, m.N))
        return KG.tocsc()[self.bu, :][:, self.bu]


@pytest.mark.parametrize('quadratic', TRUNCATIONS)
def test_cantilever_large_deflection_against_panels(quadratic):
    r"""The square isotropic cantilever of ``test_follower_pressure.py``,
    clamped at `x = 0`, the other edges free, under a uniform follower
    pressure pushing along `+z`, in five equal load increments, against
    panels within 1e-3 for the tip deflection `w` and the tip displacement
    `u`"""
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
    tip = np.where(isclose(m.x, a) & isclose(m.y, b/2))[0][0]

    callables = FollowerCallables(m, bu, func, quadratic)
    an = Analysis(calc_fext=callables.calc_fext,
                  calc_fint=callables.calc_fint,
                  calc_kC=callables.calc_kC,
                  calc_kG=callables.calc_kG)
    an.initialInc = 0.2
    an.maxInc = 0.2
    increments, cs = an.static(NLgeom=True)
    # every increment converged, none was cut
    assert np.allclose(increments, [0.2, 0.4, 0.6, 0.8, 1.]), increments
    u = callables.expand(cs[-1])

    w_tip = u[DOF*tip + 6]/h
    u_tip = u[DOF*tip]
    print('w/h', w_tip, 'u', u_tip)
    assert isclose(w_tip, PANELS[quadratic]['w'], rtol=1.e-3)
    assert isclose(u_tip, PANELS[quadratic]['u'], rtol=1.e-3)


if __name__ == '__main__':
    for quadratic in TRUNCATIONS:
        test_cantilever_large_deflection_against_panels(quadratic)

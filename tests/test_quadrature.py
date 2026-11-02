r"""Gauss-Legendre points and weights of get_points_weights

For every ``nint`` from 2 to 10, the quadrature of the cubic polynomial
`\xi^3 + \xi^2` from ``-1`` to ``1`` with the points and weights of
:func:`bfscplate2d.quadrature.get_points_weights` is checked against its
exact integral.

Notes
-----
The exact integral is

.. math::

    \int_{-1}^{1} \left( \xi^3 + \xi^2 \right) d\xi = \frac{2}{3}

and the Gauss-Legendre rule with ``nint`` points integrates exactly the
polynomials of degree up to `2 \, nint - 1`, i.e. at least 3 for
``nint >= 2``, such that the quadrature must reproduce `2/3` to round-off
for every ``nint`` tested.

"""
import sys
sys.path.append('..')

import numpy as np

from bfscplate2d.quadrature import get_points_weights

def test_quadrature():
    func = lambda xi: xi**3 + xi**2
    func_int_an = lambda xi: 1/4*xi**4 + 1/3*xi**3
    int_an = func_int_an(+1) - func_int_an(-1)
    for nint in range(2, 11):
        points, weights = get_points_weights(nint)
        int_num = 0
        for xi, wi in zip(points, weights):
            int_num += wi*func(xi)
        assert np.isclose(int_an, int_num)


if __name__ == '__main__':
    test_quadrature()



"""
Follower pressure for BFSC plate 2D: force vector and load stiffness matrix

The pressure ``pressure`` (force per unit undeformed area, a function of the
undeformed coordinates) acts on the area vector of the deformed mid-surface

    dW_p = int p a.du dx dy,    a = X,x x X,y,    X = X0 + u

    A = X,x = (1 + u,x,  v,x,  w,x)
    B = X,y = (u,y,  1 + v,y,  w,y)

the plate case ``kw = kv = 0`` of
``panels/theory/shells/follower_pressure/follower_pressure.md``. With ``a =
a0 + a1 + a2``, of orders 0, 1 and 2 in the displacement gradients,

    a0 = (0, 0, 1)
    a1 = (-w,x,  -w,y,  u,x + v,y)
    a2 = (v,x w,y - w,x v,y,  w,x u,y - u,x w,y,  u,x v,y - v,x u,y)

``ref`` multiplies ``a0`` (``reference`` of the .pyx) and ``qf`` multiplies
``a2`` (``quadratic`` of the .pyx). The force vector and the load stiffness
are

    F_p = int p (Nu^T a_x + Nv^T a_y + Nw^T a_z) dA
    KCfollower = -dF_p/due

KCfollower is unsymmetric in general and is printed in full. Its sparse
layout is that of ``qf = 1``, such that ``KCfollower_SPARSE_SIZE`` does not
depend on the truncation; with ``qf = 0`` the [u, v] and [v, u] blocks are
written as zeros.

The rows of the interpolation are the operators ``Nu``, ``Nv`` and ``Nw`` of
three rows each: row 1 the function, row 2 its derivative with respect to
`x` and row 3 with respect to `y`, e.g. ``Nw2_08`` is the entry of `N_{w,x}`
at the 8th degree of freedom of the element.
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix

num_nodes = 4
DOF = 10


def main():
    var('xi, eta, lex, ley, weight, pressure, qf, ref')
    var('ux, uy, vx, vy, wx, wy')
    var('ax, ay, az')

    ONE = sympy.Integer(1)

    # shape functions
    # - from Reference:
    #     OCHOA, O. O.; REDDY, J. N. Finite Element Analysis of Composite Laminates. Dordrecht: Springer, 1992.
    # cubic
    Hi = lambda xii, etai: ONE/16.*(xi + xii)**2*(xi*xii - 2)*(eta+etai)**2*(eta*etai - 2)
    Hxi = lambda xii, etai: -lex/32.*xii*(xi + xii)**2*(xi*xii - 1)*(eta + etai)**2*(eta*etai - 2)
    Hyi = lambda xii, etai: -ley/32.*(xi + xii)**2*(xi*xii - 2)*etai*(eta + etai)**2*(eta*etai - 1)
    Hxyi = lambda xii, etai: lex*ley/64.*xii*(xi + xii)**2*(xi*xii - 1)*etai*(eta + etai)**2*(eta*etai - 1)

    # node 1 (-1, -1)
    # node 2 (+1, -1)
    # node 3 (+1, +1)
    # node 4 (-1, +1)

    Nu = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
        Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
        ]])
    Nv = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
        0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
        ]])
    Nw = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
        0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
        ]])

    # interpolation operators, rows: function, d/dx, d/dy
    def operator(name, N):
        op = Matrix([N, (2/lex)*N.diff(xi), (2/ley)*N.diff(eta)])
        ops = []
        print()
        for i in range(op.shape[0]):
            opis = []
            for j in range(op.shape[1]):
                opij = op[i, j]
                if opij != 0:
                    print('                %s%d_%02d = %s' % (name, i+1, j+1, str(opij)))
                    opis.append(symbols('%s%d_%02d' % (name, i+1, j+1)))
                else:
                    opis.append(0)
            ops.append(opis)
        print()
        return Matrix(ops)

    Nus = operator('Nu', Nu)
    Nvs = operator('Nv', Nv)
    Nws = operator('Nw', Nw)

    # displacement gradients at the integration point
    ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, num_nodes*DOF)])
    grads = {
        'ux': (Nus[1, :]*ue)[0],
        'uy': (Nus[2, :]*ue)[0],
        'vx': (Nvs[1, :]*ue)[0],
        'vy': (Nvs[2, :]*ue)[0],
        'wx': (Nws[1, :]*ue)[0],
        'wy': (Nws[2, :]*ue)[0],
        }
    for name in ('ux', 'uy', 'vx', 'vy', 'wx', 'wy'):
        print('%s =' % name, grads[name])

    # area vector of the deformed mid-surface, plate: kw = kv = 0
    A = Matrix([1 + ux, vx, wx])
    B = Matrix([uy, 1 + vy, wy])
    a_full = A.cross(B)
    a0 = Matrix([0, 0, 1])
    a1 = Matrix([-wx, -wy, ux + vy])
    a2 = (a_full - a0 - a1).expand()
    assert (a2 - Matrix([vx*wy - wx*vy,
                         wx*uy - ux*wy,
                         ux*vy - vx*uy])).expand() == sympy.zeros(3, 1)
    a = ref*a0 + a1 + qf*a2
    print()
    print('ax =', a[0])
    print('ay =', a[1])
    print('az =', a[2])

    # force vector, with ax, ay, az computed once per integration point
    print()
    fe = weight*(lex*ley)/4.*pressure*(Nus[0, :].T*ax + Nvs[0, :].T*ay
                                       + Nws[0, :].T*az)

    def name_ind(i):
        if i >=0 and i < DOF:
            return 'c1'
        elif i >= DOF and i < 2*DOF:
            return 'c2'
        elif i >= 2*DOF and i < 3*DOF:
            return 'c3'
        elif i >= 3*DOF and i < 4*DOF:
            return 'c4'
        else:
            raise

    for i, fi in enumerate(fe):
        if fi == 0:
            continue
        si = name_ind(i)
        print('fext[%d + %s] +=' % (i%DOF, si), fi)

    # load stiffness, chain rule through the displacement gradients
    gradsyms = Matrix([ux, uy, vx, vy, wx, wy])
    dgrad = Matrix([Nus[1, :], Nus[2, :], Nvs[1, :], Nvs[2, :],
                    Nws[1, :], Nws[2, :]])
    da = a.jacobian(gradsyms)*dgrad
    KCfollower = -weight*(lex*ley)/4.*pressure*(Nus[0, :].T*da[0, :]
                                               + Nvs[0, :].T*da[1, :]
                                               + Nws[0, :].T*da[2, :])

    print()
    print('printing code for sparse implementation')
    for ind, val in np.ndenumerate(KCfollower):
        if val == 0:
            continue
        print('                k += 1')
        print('                KCfollowerv[k] +=', KCfollower[ind])
    print()
    print()
    print()
    KCfollower_SPARSE_SIZE = 0
    for ind, val in np.ndenumerate(KCfollower):
        if val == 0:
            continue
        KCfollower_SPARSE_SIZE += 1
        i, j = ind
        si = name_ind(i)
        sj = name_ind(j)
        print('        k += 1')
        print('        KCfollowerr[k] = %d+%s' % (i%DOF, si))
        print('        KCfollowerc[k] = %d+%s' % (j%DOF, sj))
    print('KCfollower_SPARSE_SIZE', KCfollower_SPARSE_SIZE)

if __name__ == '__main__':
    main()

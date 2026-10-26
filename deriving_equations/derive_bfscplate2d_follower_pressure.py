r"""
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

    F_p = int p (Su^T a_x + Sv^T a_y + Sw^T a_z) dA
    KCfollower = -dF_p/due

KCfollower is unsymmetric in general and is printed in full. Its sparse
layout is that of ``qf = 1``, such that ``KCfollower_SPARSE_SIZE`` does not
depend on the truncation; with ``qf = 0`` the [u, v] and [v, u] blocks are
written as zeros.

The rows of the interpolation are the operators ``Su``, ``Sv`` and ``Sw`` of
three rows each: row 1 the function, row 2 its derivative with respect to
`x` and row 3 with respect to `y`, e.g. ``Sw2_08`` is the entry of
`\boldsymbol{S}^w_{,x}` at the 8th degree of freedom of the element.
"""
import numpy as np
import sympy
from sympy import var, symbols, Matrix

num_nodes = 4
DOF = 10


def main():
    var('xi, eta, lex, ley, weight, pressure, qf, ref')
    var('u_x, u_y, v_x, v_y, w_x, w_y')
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

    Su = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0, 0, 0, 0,
        Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0, 0, 0, 0,
        Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0, 0, 0, 0,
        ]])
    Sv = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), 0, 0, 0, 0,
        0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), 0, 0, 0, 0,
        0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), 0, 0, 0, 0,
        ]])
    Sw = Matrix([[
       #u, du/dx, du/dy, v, dv/dx, dv/dy, w, dw/dx, dw/dy, d2w/(dxdy)
        0, 0, 0, 0, 0, 0, Hi(-1, -1), Hxi(-1, -1), Hyi(-1, -1), Hxyi(-1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, -1), Hxi(+1, -1), Hyi(+1, -1), Hxyi(+1, -1),
        0, 0, 0, 0, 0, 0, Hi(+1, +1), Hxi(+1, +1), Hyi(+1, +1), Hxyi(+1, +1),
        0, 0, 0, 0, 0, 0, Hi(-1, +1), Hxi(-1, +1), Hyi(-1, +1), Hxyi(-1, +1),
        ]])

    # interpolation operators, rows: function, d/dx, d/dy
    def operator(name, S):
        op = Matrix([S, (2/lex)*S.diff(xi), (2/ley)*S.diff(eta)])
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

    Sus = operator('Su', Su)
    Svs = operator('Sv', Sv)
    Sws = operator('Sw', Sw)

    # displacement gradients at the integration point
    ue = Matrix([symbols(r'ue[%d]' % i) for i in range(0, num_nodes*DOF)])
    grads = {
        'u_x': (Sus[1, :]*ue)[0],
        'u_y': (Sus[2, :]*ue)[0],
        'v_x': (Svs[1, :]*ue)[0],
        'v_y': (Svs[2, :]*ue)[0],
        'w_x': (Sws[1, :]*ue)[0],
        'w_y': (Sws[2, :]*ue)[0],
        }
    for name in ('u_x', 'u_y', 'v_x', 'v_y', 'w_x', 'w_y'):
        print('%s =' % name, grads[name])

    # area vector of the deformed mid-surface, plate: kw = kv = 0
    A = Matrix([1 + u_x, v_x, w_x])
    B = Matrix([u_y, 1 + v_y, w_y])
    a_full = A.cross(B)
    a0 = Matrix([0, 0, 1])
    a1 = Matrix([-w_x, -w_y, u_x + v_y])
    a2 = (a_full - a0 - a1).expand()
    assert (a2 - Matrix([v_x*w_y - w_x*v_y,
                         w_x*u_y - u_x*w_y,
                         u_x*v_y - v_x*u_y])).expand() == sympy.zeros(3, 1)
    a = ref*a0 + a1 + qf*a2
    print()
    print('ax =', a[0])
    print('ay =', a[1])
    print('az =', a[2])

    # force vector, with ax, ay, az computed once per integration point
    print()
    fe = weight*(lex*ley)/4.*pressure*(Sus[0, :].T*ax + Svs[0, :].T*ay
                                       + Sws[0, :].T*az)

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
    gradsyms = Matrix([u_x, u_y, v_x, v_y, w_x, w_y])
    dgrad = Matrix([Sus[1, :], Sus[2, :], Svs[1, :], Svs[2, :],
                    Sws[1, :], Sws[2, :]])
    da = a.jacobian(gradsyms)*dgrad
    KCfollower = -weight*(lex*ley)/4.*pressure*(Sus[0, :].T*da[0, :]
                                               + Svs[0, :].T*da[1, :]
                                               + Sws[0, :].T*da[2, :])

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

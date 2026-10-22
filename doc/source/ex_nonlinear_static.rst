Nonlinear static analysis
=========================

.. currentmodule:: bfscplate2d.bfscplate2d

A rectangular isotropic plate, with `w` constrained at the nodes of the
four edges, compressed along `x` below its critical load, with a transverse force at the centre large enough to reach a
deflection of about 2.6 times the thickness. The model keeps the value
arrays of `[K_{CNL}]` and `[K_G]`, which are zeroed and recomputed at each
iteration with :func:`.update_KCNL` and :func:`.update_KG`, and the
internal forces of :func:`.update_fint`:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: build_model

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: calc_KT

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: calc_fint

The full Newton-Raphson method, with the tangent stiffness matrix rebuilt at
every iteration and the external forces applied in increments:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: newton_raphson

The test checks the quadratic convergence and that the membrane stretching
makes the plate stiffer than the linearised prediction
`([K_{C0}] + [K_G])^{-1}\{f\}`, with `[K_G]` of the axial load:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: test_large_deflection

Nonlinear static analysis with structsolve
==========================================

.. currentmodule:: bfscplate2d.bfscplate2d

The plate of :doc:`ex_nonlinear_static`, solved with ``structsolve.Analysis``
of `structsolve <https://saullocastro.github.io/structsolve/>`_, see
:ref:`solving-with-structsolve`. The four callables loop over
the elements with the functions of :doc:`ex_nonlinear_static` and work on
the unknown degrees of freedom ``bu``, the partition of the free degrees of
freedom; ``calc_fint`` also keeps the internal forces of each iteration:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: Callables

The analysis applies the load in equal increments, with
``Analysis.initialInc`` and ``Analysis.maxInc``, checks that none was cut,
and computes Crisfield's relative residual, the convergence criterion of
structsolve, at each iteration of the last increment:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: newton_raphson

A consistent tangent stiffness matrix squares the residual at the last
iteration:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: assert_quadratic_convergence

With a small transverse force the solution in one increment reproduces the
linearised one, `([K_{C0}] + [K_G])^{-1}\{f\}`, with `[K_G]` of the axial
load:

.. literalinclude:: ../../tests/test_static_axial_load_nonlinear.py
    :pyobject: test_static

Linear static analysis
======================

.. currentmodule:: bfscplate2d.bfscplate2d

A rectangular isotropic plate, with `w` constrained at the nodes of the four
edges, under a uniform pressure. The stiffness matrix is assembled with
:func:`.update_KC0` and the consistent load vector of the pressure is
integrated element by element with the rows of
:meth:`.BFSCPlate2D.update_Sw`; `u` and `v` are constrained at the central
node only, which removes the in-plane rigid-body motions. The code is
extracted from one of the ``bfscplate2d`` unit tests:

.. literalinclude:: ../../tests/test_static_pressure.py
    :pyobject: test_static_pressure

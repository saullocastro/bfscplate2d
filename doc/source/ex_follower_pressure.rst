Follower pressure
=================

.. currentmodule:: bfscplate2d.bfscplate2d

The tests of the follower pressure use a mesh class with the assembly of
the force vector of :func:`.update_fext_follower`, of the load stiffness
matrix of :func:`.update_KCfollower`, and of the internal forces and the
tangent stiffness matrix. The method ``pressure`` gives the array ``p`` of
each element at its integration points:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: Mesh

Large deflection of a cantilever plate
--------------------------------------

The Newton-Raphson method with the residual `\boldsymbol{F}_{int}(\boldsymbol{u}) - \boldsymbol{F}_p(\boldsymbol{u})`
of the pressure of the load factor ``lbd`` and the tangent stiffness matrix
`\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_G + \boldsymbol{K}_F`, or with the dead pressure
`\boldsymbol{F}_p(\boldsymbol{0})` for comparison:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: newton_raphson

A square isotropic plate clamped at `x = 0`, the other edges free, under a
uniform pressure. The load stiffness matrix is unsymmetric because of the
free edges, the follower pressure gives a tip deflection measurably larger
than the dead pressure, and the two truncations of the area vector give the
same result:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: boundary_flags

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: test_cantilever_large_deflection_against_panels

Tangent stiffness matrix
------------------------

`\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_G + \boldsymbol{K}_F` against the finite
differences of the residual, for both truncations:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: test_tangent_is_jacobian_of_residual

Symmetry of the load stiffness matrix
-------------------------------------

The conditions of :doc:`follower_pressure` under which the load stiffness
matrix is symmetric:

.. literalinclude:: ../../tests/test_follower_pressure.py
    :pyobject: test_symmetry

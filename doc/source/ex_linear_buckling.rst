Linear buckling analysis
========================

.. currentmodule:: bfscplate2d.bfscplate2d

Geometric stiffness matrix of a static solution
-----------------------------------------------

A square isotropic plate, with `w` constrained at the nodes of the four
edges and in-plane point supports at the middle of the edges, under a
uniaxial compression applied as a consistent edge load with the rows of
:meth:`.BFSCPlate2D.update_Nu` at `\xi = \pm 1`. The geometric stiffness
matrix of :func:`.update_KG` is computed with the linear static solution,
and the critical force per unit length is compared, within 5 %, with `k_c
\pi^2 D/b^2`, with the buckling coefficient `k_c = 4`:

.. literalinclude:: ../../tests/test_linear_buckling_iso_SSSS.py
    :pyobject: test_linear_buckling_iso_SSSS

Constant membrane forces
------------------------

A laminated plate, with `w` constrained at the nodes of the four edges,
under a uniform compressive force per unit length, ``Nxx
= -1``, given directly to :func:`.update_KG_cte_N` without a static
analysis, such that the load multipliers are the critical forces per unit
length:

.. literalinclude:: ../../tests/test_linear_buckling_cte_N.py
    :pyobject: test_linear_buckling

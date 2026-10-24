Natural frequencies
===================

.. currentmodule:: bfscplate2d.bfscplate2d

A rectangular isotropic plate, with `w` constrained at the nodes of the
four edges. The mass matrix of :func:`.update_M` requires the thickness
``h`` and the density ``rho`` of each element. The generalized eigenvalue
problem `[K_{C0}]\{\phi\} = \omega_n^2 [M]\{\phi\}` is solved with the
dense ``scipy.linalg.eigh`` and the first natural angular frequency is
compared with the reference value of the test, within 1 %:

.. literalinclude:: ../../tests/test_natural_frequency.py
    :pyobject: test_nat_freq

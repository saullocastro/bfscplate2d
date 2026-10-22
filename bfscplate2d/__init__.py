"""
BFSCPLATE2D - Implementation of the BFSC plate finite element in 2D

The ``bfscplate2d`` namespace gives the element class
:class:`.BFSCPlate2D` and the functions :func:`.update_KC0`,
:func:`.update_KCNL`, :func:`.update_KG`, :func:`.update_KG_cte_N`,
:func:`.update_M` and :func:`.update_fint` of the module
:mod:`bfscplate2d.bfscplate2d`, with its sizes ``KC0_SPARSE_SIZE``,
``KCNL_SPARSE_SIZE``, ``KG_SPARSE_SIZE`` and ``M_SPARSE_SIZE`` of the
sparse arrays that one element fills. The quadrature points and weights
used by the functions are given by
:func:`bfscplate2d.quadrature.get_points_weights`, which is not imported in
the namespace. It also defines:

- ``DOF = 10``: number of degrees of freedom per node;
- ``INT``: NumPy integer type of the C ``long`` of the platform,
  ``np.int64`` when a C ``long`` has 8 bytes and ``np.int32`` otherwise,
  matching the ``long`` arrays of row and column indices of the
  ``update_*`` functions;
- ``DOUBLE = np.float64``: NumPy type of the arrays of values.

Author: Saullo G. P. Castro

"""
import ctypes

import numpy as np


from .bfscplate2d import BFSCPlate2D, update_KC0, update_KCNL, update_KG, update_KG_cte_N, update_M, update_fint
from .bfscplate2d import KC0_SPARSE_SIZE, KCNL_SPARSE_SIZE, KG_SPARSE_SIZE, M_SPARSE_SIZE


if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
DOF = 10


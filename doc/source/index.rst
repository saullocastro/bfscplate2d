BFSC plate finite element - bfscplate2d
=======================================

The ``bfscplate2d`` module implements the BFSC plate finite element in 2D, a
four-node rectangular element of the classical laminated plate theory, with
the nonlinear strains of von Karman, aiming linear and nonlinear analyses.
The module gives the element, :class:`.BFSCPlate2D`, and the functions that
compute the contributions of one element to the matrices and vectors of a
model, written in sparse arrays; the model, its boundary conditions and the
solution of its equations are written by the user, with NumPy and SciPy.
With ``bfscplate2d`` you can compute:

* The linear constitutive stiffness matrix `[K_{C0}]`, for linear static
  analyses, `[K_{C0}]\{u\} = \{f\}`, with :func:`.update_KC0`

* The geometric stiffness matrix `[K_G]`, for linear buckling analyses,
  `([K_{C0}] + \lambda [K_G])\{u\} = \{0\}`, of the membrane forces of a
  static solution, :func:`.update_KG`, or of constant membrane forces,
  :func:`.update_KG_cte_N`

* The mass matrix `[M]`, for natural frequency analyses,
  `([K_{C0}] - \omega^2 [M])\{u\} = \{0\}`, with :func:`.update_M`

* The internal force vector `\{f_{int}\}` and the tangent stiffness matrix
  `[K_{C0}] + [K_{CNL}] + [K_G]`, its exact Jacobian, for geometrically
  nonlinear static analyses, with :func:`.update_fint` and
  :func:`.update_KCNL`

* The force vector and the load stiffness matrix of a follower pressure,
  normal to the deformed plate, with :func:`.update_fext_follower` and
  :func:`.update_KCfollower`


Code repository
---------------

https://github.com/saullocastro/bfscplate2d


User guide
----------

.. toctree::
    :maxdepth: 1

    element.rst
    assembling.rst
    analyses.rst
    follower_pressure.rst


Usage examples
--------------

.. toctree::
    :maxdepth: 1

    ex_linear_static.rst
    ex_linear_buckling.rst
    ex_natural_frequency.rst
    ex_nonlinear_static.rst
    ex_follower_pressure.rst


bfscplate2d API
---------------

.. toctree::
    :maxdepth: 2

    api.rst


Installing bfscplate2d
----------------------

.. toctree::
    :maxdepth: 1

    install.rst


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

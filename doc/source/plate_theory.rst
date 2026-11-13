Plate theory
============

.. currentmodule:: bfscplate2d.bfscplate2d

This chapter states the plate theory that every function of
:mod:`bfscplate2d.bfscplate2d` discretises: the classical laminated plate
theory with the nonlinear strains of von Kármán [Reddy2003]_. The element
that approximates it is the subject of :doc:`element`, and the matrices that
follow from it of :doc:`element_matrices`.


Coordinates and sign conventions
--------------------------------

The plate lies in the plane `x`-`y` of a right-handed Cartesian system, with
`z` normal to the plate. The reference surface is `z = 0`; the
displacements of its points along `x`, `y` and `z` are `u(x, y)`,
`v(x, y)` and `w(x, y)`. Subscripts after a comma denote partial
derivatives, `w_{,x} = \partial w/\partial x`, as in the names of the code
(``w_x``, ``Sw_x``, ...). With this convention:

- the membrane forces `\boldsymbol{N} = \{N_{xx}, N_{yy}, N_{xy}\}^\top`
  (``Nxx``, ``Nyy``, ``Nxy``), forces per unit length, are positive in
  tension;
- the moments `\boldsymbol{M} = \{M_{xx}, M_{yy}, M_{xy}\}^\top` (``Mxx``,
  ``Myy``, ``Mxy``), moments per unit length, are the first moments
  `\int \sigma z \, dz` of the stresses through the thickness;
- a positive transverse force or pressure acts along `+z`, like a positive
  `w`.

The library has no units: the lengths, the forces and the masses of a model
are given in a consistent system, e.g. N, m and kg, and every result is in
that system.


Kinematics
----------

The classical laminated plate theory keeps the Kirchhoff hypotheses: the
normals to the reference surface remain straight, unstretched and normal to
the deformed reference surface, i.e. without transverse shear strains
`\gamma_{xz}` and `\gamma_{yz}` and with `\epsilon_{zz} = 0`. The normal
rotates by `-w_{,x}` about `y` and by `-w_{,y}` about `x`, and the
displacements of a point at the distance `z` from the reference surface are
[Reddy2003]_

.. math::
    :label: pt-displacements

    u(x, y, z) = u - z w_{,x}, \qquad
    v(x, y, z) = v - z w_{,y}, \qquad
    w(x, y, z) = w

The strain-displacement relations of a plate contain the squares and
products of all the displacement gradients. For small strains and
moderate rotations, the assumptions of von Kármán, only the squares and
products of the slopes `w_{,x}` and `w_{,y}` are kept: the slopes may be
of the order of the square root of the strains, whereas the in-plane
gradients `u_{,x}`, `u_{,y}`, `v_{,x}`, `v_{,y}` are of the order of the
strains, and their squares are neglected. The strains at `z` are then
`\boldsymbol{\epsilon} + z \boldsymbol{\kappa}`, with the membrane strains
and the curvatures of the reference surface

.. math::
    :label: pt-strains

    \boldsymbol{\epsilon} =
    \begin{Bmatrix} \epsilon_{xx} \\ \epsilon_{yy} \\ \gamma_{xy} \end{Bmatrix}
    = \begin{Bmatrix} u_{,x} \\ v_{,y} \\ u_{,y} + v_{,x} \end{Bmatrix}
    + \begin{Bmatrix} w_{,x}^2/2 \\ w_{,y}^2/2 \\ w_{,x} w_{,y} \end{Bmatrix},
    \qquad
    \boldsymbol{\kappa} =
    \begin{Bmatrix} \kappa_{xx} \\ \kappa_{yy} \\ \kappa_{xy} \end{Bmatrix}
    = \begin{Bmatrix} -w_{,xx} \\ -w_{,yy} \\ -2 w_{,xy} \end{Bmatrix}

`\gamma_{xy}` is the engineering shear strain. The first vector of
`\boldsymbol{\epsilon}` is the linear membrane strain, the second the
nonlinear one, which couples the membrane and the bending behaviour: a
deflection stretches the reference surface, and membrane forces change the
bending stiffness. The curvatures are linear in `w`. Without the nonlinear
part, Eq. :eq:`pt-strains` is the linear theory, in which the membrane and
the bending problems are coupled only through the matrix `\boldsymbol{B}`
below.


Constitutive relation
---------------------

For a laminate of linear elastic plies, the integration of the plane-stress
relation of each ply through the thickness gives the membrane forces and
the moments [Reddy2003]_

.. math::
    :label: pt-abd

    \begin{Bmatrix} \boldsymbol{N} \\ \boldsymbol{M} \end{Bmatrix}
    = \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\
                      \boldsymbol{B} & \boldsymbol{D} \end{bmatrix}
      \begin{Bmatrix} \boldsymbol{\epsilon} \\ \boldsymbol{\kappa} \end{Bmatrix}

with the extensional stiffness `\boldsymbol{A}` (force per unit length),
the extension-bending coupling `\boldsymbol{B}` (force) and the bending
stiffness `\boldsymbol{D}` (force times length), three symmetric `3 \times 3`
matrices. The `6 \times 6` matrix is the ``ABD`` attribute of
:class:`.BFSCPlate2D`, in the order of Eq. :eq:`pt-abd`, e.g. the ``ABD``
attribute of a laminate of `composites
<https://github.com/saullocastro/composites>`__. The functions read only the entries
``A11``, ``A12``, ``A16``, ``A22``, ``A26``, ``A66`` on and above the
diagonal of `\boldsymbol{A}`, and likewise of `\boldsymbol{B}` and
`\boldsymbol{D}`; the matrix is taken as symmetric.

`\boldsymbol{B} = \boldsymbol{0}` for a laminate symmetric about the
reference surface, which then separates the linear membrane and bending
problems. A reference surface that is not the mid-surface, e.g. an offset
plate, enters only through `\boldsymbol{A}`, `\boldsymbol{B}` and
`\boldsymbol{D}` computed about that surface; the mass matrix, however,
assumes the mid-surface, see :doc:`element_matrices`. For an isotropic
plate of Young's modulus `E`, Poisson's ratio `\nu` and thickness `h`,
`\boldsymbol{B} = \boldsymbol{0}`, `A_{11} = E h/(1 - \nu^2)` and the
bending stiffness is

.. math::
    :label: pt-D-isotropic

    D_{11} = D = \frac{E h^3}{12 (1 - \nu^2)}


Strain energy and equilibrium
-----------------------------

The strain energy of the plate is

.. math::
    :label: pt-strain-energy

    U = \frac{1}{2} \int_A \left( \boldsymbol{\epsilon}^\top \boldsymbol{A}
        \boldsymbol{\epsilon} + 2 \boldsymbol{\epsilon}^\top \boldsymbol{B}
        \boldsymbol{\kappa} + \boldsymbol{\kappa}^\top \boldsymbol{D}
        \boldsymbol{\kappa} \right) dA
      = \frac{1}{2} \int_A \left( \boldsymbol{N}^\top \boldsymbol{\epsilon}
        + \boldsymbol{M}^\top \boldsymbol{\kappa} \right) dA

and its first variation, with `\delta\boldsymbol{\kappa}` linear in `\delta
w` and

.. math::
    :label: pt-strain-variation

    \delta\boldsymbol{\epsilon} =
    \begin{Bmatrix} \delta u_{,x} + w_{,x} \delta w_{,x} \\
                    \delta v_{,y} + w_{,y} \delta w_{,y} \\
                    \delta u_{,y} + \delta v_{,x} + w_{,x} \delta w_{,y}
                    + w_{,y} \delta w_{,x} \end{Bmatrix}

is the internal virtual work `\delta U = \int_A (\boldsymbol{N}^\top
\delta\boldsymbol{\epsilon} + \boldsymbol{M}^\top \delta\boldsymbol{\kappa})
\, dA`. The equilibrium of the plate is the principle of virtual work,
`\delta U = \delta W` for every admissible variation, with `\delta W` the
virtual work of the external forces. For loads that do not depend on the
displacements, `\delta W` is linear in the variations; a pressure that
follows the deformation makes it depend on the displacements, see
:doc:`follower_pressure`.

Two features of `\delta U` determine the element and its matrices:

- `\boldsymbol{\kappa}` contains the second derivatives of `w`, such that
  the energy is finite for an approximation of `w` only when `w` and its
  first derivatives are continuous over the plate, see :doc:`element`;
- the products `w_{,x} \delta w_{,x}`, ... of Eq.
  :eq:`pt-strain-variation` make the internal forces nonlinear in the
  displacements, and the derivative of the term `\boldsymbol{N}^\top
  \delta\boldsymbol{\epsilon}` with respect to the displacements at
  constant `\boldsymbol{N}` is the geometric stiffness, the term that
  governs buckling, see :doc:`element_matrices`.


Limits of validity
------------------

- **Thin plates.** Without transverse shear deformation the theory
  overestimates the stiffness, and the buckling loads and natural
  frequencies, of plates whose thickness is not small compared with the
  in-plane dimensions and with the wavelengths of the deformation, and of
  laminates or sandwich plates of low transverse shear stiffness.
- **Moderate rotations.** The strains of Eq. :eq:`pt-strains` hold for
  small strains, slopes `w_{,x}`, `w_{,y}` small compared with one, and
  small in-plane rotations. They describe the membrane stiffening of
  plates deflecting by several thicknesses, but not large rotations.
- **Linear elasticity.** The ``ABD`` matrix is constant: no plasticity,
  damage or temperature dependence.
- **Flat plates.** The reference surface is a plane; curvature of the
  undeformed surface, as in shells, is not represented.

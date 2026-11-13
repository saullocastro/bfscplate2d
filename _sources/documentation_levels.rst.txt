How the documentation is organised
==================================

The documentation of ``bfscplate2d`` is written in four cumulative levels,
each with its own place, so that a reader, or a tool, can go from what a
function does to why it does it without the levels repeating each other.

.. list-table::
    :header-rows: 1

    * - Level
      - Where
      - Content
    * - 1, functional
      - docstrings of the modules ``bfscplate2d.bfscplate2d`` and
        ``bfscplate2d.quadrature``, of the class :class:`.BFSCPlate2D`, of
        its methods and of the functions, except their ``Notes`` and
        ``References`` sections; comments in the code; docstrings and
        comments of the tests; the ``README.md``
      - what each resource does: a one-line summary, an extended summary
        (what, not why), ``Parameters`` with type, unit, shape and default,
        ``Returns`` and ``Attributes``; short usage ``Examples`` only for
        the entry points of an analysis, the element class, the ``update_*``
        functions of the matrices and vectors and
        :func:`.get_points_weights`
    * - 2, theory
      - the ``Notes`` and ``References`` sections of the docstrings,
        used only for theory
      - the theory that each resource implements, local to it: the
        equations in the notation of the code, the assumptions, the
        integration rule, the meaning of each parameter in that context,
        and the published sources
    * - 3, theory in a larger context
      - the pages of the Theory section of this documentation, from
        :doc:`plate_theory` to :doc:`analyses`, with the
        :doc:`references`, and :doc:`install`
      - the plate theory, the element and its interpolation, the element
        matrices, the follower pressure, the assembly with loads and
        boundary conditions, and the analyses, like the chapters of a
        book, with links to the API instead of repeating level 2
    * - 4, worked examples
      - the notebooks in ``notebooks/`` at the root of the repository,
        rendered in the Worked examples section
      - complete analyses explained step by step: linear static, linear
        buckling, natural frequencies, nonlinear static and follower
        pressure, with plots and tables of the results against the
        literature, with stored outputs

The API pages render levels 1 and 2 together. To find the theory of a
resource, read the ``Notes`` of its docstring; to see how the resources fit
in a model, read the pages of level 3, starting from :doc:`plate_theory`.
The derivation scripts of the element matrices, in ``deriving_equations/``,
and the changelog are not part of these levels.


The tests
---------

The tests in ``tests/`` are part of level 1 (their docstrings and comments)
and of level 2 (the ``Notes`` and ``References`` of their docstrings). Each
test states in its docstring what it verifies and against what:

* a value of the literature, named in the summary, e.g. the centre
  deflection of a simply supported plate under a point load of Timoshenko
  and Woinowsky-Krieger (1959), Eq. (147), with the equation, the full
  reference and the mesh convergence that justifies the tolerance in the
  ``Notes``;
* a closed-form solution, e.g. the natural frequencies of the simply
  supported plate, with its source;
* a regression value, a stored result of the element, with the comparison
  with the literature, when there is one, in the ``Notes``;
* a property that holds for any input, e.g. the tangent stiffness matrix
  being the derivative of the internal force vector, or the force vector
  of a follower pressure at the undeformed state being that of a dead
  pressure.

The notebooks of level 4 repeat the analyses of some tests in their
theoretical context, as top-level code with the results printed and
plotted; they do not import the tests.

Installing bfscplate2d
======================

From the distributed packages
-----------------------------

Install from the distributed packages, with the dependencies NumPy, SciPy
and `composites <https://github.com/saullocastro/composites>`_, by simply
doing::

    python -m pip install bfscplate2d


From the source code
--------------------

The element is written in Cython, translated to C++ and compiled when the
package is built, such that installing from the source code requires a C++
compiler, e.g. Microsoft Visual C++ on Windows, GCC on Linux or Clang on
macOS. The package metadata, with the version, and the build requirements,
setuptools and Cython, are in ``pyproject.toml``; ``pip`` installs the build
requirements in an isolated environment::

    python -m pip install .

The requirements of the tests and of the build, including Cython and pytest,
are listed in ``requirements.txt``::

    python -m pip install -r requirements.txt

``requirements.txt`` also lists `structsolve
<https://saullocastro.github.io/structsolve/>`_, which solves the nonlinear
static analysis of the tests and of the examples and cross-checks some of
their linear buckling and natural frequency solutions, see
:doc:`analyses`. It is a requirement of the tests and of the examples of
the documentation only: bfscplate2d does not import it, and it is not
installed with bfscplate2d, not being in the ``dependencies`` of
``pyproject.toml``. The tests require a version of structsolve newer than 0.8.0,
with the module ``structsolve.convergence`` and Crisfield's convergence
criterion as default.

To build in place, for running the tests from the source tree, which also
writes ``bfscplate2d/version.py`` with the version of ``pyproject.toml``::

    python setup.py build_ext --inplace

and the tests with::

    python -m pytest tests

``setup.py`` compiles with ``/O2`` on Windows and with ``-O3
-fno-math-errno`` on Linux and macOS, without flags that change the
floating-point results, e.g. ``-ffast-math``, or that tie the binaries to
the processor that built them, e.g. ``-march=native``. On Linux it also
links with ``-static-libgcc -static-libstdc++``.

A build for the coverage of the Cython code, with line tracing and without
optimization, is requested with ``CYTHON_TRACE_NOGIL`` in the environment or
with::

    python setup.py build_ext --inplace --force --define CYTHON_TRACE

as in ``run_setup_coverage.bat``. ``setup.py`` regenerates the C++ file when
it was generated in the other mode, with or without line tracing.


Building the documentation
--------------------------

The documentation requires Sphinx, numpydoc, nbsphinx and the PyData Sphinx
theme, and the package built in place, see above::

    python -m pip install sphinx numpydoc nbsphinx pydata-sphinx-theme
    cd doc
    make html

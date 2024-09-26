.. _contribtest:

Tests
=====

C++ tests are located in ``/tests`` and Python (binding) tests in
``/python/test``. See also the documentation on :ref:`building <building>` for
the C++ tests and ``/python/test/readme.md`` for the latter.

Building and Running Tests
--------------------------

The C++ tests need to be built using the usual proecdure via CMake + Ninja.
It is usually advisable to enable assertions for debugging.

.. code-block:: sh

    # more arguments omitted
    cmake .. -DCMAKE_BUILD_TYPE=debug -DARB_WITH_ASSERTIONS=ON [..]
    ninja tests

This will produce three binaries

- ``bin/unit`` basic unit tests for Arbor.
- ``bin/unit-modcc`` unit tests for Arbor's NMODL compiler
- ``bin/unit-mpi`` unit tests for Arbor's MPI funcionality. Only produced if MPI
  enabled.
- ``bin/unit-local`` local (i.e. without MPI) version of ``unit-mpi``.

All accept some arguments determined by the testing framework, among others
filters to include/exclude tests, try ``bin/unit --help`` for more information.
Some tests might be skipped if GPU-support is not enabled. Building with GPU and
failing to locate a GPU will register as failures. Except ``unit-mpi`` each
testsuite can be run like a normal executable; for MPI tests use

.. code-block:: sh

    mpirun -n 2 bin/unit-mpi

If you are working on the MPI parts of Arbor, don't forget to try different
process counts.

There also is a collection of micro benchmarks; build these with ``ninja
ubenches``. Each benchmark will result in a separate executable which measures
the performance of a critical building block of Arbor.

For Python, two more testsuites are provided in ``python/test``. During
development these can be run like this

.. code-block:: sh

  # Assuming we have built Arbor in ~/src/arbor/build
  # and Arbor was cloned into ~/src/arbor
  PYTHONPATH=$HOME/src/arbor/build/python python3 -munittest discover -v -s $HOME/src/arbor/python/

This will use Arbor's Python module without installing. If you did install Arbor
beforehand

.. code-block:: sh

  # Assuming Arbor was cloned into ~/src/arbor
  python3 -munittest discover -v -s $HOME/src/arbor/python/

is sufficient. However, this requires an install step after each change in
Arbor.

What to test?
-------------

Adding a feature should be accompanied by tests to ensure its functionality is
sound. That means
1. identifying the core ideas of your feature;
2. finding the functions/classes that encapsulate these ideas;
3. adding a test case for each that covers it.

The core motivation is to capture the essence of your feature and to protect it
against accidental change. This is what enables us to freely add optimisations,
and refactor code as needed.

Example
^^^^^^^

This example might touch parts of Arbor that you are unfamiliar with. Don't
Panic! The details are less important than the general approach to adding tests.
Imagine adding a new feature that is intended to improve performance during the
communication step. Spikes should be transferred pre-sorted to avoid locally
sorting them after the exchange. Also, assume, just for the sake of this example,
that you decided to add your own radix sort algorithm, since you expect it to be
faster for this particular use case.

Thus, the tests added should be

1. sorting algorithm

  - the application of `sort` sorts the given array. This seems trivial, but is
    really the core of what you are doing!
  - corner cases like: empty array, all elements equal, ... are treated gracefully.
  - if the sort is intended to be stable, check that equal elements do not switch order.

2. local sorting

  - spikes are -- after sorting -- ordered by their source ids and in case of ties by time.
  - corner cases: NaN, negative numbers, ...

3. global sorting

  - after the MPI exchange, each sub-array is still sorted
  - by the guarantees of ``MPI_Allgather``, the global array is sorted

Note that we added tests that are only applicable when, e.g., MPI is enabled. Our test
runners probe the different combinations automatically, see below.

Next, we would ask you to prove that this change does as promised, ie, it
improves performance. When adding a new user-facing feature, also consider
adding an example showing off your cool new addition to Arbor.

Regression tests
^^^^^^^^^^^^^^^^

However, it's impossible to foresee every dark corner of your code. Inevitably,
bugs will occur. When fixing a bug, please add a test case that covers this
particular sequence of events to catch this bug in the future (imagine someone
inadvertently removing your fix).

C++ tests
---------

We are using the GTest library for writing tests. Each group of tests should be
contained in a ``.cpp`` file in ``test/unit`` (do not forget to add it to the
``CMakeLists.txt``!). To get access to the library and a battery of helpers
including ``common.hpp``. Test cases are defined via the ``TEST`` macro, which takes
two arguments ``group`` and ``case``. Inside cases macros like ``ASSERT_TRUE``
can be used. Another helpful feature is that the test executable accepts
arguments on the command line. Of these, we would like to point out:

- ``--gtest_catch_exceptions`` allows for disabling exception catching by the
  framework. Handy when running the tests in a debugger.
- ``--gtest_throw_on_failure`` turns missed assert into exceptions, likewise
  useful in a debugger
- ``--gtest_filter`` to filter the tests to run. Can cut down the roundtrip time
  when working on a specific feature.

For more information on GTest refer to the `documentation
<https://google.github.io/googletest/>`_` and our existing tests.

Python tests
------------

The Python tests use the `unittest
<https://docs.python.org/3/library/unittest.html>`_ and its test discovery
mechanism. For tests to be discovered, they must meet the following criteria:

* Located in an importable code folder starting from the ``python/test`` root.
  If you introduce subfolders, they must all contain a ``__init__.py`` file.
* The filenames must start with ``test_``.
* The test case classes must begin with ``Test``.
* The test functions inside the cases must begin with ``test_``.

To run the tests locally, use ``python -m unittest`` from the ``python`` directory.

Fixtures
^^^^^^^^

Multiple tests may require the same reusable piece of test setup to run. You
can speed up the test writing process for everyone by writing these reusable
pieces as a `fixture <https://en.wikipedia.org/wiki/Test_fixture#Software>`_.
A fixture is a decorator that injects the reusable piece into the test
function. Fixtures and helpers to write them are available in
``python/test/fixtures.py``. The following example shows you how to create
a fixture that returns the Arbor version and optionally the path to it:

.. code-block:: python

  import arbor

  # This decorator converts your function into a fixture decorator.
  @_fixture
  def arbor_info(return_path=False):
    if return_path:
      return (arbor.__version__, arbor.__path__)
    else:
      return (arbor.__version__,)

Whenever you are writing a test, you can now apply your fixture by calling it
with the required parameters and adding a parameter to your function with the
same name as the fixture:

.. code-block:: python

  # Import fixtures.py
  from .. import fixtures

  @fixtures.arbor_info(return_path=True)
  def test_up_to_date(arbor_info):
    ...


Feature dependent tests
-----------------------

Certain tests need to be guarded by feature flags, notably ``ARB_MPI_ENABLED``
and ``ARB_GPU_ENABLED``. Another important (**especially** when dealing with
mechanisms, modcc, and the ABI) but less obvious feature is SIMD. The
combinations arising from the cartesian product of OS=Linux|MacOS x SIMD=ON|OFF
x MPI=ON|OFF is tested automatically on GitHub CI. As no instances with GPUs are
provided, GPU features are tested via CSCS' GitLab. Such a run is initiated by
commenting ``bors try`` in the PR discussion.

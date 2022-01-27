.. _contribtest:

Tests
=====

C++ tests are located in ``/tests`` and Python (binding) tests in 
``/python/test``. See the documentation on :ref:`building <building>` for the
C++ tests and ``/python/test/readme.md`` for the latter.

Python tests
------------

The Python tests uses the `unittest
<https://docs.python.org/3/library/unittest.html>`_ and its test discovery
mechanism. For tests to be discovered they must meet the following criteria:

* Located in an importable code folder starting from the ``python/test`` root.
  If you introduce subfolders they must all contain a ``__init__.py`` file.
* The filenames must start with ``test_``.
* The test case classes must begin with ``Test``.
* The test functions inside the cases must begin with ``test_``.

To run the tests locally use `python -m unittest` from the `python` directory.

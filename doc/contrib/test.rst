.. _contribtest:

Tests
=====

C++ tests are located in ``/tests`` and Python (binding) tests in ``/python/test``.
See the documentation on :ref:`building <building>` for the C++ tests and ``/python/test/readme.md`` for the latter.

Python tests
============

The Python tests uses the `unittest
<https://docs.python.org/3/library/unittest.html>`_ and it's test discovery
mechanism. For tests to be discovered:

* The files have to be located in an importable code folder starting from the
  ``python/test`` root. If you introduce subfolders they must all have a
  ``__init__.py`` file.
* The filenames must start with ``test_``.
* The test case classes must begin with ``Test``.
* The test functions inside the cases must begin with ``test_``.

.. _pyunittest:

Python Unit Testing
====================

In order to test individual units of the python front end during development with `pybind11 <https://pybind11.readthedocs.io/en/stable/intro.html>`_, python's `unittest <https://docs.python.org/3/library/unittest.html>`_ is deployed.

Directory Structure
-------------------

The predefined directory structure in the ``python/test`` folder

* ``test\``
    * ``options.py``
    * ``unit\``
        * ``runner.py``
    * ``unit-distributed\``
        * ``runner.py``

is the entry point to file and run the unit tests.

In ``python/test`` an ``options.py`` file is located to handle (global) command line options such as ``-v{0,1,2}`` to increase the verbosity.
Further, in ``python/test/unit`` all serial unit tests ``test_xxxs.py`` are stored defining unittest classes with test methods and its own test suite, whereas in ``python/test/unit_distributed`` all distributed/ parallel unit tests are located.

In each subfolder ``python/test/unit`` and ``python/test/unit_distributed`` a ``runner.py`` execution file/ module is defined to run all tests included in the test suite of the respective subfolder.

Testing
--------

The unit tests are started in the respective subfolder ``python/test/unit`` for serial tests and ``python/test/unit_distributed`` for tests related to a distributed execution.

In the respective folder the module ``runner`` is used to start the test.


.. container:: example-code

    .. code-block:: bash

        $ python -m runner

        ...
        ----------------------------------------------------------------------
        Ran 3 tests in 0.001s

        OK

The in ``options.py`` defined command line option ``-v{0,1,2}`` in- and decreases the verbosity, e.g.

.. container:: example-code

    .. code-block:: bash

        $ python -m runner -v2

        test_context (test.unit.test_contexts.Contexts) ... ok
        test_default (test.unit.test_contexts.Contexts) ... ok
        test_resources (test.unit.test_contexts.Contexts) ... ok

        ----------------------------------------------------------------------
        Ran 3 tests in 0.001s

        OK

To run a specific test in the subfolder, the test module ``test_xxxs`` needs to be executed, e.g.

.. container:: example-code

    .. code-block:: bash

        python -m test_contexts

From any folder other than the respective test folder, the python file needs to be executed, e.g.

.. container:: example-code

    .. code-block:: bash

        python path/to/test_xxxs.py

Adding New Tests
-----------------
During development of Arbor's python front end (via wrapper functions using `pybind11 <https://pybind11.readthedocs.io/en/stable/intro.html>`_) new unit tests are constantly added.
Thereby, three basic steps are performed:

1) Create a ``test_xxxs.py`` file in the according subfolder ``python/test/unit`` (serial cases) or ``python/test/unit_distributed`` (parallel cases).

2) In this file ``test_xxxs.py``

    a) import all necessary modules, e.g. ``import unittest``, ``import arbor``, ``import options``;
    b) define a unit test ``class Xxxs(unittest.TestCase)`` with test methods ``test_yyy`` using ``assert`` functions;
    c) add a ``suite()`` function consisting of all desired tests (either as tuple or all starting with `test`) and returning a unittest suite, e.g. ``unittest.makeSuite(Xxxs, ('test'))``;
    d) add a ``run()`` function with a ``runner = unittest.TextTestRunner()`` that runs the suite via ``runner.run(suite())``;
    e) finally, in ``if __name__ == "__main__":`` call ``run()``.

3) In the ``runner.py`` file

    a) ``import test_xxxs`` (and ``from test.subfolder import test_xxxs``);
    b) add the new test module ``test_xxxs`` to the ``test_modules`` list.

**Naming Convention**

    * Modules: ``test_xxxs.py`` all lower case, ending with ``s`` since module can consist of multiple classes;
    * Class(es): ``Xxxs`` first letter upper case, ending with ``s`` since class can consist of multiple test methods;
    * Methods: ``test_yyy`` all lower case, always starting with ``test`` since suite is build from all methods starting with ``test``.

.. container:: example-code

    .. code-block:: python

        # test_contexts.py

        import unittest

        import arbor

        # to be able to run .py file from child directory
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

        try:
            import options
        except ModuleNotFoundError:
            from test import options

        class Contexts(unittest.TestCase):

            def test_context(self):
                alloc = arbor.proc_allocation()

                ctx1 = arbor.context()

                self.assertEqual(ctx1.threads, alloc.threads)
                self.assertEqual(ctx1.has_gpu, alloc.has_gpu)

                # default construction does not use GPU or MPI
                self.assertEqual(ctx1.threads, 1)
                self.assertFalse(ctx1.has_gpu)
                self.assertFalse(ctx1.has_mpi)
                self.assertEqual(ctx1.ranks, 1)
                self.assertEqual(ctx1.rank, 0)

                # change allocation
                alloc.threads = 23
                self.assertEqual(alloc.threads, 23)
                alloc.gpu_id = -1
                self.assertEqual(alloc.gpu_id, -1)

                # test context construction with proc_allocation()
                ctx2 = arbor.context(alloc)
                self.assertEqual(ctx2.threads, alloc.threads)
                self.assertEqual(ctx2.has_gpu, alloc.has_gpu)
                self.assertEqual(ctx2.ranks, 1)
                self.assertEqual(ctx2.rank, 0)


        def suite():
        # specify class and test functions in tuple (here: all tests starting with 'test' from class Contexts
            suite = unittest.makeSuite(Contexts, ('test'))
            return suite

        def run():
            v = options.parse_arguments().verbosity
            runner = unittest.TextTestRunner(verbosity = v)
            runner.run(suite())

        if __name__ == "__main__":
            run()

.. container:: example-code

    .. code-block:: python

        # runner.py

        import unittest

        # to be able to run .py file from child directory
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

        try:
            import options
            import test_contexts
        # add more if needed
        except ModuleNotFoundError:
            from test import options
            from test.unit import test_contexts
        # add more if needed

        test_modules = [\
            test_contexts\
        ] # add more if needed

        def suite():
            loader = unittest.TestLoader()

            suites = []
            for test_module in test_modules:
                test_module_suite = test_module.suite()
                suites.append(test_module_suite)

            suite = unittest.TestSuite(suites)

            return suite

        if __name__ == "__main__":
            v = options.parse_arguments().verbosity
            runner = unittest.TextTestRunner(verbosity = v)
            runner.run(suite())

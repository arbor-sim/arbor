# -*- coding: utf-8 -*-
#
# test_contexts_mpi4py.py

import unittest

import arbor as arb

# to be able to run .py file from child directory
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    import options
except ModuleNotFoundError:
    from test import options

if options.TEST_MPI4PY == True and options.TEST_MPI == True:
    import mpi4py.MPI as mpi

"""
Contexts_mpi4py
   Goal: collect all tests for testing distributed arb.context using mpi4py
"""
# Only test class if env var ARB_WITH_MPI4PY=ON
@unittest.skipIf(options.TEST_MPI == False or options.TEST_MPI4PY == False, "ARB_WITH_MPI/-4PY=OFF")
class Contexts_mpi4py(unittest.TestCase):
    def test_initialize_mpi4py(self):
        # test mpi initialization (automatically when including mpi4py: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html)
        self.assertTrue(mpi.Is_initialized())

    def test_communicator_mpi4py(self):
        comm = arb.mpi_comm_from_mpi4py(mpi.COMM_WORLD)

        # test that set communicator is MPI_COMM_WORLD
        self.assertEqual(str(comm), '<mpi_comm: MPI_COMM_WORLD>')
        #print(comm)

    def test_context_mpi4py(self):
        comm = arb.mpi_comm_from_mpi4py(mpi.COMM_WORLD)

        # test context with mpi usage
        alloc = arb.proc_allocation()
        ctx = arb.context(alloc, comm)

        self.assertEqual(ctx.threads, alloc.threads)
        self.assertTrue(ctx.has_mpi)
        #print(ctx)

    def test_finalize_mpi4py(self):
        # test mpi finalization (automatically when including mpi4py, but only just before the Python process terminates)
        self.assertFalse(mpi.Is_finalized())

def suite():
    # specify class and test functions as tuple (here: all tests starting with 'test' from class Contexts_mpi4py
    suite = unittest.makeSuite(Contexts_mpi4py, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity
    runner = unittest.TextTestRunner(verbosity = v)
    runner.run(suite())

if __name__ == "__main__":
    run()

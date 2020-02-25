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

# check Arbor's configuration of mpi
mpi_enabled    = arb.__config__["mpi"]
mpi4py_enabled = arb.__config__["mpi4py"]

if (mpi_enabled and mpi4py_enabled):
    import mpi4py.MPI as mpi

"""
all tests for distributed arb.context using mpi4py
"""
# Only test class if env var ARB_WITH_MPI4PY=ON
@unittest.skipIf(mpi_enabled == False or mpi4py_enabled == False, "MPI/mpi4py not enabled")
class Contexts_mpi4py(unittest.TestCase):
    def test_initialized_mpi4py(self):
        # test mpi initialization (automatically when including mpi4py: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html)
        self.assertTrue(mpi.Is_initialized())

    def test_communicator_mpi4py(self):
        comm = arb.mpi_comm(mpi.COMM_WORLD)

        # test that set communicator is MPI_COMM_WORLD
        self.assertEqual(str(comm), '<arbor.mpi_comm: MPI_COMM_WORLD>')

    def test_context_mpi4py(self):
        comm = arb.mpi_comm(mpi.COMM_WORLD)

        # test context with mpi
        ctx = arb.context(mpi=comm)
        self.assertTrue(ctx.has_mpi)

    def test_context_allocation_mpi4py(self):
        comm = arb.mpi_comm(mpi.COMM_WORLD)

        # test context with alloc and mpi
        alloc = arb.proc_allocation()
        ctx = arb.context(alloc, comm)

        self.assertEqual(ctx.threads, alloc.threads)
        self.assertTrue(ctx.has_mpi)

    def test_exceptions_context_arbmpi(self):
        alloc = arb.proc_allocation()

        with self.assertRaisesRegex(RuntimeError,
            "mpi must be None, or an MPI communicator"):
            arb.context(mpi='MPI_COMM_WORLD')
        with self.assertRaisesRegex(RuntimeError,
            "mpi must be None, or an MPI communicator"):
            arb.context(alloc, mpi=0)

    def test_finalized_mpi4py(self):
        # test mpi finalization (automatically when including mpi4py, but only just before the Python process terminates)
        self.assertFalse(mpi.Is_finalized())

def suite():
    # specify class and test functions as tuple (here: all tests starting with 'test' from class Contexts_mpi4py
    suite = unittest.makeSuite(Contexts_mpi4py, ('test'))
    return suite

def run():
    v = options.parse_arguments().verbosity

    comm = arb.mpi_comm(mpi.COMM_WORLD)
    alloc = arb.proc_allocation()
    ctx = arb.context(alloc, comm)
    rank = ctx.rank

    if rank == 0:
        runner = unittest.TextTestRunner(verbosity = v)
    else:
        sys.stdout = open(os.devnull, 'w')
        runner = unittest.TextTestRunner(stream=sys.stdout)

    runner.run(suite())

if __name__ == "__main__":
    run()

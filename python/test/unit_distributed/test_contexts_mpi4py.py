# -*- coding: utf-8 -*-
#
# test_contexts_mpi4py.py

import unittest

import arbor as arb
from .. import cases

# check Arbor's configuration of mpi
mpi_enabled = arb.__config__["mpi"]
mpi4py_enabled = arb.__config__["mpi4py"]

if mpi_enabled and mpi4py_enabled:
    import mpi4py.MPI as mpi

"""
all tests for distributed arb.context using mpi4py
"""


# Only test class if env var ARB_WITH_MPI4PY=ON
@cases.skipIfNotDistributed()
class TestContexts_mpi4py(unittest.TestCase):
    def test_initialized_mpi4py(self):
        # test mpi initialization (automatically when including mpi4py: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html)
        self.assertTrue(mpi.Is_initialized())

    def test_communicator_mpi4py(self):
        comm = arb.mpi_comm(mpi.COMM_WORLD)

        # test that set communicator is MPI_COMM_WORLD
        self.assertEqual(str(comm), "<arbor.mpi_comm: MPI_COMM_WORLD>")

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

        with self.assertRaisesRegex(
            RuntimeError, "mpi must be None, or an MPI communicator"
        ):
            arb.context(mpi="MPI_COMM_WORLD")
        with self.assertRaisesRegex(
            RuntimeError, "mpi must be None, or an MPI communicator"
        ):
            arb.context(alloc, mpi=0)

    def test_finalized_mpi4py(self):
        # test mpi finalization (automatically when including mpi4py, but only just before the Python process terminates)
        self.assertFalse(mpi.Is_finalized())

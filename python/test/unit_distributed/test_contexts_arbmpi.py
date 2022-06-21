# -*- coding: utf-8 -*-
#
# test_contexts_arbmpi.py

import unittest

import arbor as arb
from .. import cases

"""
all tests for distributed arb.context using arbor mpi wrappers
"""


@cases.skipIfNotDistributed()
class TestContexts_arbmpi(unittest.TestCase):
    # Initialize mpi only once in this class (when adding classes move initialization to setUpModule()
    @classmethod
    def setUpClass(self):
        self.local_mpi = False
        if not arb.mpi_is_initialized():
            arb.mpi_init()
            self.local_mpi = True

    # Finalize mpi only once in this class (when adding classes move finalization to setUpModule()
    @classmethod
    def tearDownClass(self):
        if self.local_mpi:
            arb.mpi_finalize()

    def test_initialized_arbmpi(self):
        self.assertTrue(arb.mpi_is_initialized())

    def test_communicator_arbmpi(self):
        comm = arb.mpi_comm()

        # test that by default communicator is MPI_COMM_WORLD
        self.assertEqual(str(comm), "<arbor.mpi_comm: MPI_COMM_WORLD>")

    def test_context_arbmpi(self):
        comm = arb.mpi_comm()

        # test context with mpi
        ctx = arb.context(mpi=comm)
        self.assertTrue(ctx.has_mpi)

    def test_context_allocation_arbmpi(self):
        comm = arb.mpi_comm()

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

    def test_finalized_arbmpi(self):
        self.assertFalse(arb.mpi_is_finalized())

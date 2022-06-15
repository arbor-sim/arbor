# -*- coding: utf-8 -*-
#
# test_contexts.py

import unittest

import arbor as arb

"""
all tests for non-distributed arb.context
"""


class TestContexts(unittest.TestCase):
    def test_default_allocation(self):
        alloc = arb.proc_allocation()

        # test that by default proc_allocation has 1 thread and no GPU
        self.assertEqual(alloc.threads, 1)
        self.assertEqual(alloc.gpu_id, None)
        self.assertFalse(alloc.has_gpu)

    def test_set_allocation(self):
        alloc = arb.proc_allocation()

        # test changing allocation
        alloc.threads = 20
        self.assertEqual(alloc.threads, 20)
        alloc.gpu_id = 0
        self.assertEqual(alloc.gpu_id, 0)
        self.assertTrue(alloc.has_gpu)
        alloc.gpu_id = None
        self.assertFalse(alloc.has_gpu)

    def test_exceptions_allocation(self):
        with self.assertRaisesRegex(
            RuntimeError, "gpu_id must be None, or a non-negative integer"
        ):
            arb.proc_allocation(gpu_id=1.0)
        with self.assertRaisesRegex(
            RuntimeError, "gpu_id must be None, or a non-negative integer"
        ):
            arb.proc_allocation(gpu_id=-1)
        with self.assertRaisesRegex(
            RuntimeError, "gpu_id must be None, or a non-negative integer"
        ):
            arb.proc_allocation(gpu_id="gpu_id")
        with self.assertRaises(TypeError):
            arb.proc_allocation(threads=1.0)
        with self.assertRaisesRegex(ValueError, "threads must be a positive integer"):
            arb.proc_allocation(threads=0)
        with self.assertRaises(TypeError):
            arb.proc_allocation(threads=None)

    def test_default_context(self):
        ctx = arb.context()

        # test that by default context has 1 thread and no GPU, no MPI
        self.assertFalse(ctx.has_mpi)
        self.assertFalse(ctx.has_gpu)
        self.assertEqual(ctx.threads, 1)
        self.assertEqual(ctx.ranks, 1)
        self.assertEqual(ctx.rank, 0)

    def test_context(self):
        ctx = arb.context(threads=42, gpu_id=None)

        self.assertFalse(ctx.has_mpi)
        self.assertFalse(ctx.has_gpu)
        self.assertEqual(ctx.threads, 42)
        self.assertEqual(ctx.ranks, 1)
        self.assertEqual(ctx.rank, 0)

    def test_context_avail_threads(self):
        # test that 'avail_threads' returns at least 1.
        ctx = arb.context(threads="avail_threads", gpu_id=None)

        self.assertFalse(ctx.has_mpi)
        self.assertFalse(ctx.has_gpu)
        self.assertTrue(ctx.threads >= 1)
        self.assertEqual(ctx.ranks, 1)
        self.assertEqual(ctx.rank, 0)

    def test_context_allocation(self):
        alloc = arb.proc_allocation()

        # test context construction with proc_allocation()
        ctx = arb.context(alloc)
        self.assertEqual(ctx.threads, alloc.threads)
        self.assertEqual(ctx.has_gpu, alloc.has_gpu)
        self.assertEqual(ctx.ranks, 1)
        self.assertEqual(ctx.rank, 0)

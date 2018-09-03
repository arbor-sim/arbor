import unittest

import pyarb as arb

class test_context(unittest.TestCase):
    def test_default(self):
        ctx = arb.context()

    def test_resources(self):
        avail = arb.local_resources()

        # check that there is at least one thread
        self.assertGreaterEqual(avail.threads, 1)

        alloc = arb.proc_allocation()

        # test that by default proc_allocation takes its cues
        # from local_resources
        self.assertEqual(avail.threads, alloc.threads)
        if avail.gpus>0:
            self.assertTrue(alloc.has_gpu())
            self.assertEqual(alloc.gpu_id, 0)
        else:
            self.assertFalse(alloc.has_gpu)

        alloc.threads = 20
        self.assertEqual(alloc.threads, 20)

    def test_context(self):
        alloc = arb.proc_allocation()

        ctx1 = arb.context()
        print(ctx1)

        self.assertEqual(ctx1.threads, alloc.threads)
        self.assertEqual(ctx1.has_gpu, alloc.has_gpu)

        # default construction does not use MPI
        self.assertEqual(ctx1.has_mpi, False)
        self.assertEqual(ctx1.ranks, 1)
        self.assertEqual(ctx1.rank, 0)

        alloc.threads = 23
        print(alloc)
        alloc.gpu_id = -1

        ctx2 = arb.context(alloc)
        print(ctx2)

        #self.assertEqual(ctx1.threads, 23)
        #self.assertEqual(ctx1.has_gpu, False)

if __name__ == '__main__':
    unittest.main()

    help(arb.local_resources)

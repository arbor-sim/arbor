# -*- coding: utf-8 -*-
#
# test_domain_decompositions.py

import unittest

import arbor as arb
from .. import cases

# check Arbor's configuration of mpi and gpu
mpi_enabled = arb.__config__["mpi"]
gpu_enabled = arb.__config__["gpu"]

"""
all tests for distributed arb.domain_decomposition
"""


# Dummy recipe
class homo_recipe(arb.recipe):
    def __init__(self, n=4):
        arb.recipe.__init__(self)
        self.ncells = n

    def num_cells(self):
        return self.ncells

    def cell_description(self, gid):
        return []

    def cell_kind(self, gid):
        return arb.cell_kind.cable


# Heterogenous cell population of cable and rss cells.
# Interleaved so that cells with even gid are cable cells, and even gid are spike source cells.
class hetero_recipe(arb.recipe):
    def __init__(self, n=4):
        arb.recipe.__init__(self)
        self.ncells = n

    def num_cells(self):
        return self.ncells

    def cell_description(self, gid):
        tree = arb.segment_tree()
        tree.append(arb.mnpos, arb.mpoint(-3, 0, 0, 3), arb.mpoint(3, 0, 0, 3), tag=1)
        decor = arb.decor()
        decor.place("(location 0 0.5)", arb.gap_junction_site(), "gj")
        return arb.cable_cell(tree, decor)

    def cell_kind(self, gid):
        if gid % 2:
            return arb.cell_kind.spike_source
        else:
            return arb.cell_kind.cable

    def connections_on(self, gid):
        return []

    def event_generators(self, gid):
        return []


class gj_switch:
    def __init__(self, gid, shift):
        self.gid_ = gid
        self.shift_ = shift

    def switch(self, arg):
        default = []
        return getattr(self, "case_" + str(arg), lambda: default)()

    def case_1(self):
        return [arb.gap_junction_connection((7 + self.shift_, "gj"), "gj", 0.1)]

    def case_2(self):
        return [
            arb.gap_junction_connection((6 + self.shift_, "gj"), "gj", 0.1),
            arb.gap_junction_connection((9 + self.shift_, "gj"), "gj", 0.1),
        ]

    def case_6(self):
        return [
            arb.gap_junction_connection((2 + self.shift_, "gj"), "gj", 0.1),
            arb.gap_junction_connection((7 + self.shift_, "gj"), "gj", 0.1),
        ]

    def case_7(self):
        return [
            arb.gap_junction_connection((6 + self.shift_, "gj"), "gj", 0.1),
            arb.gap_junction_connection((1 + self.shift_, "gj"), "gj", 0.1),
        ]

    def case_9(self):
        return [arb.gap_junction_connection((2 + self.shift_, "gj"), "gj", 0.1)]


class gj_symmetric(arb.recipe):
    def __init__(self, num_ranks):
        arb.recipe.__init__(self)
        self.ncopies = num_ranks
        self.size = 10

    def num_cells(self):
        return self.size * self.ncopies

    def cell_description(self, gid):
        return []

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def gap_junctions_on(self, gid):
        shift = int((gid / self.size))
        shift *= self.size
        s = gj_switch(gid, shift)
        return s.switch(gid % self.size)


class gj_non_symmetric(arb.recipe):
    def __init__(self, num_ranks):
        arb.recipe.__init__(self)
        self.groups = num_ranks
        self.size = num_ranks

    def num_cells(self):
        return self.size * self.groups

    def cell_description(self, gid):
        tree = arb.segment_tree()
        tree.append(arb.mnpos, arb.mpoint(-3, 0, 0, 3), arb.mpoint(3, 0, 0, 3), tag=1)
        decor = arb.decor()
        decor.place("(location 0 0.5)", arb.gap_junction_site(), "gj")
        return arb.cable_cell(tree, decor)

    def cell_kind(self, gid):
        return arb.cell_kind.cable

    def gap_junctions_on(self, gid):
        group = int(gid / self.groups)
        id = gid % self.size

        if id == group and group != (self.groups - 1):
            return [arb.gap_junction_connection((gid + self.size, "gj"), "gj", 0.1)]
        elif id == group - 1:
            return [arb.gap_junction_connection((gid - self.size, "gj"), "gj", 0.1)]
        else:
            return []


@cases.skipIfNotDistributed()
class TestDomain_Decompositions_Distributed(unittest.TestCase):
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

    # 1 node with 1 cpu core, no gpus; assumes all cells will be put into cell groups of size 1
    def test_domain_decomposition_homogenous_MC(self):
        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=None, mpi=comm)
        else:
            context = arb.context(threads=1, gpu_id=None)

        N = context.ranks
        R = context.rank

        # 10 cells per domain
        n_local = 10
        n_global = n_local * N

        recipe = homo_recipe(n_global)
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_local)
        self.assertEqual(decomp.num_global_cells, n_global)
        self.assertEqual(len(decomp.groups), n_local)

        b = R * n_local
        e = (R + 1) * n_local
        gids = list(range(b, e))

        for gid in gids:
            self.assertEqual(R, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for cpu execution

        for i in gids:
            local_group = i - b
            grp = decomp.groups[local_group]

            self.assertEqual(len(grp.gids), 1)
            self.assertEqual(grp.gids[0], i)
            self.assertEqual(grp.backend, arb.backend.multicore)
            self.assertEqual(grp.kind, arb.cell_kind.cable)

    # 1 node with 1 cpu core, 1 gpu; assumes all cells will be placed on gpu in a single cell group
    @unittest.skipIf(not gpu_enabled, "GPU not enabled")
    def test_domain_decomposition_homogenous_GPU(self):

        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=0, mpi=comm)
        else:
            context = arb.context(threads=1, gpu_id=0)

        N = context.ranks
        R = context.rank

        # 10 cells per domain
        n_local = 10
        n_global = n_local * N

        recipe = homo_recipe(n_global)
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_local)
        self.assertEqual(decomp.num_global_cells, n_global)
        self.assertEqual(len(decomp.groups), 1)

        b = R * n_local
        e = (R + 1) * n_local

        gids = list(range(b, e))

        for gid in gids:
            self.assertEqual(R, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for gpu execution

        grp = decomp.groups[0]

        self.assertEqual(len(grp.gids), n_local)
        self.assertEqual(grp.gids[0], b)
        self.assertEqual(grp.gids[-1], e - 1)
        self.assertEqual(grp.backend, arb.backend.gpu)
        self.assertEqual(grp.kind, arb.cell_kind.cable)

    # 1 node with 1 cpu core, no gpus; assumes all cells will be put into cell groups of size 1
    def test_domain_decomposition_heterogenous_MC(self):
        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=None, mpi=comm)
        else:
            context = arb.context(threads=1, gpu_id=None)

        N = context.ranks
        R = context.rank

        # 10 cells per domain
        n_local = 10
        n_global = n_local * N
        n_local_groups = n_local  # 1 cell per group

        recipe = hetero_recipe(n_global)
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_local)
        self.assertEqual(decomp.num_global_cells, n_global)
        self.assertEqual(len(decomp.groups), n_local)

        b = R * n_local
        e = (R + 1) * n_local

        gids = list(range(b, e))

        for gid in gids:
            self.assertEqual(R, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for cpu execution
        grps = list(range(n_local_groups))
        kind_lists = dict()
        for i in grps:
            grp = decomp.groups[i]
            self.assertEqual(len(grp.gids), 1)
            k = grp.kind
            if k not in kind_lists:
                kind_lists[k] = []
            kind_lists[k].append(grp.gids[0])

            self.assertEqual(grp.backend, arb.backend.multicore)

        kinds = [arb.cell_kind.cable, arb.cell_kind.spike_source]
        for k in kinds:
            gids = kind_lists[k]
            self.assertEqual(len(gids), int(n_local / 2))
            for gid in gids:
                self.assertEqual(k, recipe.cell_kind(gid))

    def test_domain_decomposition_symmetric(self):
        nranks = 1
        rank = 0
        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=None, mpi=comm)
            nranks = context.ranks
            rank = context.rank
        else:
            context = arb.context(threads=1, gpu_id=None)

        recipe = gj_symmetric(nranks)
        decomp0 = arb.partition_load_balance(recipe, context)

        self.assertEqual(6, len(decomp0.groups))

        shift = int((rank * recipe.num_cells()) / nranks)

        exp_groups0 = [
            [0 + shift],
            [3 + shift],
            [4 + shift],
            [5 + shift],
            [8 + shift],
            [1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift],
        ]

        for i in range(6):
            self.assertEqual(exp_groups0[i], decomp0.groups[i].gids)

        cells_per_rank = int(recipe.num_cells() / nranks)

        for i in range(recipe.num_cells()):
            self.assertEqual(int(i / cells_per_rank), decomp0.gid_domain(i))

        # Test different group_hints
        hint1 = arb.partition_hint()
        hint1.prefer_gpu = False
        hint1.cpu_group_size = recipe.num_cells()
        hints1 = dict([(arb.cell_kind.cable, hint1)])

        decomp1 = arb.partition_load_balance(recipe, context, hints1)
        self.assertEqual(1, len(decomp1.groups))

        exp_groups1 = [
            0 + shift,
            3 + shift,
            4 + shift,
            5 + shift,
            8 + shift,
            1 + shift,
            2 + shift,
            6 + shift,
            7 + shift,
            9 + shift,
        ]

        self.assertEqual(exp_groups1, decomp1.groups[0].gids)

        for i in range(recipe.num_cells()):
            self.assertEqual(int(i / cells_per_rank), decomp1.gid_domain(i))

        hint2 = arb.partition_hint()
        hint2.prefer_gpu = False
        hint2.cpu_group_size = int(cells_per_rank / 2)
        hints2 = dict([(arb.cell_kind.cable, hint2)])

        decomp2 = arb.partition_load_balance(recipe, context, hints2)
        self.assertEqual(2, len(decomp2.groups))

        exp_groups2 = [
            [0 + shift, 3 + shift, 4 + shift, 5 + shift, 8 + shift],
            [1 + shift, 2 + shift, 6 + shift, 7 + shift, 9 + shift],
        ]

        for i in range(2):
            self.assertEqual(exp_groups2[i], decomp2.groups[i].gids)

        for i in range(recipe.num_cells()):
            self.assertEqual(int(i / cells_per_rank), decomp2.gid_domain(i))

    def test_domain_decomposition_nonsymmetric(self):
        nranks = 1
        rank = 0
        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=None, mpi=comm)
            nranks = context.ranks
            rank = context.rank
        else:
            context = arb.context(threads=1, gpu_id=None)

        recipe = gj_non_symmetric(nranks)
        decomp = arb.partition_load_balance(recipe, context)

        cells_per_rank = nranks

        # check groups
        i = 0
        for gid in range(rank * cells_per_rank, (rank + 1) * cells_per_rank):
            if gid % nranks == rank - 1:
                continue
            elif gid % nranks == rank and rank != nranks - 1:
                cg = [gid, gid + cells_per_rank]
                self.assertEqual(cg, decomp.groups[len(decomp.groups) - 1].gids)
            else:
                cg = [gid]
                self.assertEqual(cg, decomp.groups[i].gids)
                i += 1

        # check gid_domains
        for gid in range(recipe.num_cells()):
            group = int(gid / cells_per_rank)
            idx = gid % cells_per_rank
            ngroups = nranks
            if idx == group - 1:
                self.assertEqual(group - 1, decomp.gid_domain(gid))
            elif idx == group and group != ngroups - 1:
                self.assertEqual(group, decomp.gid_domain(gid))
            else:
                self.assertEqual(group, decomp.gid_domain(gid))

    def test_domain_decomposition_exceptions(self):
        nranks = 1
        if mpi_enabled:
            comm = arb.mpi_comm()
            context = arb.context(threads=1, gpu_id=None, mpi=comm)
            nranks = context.ranks
        else:
            context = arb.context(threads=1, gpu_id=None)

        recipe = gj_symmetric(nranks)

        hint1 = arb.partition_hint()
        hint1.prefer_gpu = False
        hint1.cpu_group_size = 0
        hints1 = dict([(arb.cell_kind.cable, hint1)])

        with self.assertRaisesRegex(
            RuntimeError,
            "unable to perform load balancing because cell_kind::cable has invalid suggested cpu_cell_group size of 0",
        ):
            arb.partition_load_balance(recipe, context, hints1)

        hint2 = arb.partition_hint()
        hint2.prefer_gpu = True
        hint2.gpu_group_size = 0
        hints2 = dict([(arb.cell_kind.cable, hint2)])

        with self.assertRaisesRegex(
            RuntimeError,
            "unable to perform load balancing because cell_kind::cable has invalid suggested gpu_cell_group size of 0",
        ):
            arb.partition_load_balance(recipe, context, hints2)

# -*- coding: utf-8 -*-
#
# test_domain_decomposition.py

import unittest

import arbor as arb

# check Arbor's configuration of mpi and gpu
gpu_enabled = arb.__config__["gpu"]

"""
all tests for non-distributed arb.domain_decomposition
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
        return []

    def cell_kind(self, gid):
        if gid % 2:
            return arb.cell_kind.spike_source
        else:
            return arb.cell_kind.cable


class TestDomain_Decompositions(unittest.TestCase):
    # 1 cpu core, no gpus; assumes all cells will be put into cell groups of size 1
    def test_domain_decomposition_homogenous_CPU(self):
        n_cells = 10
        recipe = homo_recipe(n_cells)
        context = arb.context()
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_cells)
        self.assertEqual(decomp.num_global_cells, n_cells)
        self.assertEqual(len(decomp.groups), n_cells)

        gids = list(range(n_cells))
        for gid in gids:
            self.assertEqual(0, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for cpu execution
        for i in gids:
            grp = decomp.groups[i]
            self.assertEqual(len(grp.gids), 1)
            self.assertEqual(grp.gids[0], i)
            self.assertEqual(grp.backend, arb.backend.multicore)
            self.assertEqual(grp.kind, arb.cell_kind.cable)

    # 1 cpu core, 1 gpu; assumes all cells will be placed on gpu in a single cell group
    @unittest.skipIf(not gpu_enabled, "GPU not enabled")
    def test_domain_decomposition_homogenous_GPU(self):
        n_cells = 10
        recipe = homo_recipe(n_cells)
        context = arb.context(threads=1, gpu_id=0)
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_cells)
        self.assertEqual(decomp.num_global_cells, n_cells)
        self.assertEqual(len(decomp.groups), 1)

        gids = range(n_cells)
        for gid in gids:
            self.assertEqual(0, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for gpu execution

        grp = decomp.groups[0]

        self.assertEqual(len(grp.gids), n_cells)
        self.assertEqual(grp.gids[0], 0)
        self.assertEqual(grp.gids[-1], n_cells - 1)
        self.assertEqual(grp.backend, arb.backend.gpu)
        self.assertEqual(grp.kind, arb.cell_kind.cable)

    # 1 cpu core, no gpus; assumes all cells will be put into cell groups of size 1
    def test_domain_decomposition_heterogenous_CPU(self):
        n_cells = 10
        recipe = hetero_recipe(n_cells)
        context = arb.context()
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_cells)
        self.assertEqual(decomp.num_global_cells, n_cells)
        self.assertEqual(len(decomp.groups), n_cells)

        gids = list(range(n_cells))
        for gid in gids:
            self.assertEqual(0, decomp.gid_domain(gid))

        # Each cell group contains 1 cell of kind cable
        # Each group should also be tagged for cpu execution
        grps = list(range(n_cells))
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
            self.assertEqual(len(gids), int(n_cells / 2))
            for gid in gids:
                self.assertEqual(k, recipe.cell_kind(gid))

    # 1 cpu core, 1 gpu; assumes cable cells will be placed on gpu in a single cell group; spike cells are on cpu in cell groups of size 1
    @unittest.skipIf(not gpu_enabled, "GPU not enabled")
    def test_domain_decomposition_heterogenous_GPU(self):
        n_cells = 10
        recipe = hetero_recipe(n_cells)
        context = arb.context(threads=1, gpu_id=0)
        decomp = arb.partition_load_balance(recipe, context)

        self.assertEqual(decomp.num_local_cells, n_cells)
        self.assertEqual(decomp.num_global_cells, n_cells)

        # one cell group with n_cells/2 on gpu, and n_cells/2 groups on cpu
        expected_groups = int(n_cells / 2) + 1
        self.assertEqual(len(decomp.groups), expected_groups)

        grps = range(expected_groups)
        n = 0
        # iterate over each group and test its properties
        for i in grps:
            grp = decomp.groups[i]
            k = grp.kind
            if k == arb.cell_kind.cable:
                self.assertEqual(grp.backend, arb.backend.gpu)
                self.assertEqual(len(grp.gids), int(n_cells / 2))
                for gid in grp.gids:
                    self.assertTrue(gid % 2 == 0)
                    n += 1
            elif k == arb.cell_kind.spike_source:
                self.assertEqual(grp.backend, arb.backend.multicore)
                self.assertEqual(len(grp.gids), 1)
                self.assertTrue(grp.gids[0] % 2)
                n += 1
        self.assertEqual(n_cells, n)

    def test_domain_decomposition_hints(self):
        n_cells = 20
        recipe = hetero_recipe(n_cells)
        context = arb.context()
        # The hints perfer the multicore backend, so the decomposition is expected
        # to never have cell groups on the GPU, regardless of whether a GPU is
        # available or not.
        cable_hint = arb.partition_hint()
        cable_hint.prefer_gpu = False
        cable_hint.cpu_group_size = 3
        spike_hint = arb.partition_hint()
        spike_hint.prefer_gpu = False
        spike_hint.cpu_group_size = 4
        hints = dict(
            [
                (arb.cell_kind.cable, cable_hint),
                (arb.cell_kind.spike_source, spike_hint),
            ]
        )

        decomp = arb.partition_load_balance(recipe, context, hints)

        exp_cable_groups = [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18]]
        exp_spike_groups = [[1, 3, 5, 7], [9, 11, 13, 15], [17, 19]]

        cable_groups = []
        spike_groups = []

        for g in decomp.groups:
            self.assertTrue(
                g.kind == arb.cell_kind.cable or g.kind == arb.cell_kind.spike_source
            )

            if g.kind == arb.cell_kind.cable:
                cable_groups.append(g.gids)
            elif g.kind == arb.cell_kind.spike_source:
                spike_groups.append(g.gids)

        self.assertEqual(exp_cable_groups, cable_groups)
        self.assertEqual(exp_spike_groups, spike_groups)

    def test_domain_decomposition_exceptions(self):
        n_cells = 20
        recipe = hetero_recipe(n_cells)
        context = arb.context()
        # The hints perfer the multicore backend, so the decomposition is expected
        # to never have cell groups on the GPU, regardless of whether a GPU is
        # available or not.
        cable_hint = arb.partition_hint()
        cable_hint.prefer_gpu = False
        cable_hint.cpu_group_size = 0
        spike_hint = arb.partition_hint()
        spike_hint.prefer_gpu = False
        spike_hint.gpu_group_size = 1
        hints = dict(
            [
                (arb.cell_kind.cable, cable_hint),
                (arb.cell_kind.spike_source, spike_hint),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "unable to perform load balancing because cell_kind::cable has invalid suggested cpu_cell_group size of 0",
        ):
            arb.partition_load_balance(recipe, context, hints)

        cable_hint = arb.partition_hint()
        cable_hint.prefer_gpu = False
        cable_hint.cpu_group_size = 1
        spike_hint = arb.partition_hint()
        spike_hint.prefer_gpu = True
        spike_hint.gpu_group_size = 0
        hints = dict(
            [
                (arb.cell_kind.cable, cable_hint),
                (arb.cell_kind.spike_source, spike_hint),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "unable to perform load balancing because cell_kind::spike_source has invalid suggested gpu_cell_group size of 0",
        ):
            arb.partition_load_balance(recipe, context, hints)

# -*- coding: utf-8 -*-
#
# test_multiple_connections.py

import unittest
import types
import numpy as np

import arbor as arb
from .. import fixtures

"""
Tests for multiple connections onto the same postsynaptic label and for one
connection that has the same net impact as the multiple-connection paradigm,
thereby testing the selection policies 'round_robin', 'round_robin_halt', and
'univalent'

NOTE: In principle, a plasticity (STDP) mechanism is employed here to test if a
      selected connection uses the correct instance of the mechanism. Thus, the
      scenario in Test #1 is intentionally "a wrong one", as opposed to the
      scenario in Test #2. In Test #1, one presynaptic neuron effectively
      connects _via one synapse_ to two postsynaptic neurons, and the spike at
      t=0.8ms in presynaptic neuron 0 will enhance potentiation in both the
      first and the second synapse mechanism. In Test #2, this is prevented by
      the 'round_robin_halt' policy, whereby the potentiation in the second
      synapse mechanism is only enhanced by spikes of presynaptic neuron 1.
"""


class TestMultipleConnections(unittest.TestCase):

    # Constructor (overridden)
    def __init__(self, args):
        super(TestMultipleConnections, self).__init__(args)

        self.runtime = 2  # ms
        self.dt = 0.01  # ms

    # Method creating a new mechanism for a synapse with STDP
    def create_syn_mechanism(self, scale_contrib=1):
        # create new synapse mechanism
        syn_mechanism = arb.mechanism("expsyn_stdp")

        # set pre- and postsynaptic contributions for STDP
        syn_mechanism.set("Apre", 0.01 * scale_contrib)
        syn_mechanism.set("Apost", -0.01 * scale_contrib)

        # set minimal decay time
        syn_mechanism.set("tau", self.dt)

        return syn_mechanism

    # Method that does the final evaluation for all tests
    def evaluate_outcome(self, sim, handle_mem):
        # membrane potential should temporarily be above the spiking threshold at around 1.0 ms (only testing this if the current node keeps the data, cf. GitHub issue #1892)
        if len(sim.samples(handle_mem)) > 0:
            data_mem, _ = sim.samples(handle_mem)[0]
            # print(data_mem[(data_mem[:, 0] >= 1.0), 1])
            self.assertGreater(data_mem[(np.round(data_mem[:, 0], 2) == 1.02), 1], -10)
            self.assertLess(data_mem[(np.round(data_mem[:, 0], 2) == 1.05), 1], -10)

        # neuron 3 should spike at around 1.0 ms, when the added input from all connections will cause threshold crossing
        spike_times = sim.spikes()["time"]
        spike_gids = sim.spikes()["source"]["gid"]
        # print(list(zip(*[spike_times, spike_gids])))
        self.assertGreater(sum(spike_gids == 3), 0)
        self.assertAlmostEqual(spike_times[(spike_gids == 3)][0], 1.00, delta=0.04)

    # Method that does additional evaluation for Test #1
    def evaluate_additional_outcome_1(self, sim, handle_mem):
        # order of spiking neurons (also cf. 'test_spikes.py')
        spike_gids = sim.spikes()["source"]["gid"]
        self.assertEqual([2, 1, 0, 3, 3], spike_gids.tolist())

        # neuron 3 should spike again at around 1.8 ms, when the added input from all connections will cause threshold crossing
        spike_times = sim.spikes()["time"]
        self.assertAlmostEqual(spike_times[(spike_gids == 3)][1], 1.80, delta=0.04)

    # Method that does additional evaluation for Test #2 and Test #3
    def evaluate_additional_outcome_2_3(self, sim, handle_mem):
        # order of spiking neurons (also cf. 'test_spikes.py')
        spike_gids = sim.spikes()["source"]["gid"]
        self.assertEqual([2, 1, 0, 3], spike_gids.tolist())

    # Method that runs the main part of Test #1 and Test #2
    def rr_main(self, context, art_spiker_recipe, weight, weight2):
        # define new method 'cell_description()' and overwrite the original one in the 'art_spiker_recipe' object
        create_syn_mechanism = self.create_syn_mechanism

        def cell_description(self, gid):
            # spike source neuron
            if gid < 3:
                return arb.spike_source_cell(
                    "spike_source", arb.explicit_schedule(self.trains[gid])
                )

            # spike-receiving cable neuron
            elif gid == 3:
                tree, labels, decor = self._cable_cell_elements()

                scale_stdp = 0.5  # use only half of the original magnitude for STDP because two connections will come into play

                decor.place(
                    '"midpoint"',
                    arb.synapse(create_syn_mechanism(scale_stdp)),
                    "postsyn_target",
                )  # place synapse for input from one presynaptic neuron at the center of the soma
                decor.place(
                    '"midpoint"',
                    arb.synapse(create_syn_mechanism(scale_stdp)),
                    "postsyn_target",
                )  # place synapse for input from another presynaptic neuron at the center of the soma
                # (using the same label as above!)
                return arb.cable_cell(tree, decor, labels)

        art_spiker_recipe.cell_description = types.MethodType(
            cell_description, art_spiker_recipe
        )

        # read connections from recipe for testing
        connections_from_recipe = art_spiker_recipe.connections_on(3)

        # connection #1 from neuron 0 to 3
        self.assertEqual(connections_from_recipe[0].dest.label, "postsyn_target")
        self.assertAlmostEqual(connections_from_recipe[0].weight, weight)
        self.assertAlmostEqual(connections_from_recipe[0].delay, 0.2)

        # connection #2 from neuron 0 to 3
        self.assertEqual(connections_from_recipe[1].dest.label, "postsyn_target")
        self.assertAlmostEqual(connections_from_recipe[1].weight, weight)
        self.assertAlmostEqual(connections_from_recipe[1].delay, 0.2)

        # connection #1 from neuron 1 to 3
        self.assertEqual(connections_from_recipe[2].dest.label, "postsyn_target")
        self.assertAlmostEqual(connections_from_recipe[2].weight, weight2)
        self.assertAlmostEqual(connections_from_recipe[2].delay, 1.4)

        # connection #2 from neuron 1 to 3
        self.assertEqual(connections_from_recipe[3].dest.label, "postsyn_target")
        self.assertAlmostEqual(connections_from_recipe[3].weight, weight2)
        self.assertAlmostEqual(connections_from_recipe[3].delay, 1.4)

        # construct domain_decomposition and simulation object
        sim = arb.simulation(art_spiker_recipe, context)
        sim.record(arb.spike_recording.all)

        # create schedule and handle to record the membrane potential of neuron 3
        reg_sched = arb.regular_schedule(0, self.dt, self.runtime)
        handle_mem = sim.sample((3, 0), reg_sched)

        # run the simulation
        sim.run(self.runtime, self.dt)

        return sim, handle_mem

    # Test #1 (for 'round_robin')
    @fixtures.context()
    @fixtures.art_spiker_recipe()
    @fixtures.sum_weight_hh_spike()
    @fixtures.sum_weight_hh_spike_2()
    def test_multiple_connections_rr_no_halt(
        self, context, art_spiker_recipe, sum_weight_hh_spike, sum_weight_hh_spike_2
    ):
        weight = (
            sum_weight_hh_spike / 2
        )  # connection strength which is, summed over two connections, just enough to evoke an immediate spike at t=1ms
        weight2 = (
            0.97 * sum_weight_hh_spike_2 / 2
        )  # connection strength which is, summed over two connections, just NOT enough to evoke an immediate spike at t=1.8ms

        # define new method 'connections_on()' and overwrite the original one in the 'art_spiker_recipe' object
        def connections_on(self, gid):
            # incoming to neurons 0--2
            if gid < 3:
                return []

            # incoming to neuron 3
            elif gid == 3:
                source_label_0 = arb.cell_global_label(
                    0, "spike_source"
                )  # referring to the "spike_source" label of neuron 0
                source_label_1 = arb.cell_global_label(
                    1, "spike_source"
                )  # referring to the "spike_source" label of neuron 1

                target_label_rr = arb.cell_local_label(
                    "postsyn_target", arb.selection_policy.round_robin
                )  # referring to the current item in the "postsyn_target" label group of neuron 3, moving to the next item afterwards

                conn_0_3_n1 = arb.connection(
                    source_label_0, target_label_rr, weight, 0.2
                )  # first connection from neuron 0 to 3
                conn_0_3_n2 = arb.connection(
                    source_label_0, target_label_rr, weight, 0.2
                )  # second connection from neuron 0 to 3
                # NOTE: this is not connecting to the same target label item as 'conn_0_3_n1' because 'round_robin' has been used before!
                conn_1_3_n1 = arb.connection(
                    source_label_1, target_label_rr, weight2, 1.4
                )  # first connection from neuron 1 to 3
                conn_1_3_n2 = arb.connection(
                    source_label_1, target_label_rr, weight2, 1.4
                )  # second connection from neuron 1 to 3
                # NOTE: this is not connecting to the same target label item as 'conn_1_3_n1' because 'round_robin' has been used before!

                return [conn_0_3_n1, conn_0_3_n2, conn_1_3_n1, conn_1_3_n2]

        art_spiker_recipe.connections_on = types.MethodType(
            connections_on, art_spiker_recipe
        )

        # run the main part of this test
        sim, handle_mem = self.rr_main(context, art_spiker_recipe, weight, weight2)

        # evaluate the outcome
        self.evaluate_outcome(sim, handle_mem)
        self.evaluate_additional_outcome_1(sim, handle_mem)

    # Test #2 (for the combination of 'round_robin_halt' and 'round_robin')
    @fixtures.context()
    @fixtures.art_spiker_recipe()
    @fixtures.sum_weight_hh_spike()
    @fixtures.sum_weight_hh_spike_2()
    def test_multiple_connections_rr_halt(
        self, context, art_spiker_recipe, sum_weight_hh_spike, sum_weight_hh_spike_2
    ):
        weight = (
            sum_weight_hh_spike / 2
        )  # connection strength which is, summed over two connections, just enough to evoke an immediate spike at t=1ms
        weight2 = (
            0.97 * sum_weight_hh_spike_2 / 2
        )  # connection strength which is, summed over two connections, just NOT enough to evoke an immediate spike at t=1.8ms

        # define new method 'connections_on()' and overwrite the original one in the 'art_spiker_recipe' object
        def connections_on(self, gid):
            # incoming to neurons 0--2
            if gid < 3:
                return []

            # incoming to neuron 3
            elif gid == 3:
                source_label_0 = arb.cell_global_label(
                    0, "spike_source"
                )  # referring to the "spike_source" label of neuron 0
                source_label_1 = arb.cell_global_label(
                    1, "spike_source"
                )  # referring to the "spike_source" label of neuron 1

                target_label_rr_halt = arb.cell_local_label(
                    "postsyn_target", arb.selection_policy.round_robin_halt
                )  # referring to the current item in the "postsyn_target" label group of neuron 3
                target_label_rr = arb.cell_local_label(
                    "postsyn_target", arb.selection_policy.round_robin
                )  # referring to the current item in the "postsyn_target" label group of neuron 3, moving to the next item afterwards

                conn_0_3_n1 = arb.connection(
                    source_label_0, target_label_rr_halt, weight, 0.2
                )  # first connection from neuron 0 to 3
                conn_0_3_n2 = arb.connection(
                    source_label_0, target_label_rr, weight, 0.2
                )  # second connection from neuron 0 to 3
                conn_1_3_n1 = arb.connection(
                    source_label_1, target_label_rr_halt, weight2, 1.4
                )  # first connection from neuron 1 to 3
                conn_1_3_n2 = arb.connection(
                    source_label_1, target_label_rr, weight2, 1.4
                )  # second connection from neuron 1 to 3

                return [conn_0_3_n1, conn_0_3_n2, conn_1_3_n1, conn_1_3_n2]

        art_spiker_recipe.connections_on = types.MethodType(
            connections_on, art_spiker_recipe
        )

        # run the main part of this test
        sim, handle_mem = self.rr_main(context, art_spiker_recipe, weight, weight2)

        # evaluate the outcome
        self.evaluate_outcome(sim, handle_mem)
        self.evaluate_additional_outcome_2_3(sim, handle_mem)

    # Test #3 (for 'univalent')
    @fixtures.context()
    @fixtures.art_spiker_recipe()
    @fixtures.sum_weight_hh_spike()
    @fixtures.sum_weight_hh_spike_2()
    def test_multiple_connections_uni(
        self, context, art_spiker_recipe, sum_weight_hh_spike, sum_weight_hh_spike_2
    ):
        weight = sum_weight_hh_spike  # connection strength which is just enough to evoke an immediate spike at t=1ms (equaling the sum of two connections in Test #2)
        weight2 = (
            0.97 * sum_weight_hh_spike_2
        )  # connection strength which is just NOT enough to evoke an immediate spike at t=1.8ms (equaling the sum of two connections in Test #2)

        # define new method 'connections_on()' and overwrite the original one in the 'art_spiker_recipe' object
        def connections_on(self, gid):
            # incoming to neurons 0--2
            if gid < 3:
                return []

            # incoming to neuron 3
            elif gid == 3:
                source_label_0 = arb.cell_global_label(
                    0, "spike_source"
                )  # referring to the "spike_source" label of neuron 0
                source_label_1 = arb.cell_global_label(
                    1, "spike_source"
                )  # referring to the "spike_source" label of neuron 1

                target_label_uni_n1 = arb.cell_local_label(
                    "postsyn_target_1", arb.selection_policy.univalent
                )  # referring to an only item in the "postsyn_target_1" label group of neuron 3
                target_label_uni_n2 = arb.cell_local_label(
                    "postsyn_target_2", arb.selection_policy.univalent
                )  # referring to an only item in the "postsyn_target_2" label group of neuron 3

                conn_0_3 = arb.connection(
                    source_label_0, target_label_uni_n1, weight, 0.2
                )  # connection from neuron 0 to 3
                conn_1_3 = arb.connection(
                    source_label_1, target_label_uni_n2, weight2, 1.4
                )  # connection from neuron 1 to 3

                return [conn_0_3, conn_1_3]

        art_spiker_recipe.connections_on = types.MethodType(
            connections_on, art_spiker_recipe
        )

        # define new method 'cell_description()' and overwrite the original one in the 'art_spiker_recipe' object
        create_syn_mechanism = self.create_syn_mechanism

        def cell_description(self, gid):
            # spike source neuron
            if gid < 3:
                return arb.spike_source_cell(
                    "spike_source", arb.explicit_schedule(self.trains[gid])
                )

            # spike-receiving cable neuron
            elif gid == 3:
                tree, labels, decor = self._cable_cell_elements()

                decor.place(
                    '"midpoint"',
                    arb.synapse(create_syn_mechanism()),
                    "postsyn_target_1",
                )  # place synapse for input from one presynaptic neuron at the center of the soma
                decor.place(
                    '"midpoint"',
                    arb.synapse(create_syn_mechanism()),
                    "postsyn_target_2",
                )  # place synapse for input from another presynaptic neuron at the center of the soma
                # (using another label as above!)

                return arb.cable_cell(tree, decor, labels)

        art_spiker_recipe.cell_description = types.MethodType(
            cell_description, art_spiker_recipe
        )

        # read connections from recipe for testing
        connections_from_recipe = art_spiker_recipe.connections_on(3)

        # connection from neuron 0 to 3
        self.assertEqual(connections_from_recipe[0].dest.label, "postsyn_target_1")
        self.assertAlmostEqual(connections_from_recipe[0].weight, weight)
        self.assertAlmostEqual(connections_from_recipe[0].delay, 0.2)

        # connection from neuron 1 to 3
        self.assertEqual(connections_from_recipe[1].dest.label, "postsyn_target_2")
        self.assertAlmostEqual(connections_from_recipe[1].weight, weight2)
        self.assertAlmostEqual(connections_from_recipe[1].delay, 1.4)

        # construct simulation object
        sim = arb.simulation(art_spiker_recipe, context)
        sim.record(arb.spike_recording.all)

        # create schedule and handle to record the membrane potential of neuron 3
        reg_sched = arb.regular_schedule(0, self.dt, self.runtime)
        handle_mem = sim.sample((3, 0), reg_sched)

        # run the simulation
        sim.run(self.runtime, self.dt)

        # evaluate the outcome
        self.evaluate_outcome(sim, handle_mem)
        self.evaluate_additional_outcome_2_3(sim, handle_mem)

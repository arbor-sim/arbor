# -*- coding: utf-8 -*-
#
# test_multiple_connections.py

import unittest
import types
import numpy as np

import arbor as arb
from .. import fixtures

"""
tests for multiple excitatory and inhibitory connections onto the same postsynaptic label and for one connection that has the same net impact as the multiple-connection paradigm,
thereby testing the selection policies 'round_robin', 'round_robin_halt', and 'univalent' (in principle testing if the right instances of mechanisms are used)
"""

class TestMultipleConnections(unittest.TestCase):
	
	# Method that does the final evaluation for both tests
	def evaluateOutcome(self, sim, handle_mem):

		# membrane potential should temporarily be above the spiking threshold at around 1.0 ms (only testing this if the current node keeps the data, cf. GitHub issue #1892)
		if len(sim.samples(handle_mem)) > 0:
			data_mem, _ = sim.samples(handle_mem)[0]
			print(data_mem[(data_mem[:, 0] >= 1.0), 1])
			self.assertGreater(data_mem[(np.round(data_mem[:, 0], 2) == 1.04), 1], -10)
			self.assertLess(data_mem[(np.round(data_mem[:, 0], 2) == 1.06), 1], -10)

		# spike in neuron 3 should occur at around 1.0 ms, when the added input from all connections will cause threshold crossing
		spike_times = sim.spikes()["time"]
		spike_gids = sim.spikes()["source"]["gid"]
		print(list(zip(*[spike_times, spike_gids])))
		self.assertGreater(sum(spike_gids == 3), 0)
		self.assertEqual([2, 1, 0, 3], spike_gids.tolist())
		self.assertAlmostEqual(spike_times[(spike_gids == 3)][0], 1.00, delta=0.04)

	@fixtures.context
	@fixtures.art_spiker_recipe
	@fixtures.sum_weight_hh_spike
	def test_multiple_connections(self, context, art_spiker_recipe, sum_weight_hh_spike):
		runtime = 2 # ms
		dt = 0.01 # ms
		weight = sum_weight_hh_spike # connection strength which is just enough to evoke an immediate spike; equals the summed weight of all connections (exc. and inh.) from neuron 0 and 1

		# define new method 'connections_on()' and overwrite the original one in the 'art_spiker_recipe' object
		def connections_on(self, gid):
			# incoming to neurons 0--2
			if gid < 3:
				return []
			
			# incoming to neuron 3
			elif gid == 3:
				source_label_0 = arb.cell_global_label(0, "spike_source") # referring to the "spike_source" label of neuron 0
				source_label_1 = arb.cell_global_label(1, "spike_source") # referring to the "spike_source" label of neuron 1

				target_label_rr_halt = arb.cell_local_label("postsyn_target", arb.selection_policy.round_robin_halt) # referring to the current item in the "postsyn_target" label group of neuron 3
				target_label_rr = arb.cell_local_label("postsyn_target", arb.selection_policy.round_robin) # referring to the current item in the "postsyn_target" label group of neuron 3, moving to the next item afterwards

				conn_0_3_n1 = arb.connection(source_label_0, target_label_rr_halt, 1, 0.2) # first (exc.) connection from neuron 0 to 3
				conn_0_3_n2 = arb.connection(source_label_0, target_label_rr_halt, 1, 0.2) # second (exc.) connection from neuron 0 to 3
				conn_0_3_n3 = arb.connection(source_label_0, target_label_rr, 1, 0.2) # third (exc.) connection from neuron 0 to 3
				conn_1_3_n1 = arb.connection(source_label_1, target_label_rr_halt, 1, 0.6) # first (inh.) connection from neuron 1 to 3
				conn_1_3_n2 = arb.connection(source_label_1, target_label_rr, 1, 0.6) # second (inh.) connection from neuron 1 to 3

				return [conn_0_3_n1, conn_0_3_n2, conn_0_3_n3, conn_1_3_n1, conn_1_3_n2]
		art_spiker_recipe.connections_on = types.MethodType(connections_on, art_spiker_recipe)

		# define new method 'cell_description()' and overwrite the original one in the 'art_spiker_recipe' object
		def cell_description(self, gid):
			# spike source neuron
			if gid < 3:
				return arb.spike_source_cell("spike_source", arb.explicit_schedule(self.trains[gid]))

			# spike-receiving cable neuron
			elif gid == 3:
				tree, labels, decor = self._cable_cell_elements()

				syn_mechanism1 = arb.mechanism("expsyn_curr")
				syn_mechanism1.set('w', weight) # set weight for excitation
				syn_mechanism1.set("tau", dt) # set minimal decay time
				
				syn_mechanism2 = arb.mechanism("expsyn_curr")
				syn_mechanism2.set('w', -weight) # set weight for inhibition
				syn_mechanism2.set("tau", dt) # set minimal decay time

				decor.place('"midpoint"', arb.synapse(syn_mechanism2), "postsyn_target") # place synaptic input from one presynaptic neuron at the center of the soma
				decor.place('"midpoint"', arb.synapse(syn_mechanism1), "postsyn_target") # place synaptic input from another presynaptic neuron at the center of the soma
				                                                                         # (using the same label as above!)
				return arb.cable_cell(tree, labels, decor)
		art_spiker_recipe.cell_description = types.MethodType(cell_description, art_spiker_recipe)

		# read connections from recipe for testing
		connections_from_recipe = art_spiker_recipe.connections_on(3)

		# connection #1 from neuron 0 to 3
		self.assertEqual(connections_from_recipe[0].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[0].weight, 1)
		self.assertAlmostEqual(connections_from_recipe[0].delay, 0.2)

		# connection #2 from neuron 0 to 3
		self.assertEqual(connections_from_recipe[1].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[1].weight, 1)
		self.assertAlmostEqual(connections_from_recipe[1].delay, 0.2)

		# connection #3 from neuron 0 to 3
		self.assertEqual(connections_from_recipe[2].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[2].weight, 1)
		self.assertAlmostEqual(connections_from_recipe[2].delay, 0.2)

		# connection #1 from neuron 1 to 3
		self.assertEqual(connections_from_recipe[3].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[3].weight, 1)
		self.assertAlmostEqual(connections_from_recipe[3].delay, 0.6)

		# connection #2 from neuron 1 to 3
		self.assertEqual(connections_from_recipe[4].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[4].weight, 1)
		self.assertAlmostEqual(connections_from_recipe[4].delay, 0.6)

		# construct domain_decomposition and simulation object
		dd = arb.partition_load_balance(art_spiker_recipe, context) 
		sim = arb.simulation(art_spiker_recipe, dd, context)
		sim.record(arb.spike_recording.all)

		# create schedule and handle to record the membrane potential of neuron 3
		reg_sched = arb.regular_schedule(0, dt, runtime)
		handle_mem = sim.sample((3, 0), reg_sched)

		# run the simulation
		sim.run(runtime, dt)
	
		# evaluate the outcome
		self.evaluateOutcome(sim, handle_mem)

	@fixtures.context
	@fixtures.art_spiker_recipe
	@fixtures.sum_weight_hh_spike
	def test_uni_connection(self, context, art_spiker_recipe, sum_weight_hh_spike):
		runtime = 2 # ms
		dt = 0.01 # ms
		weight = sum_weight_hh_spike # connection strength equaling the net sum of the connections in test_multiple_connections() (just enough to evoke an immediate spike)

		# define new method 'connections_on()' and overwrite the original one in the 'art_spiker_recipe' object
		def connections_on(self, gid):
			# incoming to neurons 0--2
			if gid < 3:
				return []
			
			# incoming to neuron 3
			elif gid == 3:
				source_label_0 = arb.cell_global_label(0, "spike_source") # referring to the "spike_source" label of neuron 0
				target_label_uni = arb.cell_local_label("postsyn_target", arb.selection_policy.univalent) # referring to an only item in the "postsyn_target" label group of neuron 3
				conn_0_3 = arb.connection(source_label_0, target_label_uni, weight, 0.2) # connection from neuron 0 to 3

				return [conn_0_3]
		art_spiker_recipe.connections_on = types.MethodType(connections_on, art_spiker_recipe)

		# define new method 'cell_description()' and overwrite the original one in the 'art_spiker_recipe' object
		def cell_description(self, gid):
			# spike source neuron
			if gid < 3:
				return arb.spike_source_cell("spike_source", arb.explicit_schedule(self.trains[gid]))

			# spike-receiving cable neuron
			elif gid == 3:
				tree, labels, decor = self._cable_cell_elements()
				syn_mechanism = arb.mechanism("expsyn_curr")
				syn_mechanism.set('w', weight) # set weight for excitation
				syn_mechanism.set("tau", dt) # set minimal decay time
				decor.place('"midpoint"', arb.synapse(syn_mechanism), "postsyn_target") # place synaptic input for one neuron at the center of the soma

				return arb.cable_cell(tree, labels, decor)
		art_spiker_recipe.cell_description = types.MethodType(cell_description, art_spiker_recipe)

		# read connections from recipe for testing
		connections_from_recipe = art_spiker_recipe.connections_on(3)

		# connection from neuron 0 to 3
		self.assertEqual(connections_from_recipe[0].dest.label, "postsyn_target")
		self.assertAlmostEqual(connections_from_recipe[0].weight, weight)
		self.assertAlmostEqual(connections_from_recipe[0].delay, 0.2)

		# construct domain_decomposition and simulation object
		dd = arb.partition_load_balance(art_spiker_recipe, context) 
		sim = arb.simulation(art_spiker_recipe, dd, context)
		sim.record(arb.spike_recording.all)

		# create schedule and handle to record the membrane potential of neuron 3
		reg_sched = arb.regular_schedule(0, dt, runtime)
		handle_mem = sim.sample((3, 0), reg_sched)

		# run the simulation
		sim.run(runtime, dt)
	
		# evaluate the outcome
		self.evaluateOutcome(sim, handle_mem)


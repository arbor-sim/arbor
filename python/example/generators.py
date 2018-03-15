import sys
import pyarb as arb
from pyarb import cell_member as cmem

class generator_recipe(arb.recipe):

    def num_cells(self):
        return 1

    def cell_description(self, gid):
        cell = arb.make_soma_cell()
        loc = arb.segment_location(0, 0.5)
        cell.add_synapse(loc)
        cell.add_detector(loc, 0)
        return cell

    def num_targets(self, gid):
        return 1

    def num_sources(self, gid):
        return 1

    def kind(self, gid):
        return arb.cell_kind.cable1d

    def event_generators(self, gid):
        # excitatory generator
        g1 = arb.poisson_generator()
        g1.target      = cmem(gid,0)
        g1.weight      = 0.001     # excit. weight
        g1.tstart      = 0
        g1.rate_per_ms = 100/1000  # frequency 500 Hz.
        g1.seed        = gid

        # inhibitory generator
        g2 = arb.poisson_generator()
        g2.target      = cmem(gid,0)
        g2.weight      = -0.001   # inhib. weight
        g2.tstart      = 0
        g2.rate_per_ms = 200/1000  # frequency 20 Hz.
        g2.seed        = gid
        return [g1, g2]


recipe = generator_recipe()

decomp = arb.partition_load_balance(recipe)

model = arb.model(recipe, decomp)

recorder = arb.make_spike_recorder(model)

model.run(500, 0.025)

spikes = recorder.spikes
print('There were ', len(spikes), ' spikes:')
for spike in spikes:
    print('  cell %2d at %8.3f ms'%(spike.source.gid, spike.time))



from common import config, metering, neuron_tools as nrn

env = config.load_env()

if env.mpi:
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank==0:
        print(env)
else:
    print(env)

from neuron import h

import mitral_point_gj
import parameters
import json
from matplotlib.pyplot import *



# A Ring network
class ring_network:
    def __init__(self, params):
        self.pc = h.ParallelContext()
        self.d_rank = int(self.pc.id())
        self.d_size = int(self.pc.nhost())

        self.num_cells = params.num_cells
        self.min_delay = params.min_delay
        self.cell_params = params.cell
        self.cells = []

        # distribute gid in round robin
        self.gids = range(self.d_rank, self.num_cells, self.d_size)

        # generate the cells
        for gid in self.gids:
            c = mitral_point_gj.cell(gid, self.cell_params)

            # attach delayed stimulus
            c.add_iclamp(gid*10, env.duration, 0.02, 'tuft0', 0.25)

            self.cells.append(c)

            # register this gid
            self.pc.set_gid2node(gid, self.d_rank)

            # This is the neuronic way to register a cell in a distributed
            # context. The netcon isn't meant to be used, so we hope that the
            # garbage collection does its job.
            nc = h.NetCon(c.soma(0.5)._ref_v, None, sec=c.soma)
            nc.threshold = 10
            self.pc.cell(gid, nc) # Associate the cell with this host and gid

        # connect consecutive somas with gap junctions
        if(env.gap):
            for gid in self.gids[:-1]:
                self.cells[gid].add_point_gap(self.cells[gid + 1], 0.00037, 'tuft0', 'tuft0')
            # this method must be called after all the calls to source_var() and target_var()
            # and before initializing the simulation
            self.pc.setup_transfer()

        # change the parameters for the last cell
        if(env.tweak):
            self.cells[-1].sections["tuft0"].gbar_nax = 0.0

# hoc setup
nrn.hoc_setup()

# create environment
ctx = nrn.neuron_context(env)

meter = metering.meter(env.mpi)
meter.start()

# build the model #####
params = parameters.model_parameters(env.parameter_file)
model = ring_network(params)

# set recorder for time and voltage
soma0_voltage, time0_points = model.cells[0].set_recorder()
soma1_voltage, time1_points = model.cells[1].set_recorder()

# do stdinit after set_recorder()
h.stdinit()

#####

ctx.init(params.min_delay, env.dt)

# set up spike output
spikes = nrn.spike_record()

meter.checkpoint('model-init')

# run the simulation
data = ctx.run(env.duration)

meter.checkpoint('model-run')

meter.print()

prefix = env.opath+'/nrn_'+params.name+'_';

report = metering.report_from_meter(meter)
report.to_file(prefix+'meters.json')

spikes.print(prefix+'spikes.gdf')

# print in correct format
with open('neuron.json', 'w') as f:
    f.write('[{\"units\": \"mV\",')
    f.write('\"data\": {\"soma0\": [')

    for x in soma0_voltage:
        f.write("%s,\n" % x)
    f.write("%s\n" % soma0_voltage[-1])

    f.write('],\n\"soma1\": [')

    for x in soma1_voltage:
        f.write("%s,\n" % x)
    f.write("%s\n" % soma1_voltage[-1])

    f.write('],\n\"time\": [')

    for x in time0_points:
        f.write("%s,\n" % x)
    f.write("%s\n" % time0_points[-1])

    f.write(']}}]')

soma0_plot = plot(time0_points, soma0_voltage, color='black')
soma1_plot = plot(time1_points, soma1_voltage, color='red')

legend(soma0_plot + soma1_plot, ['soma0', 'soma1'])
xlabel('time (ms)')
show()



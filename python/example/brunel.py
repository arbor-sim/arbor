import sys
import argparse
import random
import pyarb as arb
import matplotlib.pyplot as plt

def parse_clargs():
    p = argparse.ArgumentParser(description='A example python script that implements a Brunel benchmark.')
    p.add_argument("-n", "--nin", type=int, default=100,
            help="number of inhibitory cells: total cells = 5*n")
    return p.parse_args()

class brunel_recipe(arb.recipe):
    def __init__(self,
            n_exc, n_inh,
            ϵ, weight, delay, rel_inh_strength, λ):

        arb.recipe.__init__(self)

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.delay = delay

        self.weight_exc = weight
        self.weight_inh = -rel_inh_strength * weight
        self.weight_ext =  weight
        self.Ce = round(ϵ * n_exc)
        self.Ci = round(ϵ * n_inh)

        self.λ = λ*1e-3      # convert to spikes per ms

    def num_cells(self):
        # The total number of cells in the model is the sum of the exitatory
        # inhibitory populations.
        return self.n_exc + self.n_inh

    '''
    cell_description returns a description of the cell with a given gid.
    In this model all cells are lif_cells
    '''
    def cell_description(self, gid):
        cell = arb.lif_cell()
        cell.C_m    = 250 # pF (capacitance)
        cell.tau_m  =  20 # ms (time constant of membrane potential)
        cell.t_ref  =   2 # ms (refactory period)
        cell.E_L    =   0 # mv
        cell.V_reset=   0 # mv
        cell.V_m    =   0 # mv
        cell.V_th   =  20 # mv (threshold θ)
        return cell;

    '''
    Each cell has 1 target synapse.
    '''
    def num_targets(self, gid):
        return 1

    '''
    Each cell has 1 source, corresponding to a spike detector.
    '''
    def num_sources(self, gid):
        return 1

    '''
    The kind method returns the type of cell with gid.
    Note: this must agree with the type returned by cell_description.
    '''
    def kind(self, gid):
        return arb.cell_kind.lif

    # Make a ring network
    def connections_on(self, gid):
        r = random.Random()
        r.seed(gid)
        ne = self.n_exc
        ni = self.n_inh

        conns = []

        # add incoming excitatory connections.
        sources = r.sample(range(0, ne), k=self.Ce)
        for i in sources:
            src = arb.cell_member(i, 0)
            tgt = arb.cell_member(gid, 0)
            conns.append(arb.connection(src, tgt, self.weight_exc, self.delay))

        # add incoming inhibitory connections.
        sources = r.sample(range(ne, ne+ni), k=self.Ci)
        for i in sources:
            src = arb.cell_member(i, 0)
            tgt = arb.cell_member(gid, 0)
            conns.append(arb.connection(src, tgt, self.weight_inh, self.delay))

        return conns

    def event_generators(self, gid):
        # excitatory generator
        g = arb.poisson_generator()
        g.target      = arb.cell_member(gid,0)
        g.weight      = self.weight_ext
        g.tstart      = 0
        g.rate_per_ms = self.λ
        g.seed        = gid

        return [g]

args = parse_clargs()

# Use meters to measure the time taken to build and run the model.
meters = arb.meter_manager()

Ni = args.nin
Ne = 4*Ni

ϵ = 0.1
delay = 1.5
g = 5

weight = 30
λ = 17789 # frequency in Hz

recipe = brunel_recipe(Ne, Ni, ϵ, weight, delay, g, λ)

meters.start()
decomp = arb.partition_load_balance(recipe)
meters.checkpoint('domain decomp')

print ('building model')
model = arb.model(recipe, decomp)
recorder = arb.make_spike_recorder(model)
meters.checkpoint('model build')

print ('running model')
tsim = 1000
model.run(tsim, 0.1)
meters.checkpoint('model run')

print(arb.make_meter_report(meters))

spikes = recorder.spikes

arb.profiler_output(0.001, False)
print('=== there were ', len(spikes), ' spikes')
spikes_per_cell = len(spikes)/(Ne+Ni)
print('=== spike frequency: ', spikes_per_cell/(tsim*1e-3))

'''
Plot spikes from the first n_sample_cells of the inhibitory and excitatory cell
populations.
'''
n_sample_cells = 50
exc_spikes = [
        [s.source.gid, s.time]
        for s in spikes if s.source.gid<n_sample_cells]
inh_spikes = [
        [s.source.gid-Ne+n_sample_cells, s.time]
        for s in spikes if (s.source.gid>=Ne and s.source.gid<Ne+n_sample_cells)]

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Spike Time [ms]', fontsize=25)
ax.set_ylabel('Neuron ID', fontsize=25)

gids  = [s[0] for s in exc_spikes]
times = [s[1] for s in exc_spikes]
ax.scatter(times, gids, s=0.2, label='exc')

gids  = [s[0] for s in inh_spikes]
times = [s[1] for s in inh_spikes]
ax.scatter(times, gids, s=0.2, label='inh')

ax.legend()
fig.tight_layout()
fig.savefig("spikes.pdf")


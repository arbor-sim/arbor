#!/usr/bin/env python3
# This script is included in documentation. Adapt line numbers if touched.
#
# Authors: Sebastian Schmitt
#          Fabian Bösch
#
# Single-cell simulation: Calcium-based synapse which models synaptic efficacy as proposed by
# Graupner and Brunel, PNAS 109 (10): 3991-3996 (2012); https://doi.org/10.1073/pnas.1109359109,
# https://www.pnas.org/doi/10.1073/pnas.1220044110.
# The synapse dynamics is affected by additive white noise. The results reproduce the spike
# timing-dependent plasticity curve for the DP case described in Table S1 (supplemental material).

import arbor
import random
import multiprocessing
import numpy  # You may have to pip install these.
import pandas  # You may have to pip install these.
import seaborn  # You may have to pip install these.

# (1) Set simulation paramters

# Spike response delay (ms)
D = 13.7
# Spike frequency in Hertz
f = 1.0
# Number of spike pairs
num_spikes = 60
# time lag resolution
stdp_dt_step = 5.0
# Maximum time lag
stdp_max_dt = 100.0
# Ensemble size per initial value
ensemble_per_rho_0 = 1000
# Simulation time step
dt = 0.025
# List of initial values for 2 states
rho_0 = [0] * ensemble_per_rho_0 + [1] * ensemble_per_rho_0
# We need a synapse for each sample path
num_synapses = len(rho_0)
# Time lags between spike pairs (post-pre: < 0, pre-post: > 0)
stdp_dt = numpy.arange(-stdp_max_dt, stdp_max_dt + stdp_dt_step, stdp_dt_step)


# (2) Make the cell

# Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# Create and set up a decor object
decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("pas"))
    .place('"midpoint"', arbor.synapse("expsyn"), "driving_synapse")
    .place('"midpoint"', arbor.threshold_detector(-10), "detector")
)
for i in range(num_synapses):
    mech = arbor.mechanism("calcium_based_synapse")
    mech.set("rho_0", rho_0[i])
    decor.place('"midpoint"', arbor.synapse(mech), f"calcium_synapse_{i}")

# Create cell
cell = arbor.cable_cell(tree, decor, labels)


# (3) Create extended catalogue including stochastic mechanisms

cable_properties = arbor.neuron_cable_properties()
cable_properties.catalogue = arbor.default_catalogue()
cable_properties.catalogue.extend(arbor.stochastic_catalogue(), "")


# (4) Recipe


class stdp_recipe(arbor.recipe):
    def __init__(self, cell, props, gens):
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_props = props
        self.the_gens = gens

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def global_properties(self, kind):
        return self.the_props

    def probes(self, gid):
        return [arbor.cable_probe_point_state_cell("calcium_based_synapse", "rho")]

    def event_generators(self, gid):
        return self.the_gens


# (5) run simulation for a given time lag


def run(time_lag):

    # Time between stimuli
    T = 1000.0 / f

    # Simulation duration
    t1 = num_spikes * T

    # Time difference between post and pre spike including delay
    d = -time_lag + D

    # Stimulus and sample times
    t0_post = 0.0 if d >= 0 else -d
    t0_pre = d if d >= 0 else 0.0
    stimulus_times_post = numpy.arange(t0_post, t1, T)
    stimulus_times_pre = numpy.arange(t0_pre, t1, T)
    sched_post = arbor.explicit_schedule(stimulus_times_post)
    sched_pre = arbor.explicit_schedule(stimulus_times_pre)

    # Create strong enough driving stimulus
    generators = [arbor.event_generator("driving_synapse", 1.0, sched_post)]

    # Stimulus for calcium synapses
    for i in range(num_synapses):
        # Zero weight -> just modify synaptic weight via stdp
        generators.append(arbor.event_generator(f"calcium_synapse_{i}", 0.0, sched_pre))

    # Create recipe
    recipe = stdp_recipe(cell, cable_properties, generators)

    # Select one thread and no GPU
    alloc = arbor.proc_allocation(threads=1, gpu_id=None)
    context = arbor.context(alloc, mpi=None)
    domains = arbor.partition_load_balance(recipe, context)

    # Get random seed
    random_seed = random.getrandbits(64)

    # Create simulation
    sim = arbor.simulation(recipe, context, domains, random_seed)

    # Register prope to read out stdp curve
    handle = sim.sample((0, 0), arbor.explicit_schedule([t1 - dt]))

    # Run simulation
    sim.run(t1, dt)

    # Process sampled data
    data, meta = sim.samples(handle)[0]
    data_down = data[-1, 1 : ensemble_per_rho_0 + 1]
    data_up = data[-1, ensemble_per_rho_0 + 1 :]
    # Initial fraction of synapses in DOWN state
    beta = 0.5
    # Synaptic strength ratio UP to DOWN (w1/w0)
    b = 5
    # Transition propability form DOWN to UP
    P_U = numpy.mean(data_down > 0.5)
    # Transition probability from UP to DOWN
    P_D = numpy.mean(data_up < 0.5)
    # Return change in synaptic strength
    return (
        (1 - P_U) * beta + P_D * (1 - beta) + b * (P_U * beta + (1 - P_D) * (1 - beta))
    ) / (beta + (1 - beta) * b)


with multiprocessing.Pool() as p:
    results = p.map(run, stdp_dt)
print(stdp_dt)
print(results)

seaborn.set_theme()
df = pandas.DataFrame({"ds": results, "ms": stdp_dt})
plt = seaborn.relplot(kind="line", data=df, x="ms", y="ds")
plt.set_xlabels("lag time difference (ms)")
plt.set_ylabels("change in synaptic strenght (after/before)")
plt.savefig("calcium_stdp.svg")

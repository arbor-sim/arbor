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
num_spikes = 30
# time lag resolution
stdp_dt_step = 20.0
# Maximum time lag
stdp_max_dt = 100.0
# Ensemble size per initial value
ensemble_per_rho_0 = 100
# Simulation time step
dt = 0.1
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
    # Transition indicator form DOWN to UP
    P_UA = (data_down > 0.5).astype(float)
    # Transition indicator from UP to DOWN
    P_DA = (data_up < 0.5).astype(float)
    # Return change in synaptic strength
    ds_A = (
        (1 - P_UA) * beta
        + P_DA * (1 - beta)
        + b * (P_UA * beta + (1 - P_DA) * (1 - beta))
    ) / (beta + (1 - beta) * b)
    return pandas.DataFrame({"ds": ds_A, "ms": time_lag, "type": "Arbor"})


with multiprocessing.Pool() as p:
    results = p.map(run, stdp_dt)

ref = numpy.array(
    [
        [-100, 0.9793814432989691],
        [-95, 0.981715028725338],
        [-90, 0.9932274542583821],
        [-85, 0.982392230227282],
        [-80, 0.9620761851689686],
        [-75, 0.9688482001884063],
        [-70, 0.9512409611378684],
        [-65, 0.940405737106768],
        [-60, 0.9329565205853866],
        [-55, 0.9146720800329048],
        [-50, 0.8896156244609853],
        [-45, 0.9024824529979171],
        [-40, 0.8252814817763271],
        [-35, 0.8171550637530018],
        [-30, 0.7656877496052755],
        [-25, 0.7176064429672677],
        [-20, 0.7582385330838939],
        [-15, 0.7981934216985763],
        [-10, 0.8835208109434913],
        [-5, 0.9390513341028807],
        [0, 0.9927519271849183],
        [5, 1.2354639175257733],
        [10, 1.2255075694250952],
        [15, 1.1760718597832],
        [20, 1.1862298823123565],
        [25, 1.1510154042112806],
        [30, 1.125958948639361],
        [35, 1.1205413366238108],
        [40, 1.0812636495110723],
        [45, 1.0717828284838595],
        [50, 1.0379227533866708],
        [55, 1.0392771563905585],
        [60, 1.023024320343908],
        [65, 1.046049171409996],
        [70, 1.040631559394446],
        [75, 1.0257331263516831],
        [80, 1.0013538722817072],
        [85, 1.0121890963128077],
        [90, 1.0013538722817072],
        [95, 1.0094802903050326],
        [100, 0.9918730512544945],
    ]
)
df_ref = pandas.DataFrame({"ds": ref[:, 1], "ms": ref[:, 0], "type": "Reference"})

df = pandas.concat(results, ignore_index=True)
df = pandas.concat([df, df_ref], ignore_index=True)
plt = seaborn.relplot(kind="line", data=df, x="ms", y="ds", hue="type")
plt.set_xlabels("lag time difference (ms)")
plt.set_ylabels("change in synaptic strenght (after/before)")
plt._legend.set_title("")
plt.savefig("calcium_stdp.svg")

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

import arbor as A
from arbor import units as U
import numpy as np  # You may have to pip install these.
import pandas as pd  # You may have to pip install these.
import seaborn as sns  # You may have to pip install these.

# (1) Set simulation paramters
# Spike response delay
D = 13.7 * U.ms
# Spike frequency
f = 1000 * U.Hz
# Number of spike pairs
num_spikes = 30
# time lag resolution
stdp_dt_step = 20.0
# Maximum time lag
stdp_max_dt = 100.0
# Ensemble size per initial value
ensemble_per_rho_0 = 100
# Simulation time step
dt = 0.1 * U.ms
# List of initial values for 2 states; we need a synapse for each sample path
rho_0 = [0] * ensemble_per_rho_0 + [1] * ensemble_per_rho_0
# Time lags between spike pairs (post-pre: < 0, pre-post: > 0)
stdp_dt = np.arange(-stdp_max_dt, stdp_max_dt + stdp_dt_step, stdp_dt_step) * U.ms
# Time between stimuli
T = 1000.0 / f
# Simulation duration
tfinal = num_spikes * T

# (2) Make the cell
# Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = A.segment_tree()
tree.append(A.mnpos, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)

# Define the soma and its midpoint
labels = A.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# Create and set up a decor object
decor = (
    A.decor()
    .set_property(Vm=-40 * U.mV)
    .paint('"soma"', A.density("pas"))
    .place('"midpoint"', A.synapse("expsyn"), "driving_synapse")
    .place('"midpoint"', A.threshold_detector(-10 * U.mV), "detector")
)
for ix, rho in enumerate(rho_0):
    decor.place(
        '"midpoint"',
        A.synapse("calcium_based_synapse", rho_0=rho),
        f"calcium_synapse_{ix}",
    )

# Create cell
cell = A.cable_cell(tree, decor, labels)


# (3) Recipe
class stdp_recipe(A.recipe):
    def __init__(self, cell, time_lags):
        A.recipe.__init__(self)
        self.the_cell = cell
        # create extended catalogue including stochastic mechanisms
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue.extend(A.stochastic_catalogue())
        self.time_lags = time_lags
        self.num = len(time_lags)

    def num_cells(self):
        return self.num

    def cell_kind(self, _):
        return A.cell_kind.cable

    def cell_description(self, _):
        return self.the_cell

    def global_properties(self, _):
        return self.the_props

    def probes(self, _):
        return [A.cable_probe_point_state_cell("calcium_based_synapse", "rho", "rho")]

    def event_generators(self, gid):
        # Time difference between post and pre spike including delay
        d = D - self.time_lags[gid]
        # Stimulus and sample times
        t0_post = max(-d.value, 0) * U.ms
        t0_pre = max(d.value, 0) * U.ms
        sched_post = A.regular_schedule(t0_post, T, tfinal)
        sched_pre = A.regular_schedule(t0_pre, T, tfinal)

        # Create strong enough driving stimulus
        generators = [A.event_generator("driving_synapse", 1.0, sched_post)]

        # Stimulus for calcium synapses
        for ix, _ in enumerate(rho_0):
            # Zero weight -> just modify synaptic weight via stdp
            generators.append(
                A.event_generator(f"calcium_synapse_{ix}", 0.0, sched_pre)
            )
        return generators


# (4) run simulation for all lags
# Create recipe
rec = stdp_recipe(cell, stdp_dt)

# Create simulation
print(A.config())
sim = A.simulation(rec, seed=42)

# Register probe to read out stdp curve
handles = [
    sim.sample((gid, "rho"), A.explicit_schedule([tfinal - dt]))
    for gid in range(len(stdp_dt))
]

sim.record(A.spike_recording.all)

# Run simulation
sim.run(tfinal, dt)


# (5) Process sampled data
# Add reference
ref = np.array(
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

results = [pd.DataFrame({"ds": ref[:, 1], "ms": ref[:, 0], "type": "Reference"})]
for handle, time_lag in zip(handles, stdp_dt):
    data, _ = sim.samples(handle)[0]
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
    results.append(pd.DataFrame({"ds": ds_A, "ms": time_lag.value, "type": "Arbor"}))

df = pd.concat(results, ignore_index=True)
plt = sns.relplot(kind="line", data=df, x="ms", y="ds", hue="type")
plt.set_xlabels("lag time difference (ms)")
plt.set_ylabels("change in synaptic strenght (after/before)")
plt._legend.set_title("")
plt.savefig("calcium_stdp.svg")

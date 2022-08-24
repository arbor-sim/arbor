#!/usr/bin/env python3

import arbor
import pandas
import seaborn
import sys

try:
    from bluepyopt import ephys
except ImportError:
    raise ImportError("Please install bluepyopt to run this example.")

# (1) Read the cell JSON description referencing morphology, label dictionary and decor.

if len(sys.argv) < 2:
    print("No JSON file passed to the program")
    sys.exit(0)

cell_json_filename = sys.argv[1]
cell_json, morpho, labels, decor = ephys.create_acc.read_acc(cell_json_filename)

# (2) Create and populate the label dictionary.

# Locsets:

# Add a labels for the root of the morphology and all the terminal points
labels["root"] = "(root)"
labels["terminal"] = "(terminal)"
# Add a label for the terminal locations in the "axon" region:
labels["axon_terminal"] = '(restrict (locset "terminal") (region "axon"))'

# (3) Create and populate the decor.

# Place stimuli and spike detectors.
decor.place('"root"', arbor.iclamp(10, 10, current=50), "iclamp0")
decor.place('"root"', arbor.iclamp(30, 1, current=20), "iclamp1")
decor.place('"root"', arbor.iclamp(50, 1, current=20), "iclamp2")
decor.place('"axon_terminal"', arbor.spike_detector(-10), "detector")

# Single CV for the "soma" region
soma_policy = arbor.cv_policy_single('"soma"')
# Single CV for the "soma" region
dflt_policy = arbor.cv_policy_max_extent(10)
# default policy everywhere except the soma
policy = dflt_policy | soma_policy
# Set cv_policy
decor.discretization(policy)

# (4) Create the cell.

cell = arbor.cable_cell(morpho, labels, decor)

# (5) Declare a probe.

probe = arbor.cable_probe_membrane_voltage('"axon_terminal"')

# (6) Create a class that inherits from arbor.recipe
class single_recipe(arbor.recipe):

    # (6.1) Define the class constructor
    def __init__(self, cell, probes):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes

        self.the_props = arbor.neuron_cable_properties()

        # Add catalogues with qualifiers
        self.the_props.catalogue = arbor.catalogue()
        self.the_props.catalogue.extend(arbor.default_catalogue(), "default::")
        self.the_props.catalogue.extend(arbor.bbp_catalogue(), "BBP::")

    # (6.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (6.3) Override the num_targets method
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # (6.4) Override the cell_description method
    def cell_description(self, gid):
        return self.the_cell

    # (6.5) Override the probes method
    def probes(self, gid):
        return self.the_probes

    # (6.6) Override the connections_on method
    def connections_on(self, gid):
        return []

    # (6.7) Override the gap_junction_on method
    def gap_junction_on(self, gid):
        return []

    # (6.8) Override the event_generators method
    def event_generators(self, gid):
        return []

    # (6.9) Overrode the global_properties method
    def global_properties(self, gid):
        return self.the_props


# Instantiate recipe
# Pass the probe in a list because that it what single_recipe expects.
recipe = single_recipe(cell, [probe])

# (4) Create an execution context
context = arbor.context()

# (5) Create a domain decomposition
domains = arbor.partition_load_balance(recipe, context)

# (6) Create a simulation
sim = arbor.simulation(recipe, context, domains)

# Instruct the simulation to record the spikes and sample the probe
sim.record(arbor.spike_recording.all)

probe_id = arbor.cell_member(0, 0)
handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

# (7) Run the simulation
sim.run(tfinal=100, dt=0.025)

# (8) Print or display the results
spikes = sim.spikes()
print(len(spikes), "spikes recorded:")
for s in spikes:
    print(s)

data = []
meta = []
for d, m in sim.samples(handle):
    data.append(d)
    meta.append(m)

df_list = []
for i in range(len(data)):
    df_list.append(
        pandas.DataFrame(
            {
                "t/ms": data[i][:, 0],
                "U/mV": data[i][:, 1],
                "Location": str(meta[i]),
                "Variable": "voltage",
            }
        )
    )
df = pandas.concat(df_list, ignore_index=True)
seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Location", col="Variable", ci=None
).savefig("single_cell_bpo_l5pc.svg")

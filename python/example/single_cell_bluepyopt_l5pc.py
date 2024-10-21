#!/usr/bin/env python3

import arbor as A
import pandas
import seaborn
import sys

try:
    from bluepyopt import ephys
except ImportError:
    print("Please install bluepyopt to run this example.")
    exit(-42)

# (1) Read the cell JSON description referencing morphology, label dictionary and decor.
if len(sys.argv) < 2:
    print("No JSON file passed to the program")
    sys.exit(0)

cell_json_filename = sys.argv[1]
cell_json, morpho, decor, labels = ephys.create_acc.read_acc(cell_json_filename)

# (2) Define labels for stimuli and voltage recordings.
labels["soma_center"] = "(location 0 0.5)"
labels["dend1"] = (
    '(restrict-to (distal-translate (proximal (region "apic")) 660)'
    " (proximal-interval (distal (branch 123))))"
)

# (3) Define stimulus and spike detector, adjust discretization
decor.place(
    '"soma_center"', A.iclamp(tstart=295, duration=5, current=1.9), "soma_iclamp"
)

# Add spike detector
decor.place('"soma_center"', A.threshold_detector(-10), "detector")

# Adjust discretization (single CV on soma, default everywhere else)
cvp = A.cv_policy_max_extent(1.0) | A.cv_policy_single('"soma"')

# (4) Create the cell.
cell = A.cable_cell(morpho, decor, labels, cvp)

# (5) Declare a probe.
probe = A.cable_probe_membrane_voltage('"dend1"')


# (6) Create a class that inherits from A.recipe
class single_recipe(A.recipe):
    # (6.1) Define the class constructor
    def __init__(self, cell, probes):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        super().__init__()
        self.the_cell = cell
        self.the_probes = probes

        self.the_props = A.neuron_cable_properties()

        # Add catalogues with explicit qualifiers
        self.the_props.catalogue = A.catalogue()
        self.the_props.catalogue.extend(A.default_catalogue(), "default::")
        self.the_props.catalogue.extend(A.bbp_catalogue(), "BBP::")

    # (6.2) Override the num_cells method
    def num_cells(self):
        return 1

    # (6.3) Override the cell_kind method
    def cell_kind(self, gid):
        return A.cell_kind.cable

    # (6.4) Override the cell_description method
    def cell_description(self, gid):
        return self.the_cell

    # (6.5) Override the probes method
    def probes(self, gid):
        return self.the_probes

    # (6.6) Overrode the global_properties method
    def global_properties(self, gid):
        return self.the_props


# Instantiate recipe
# Pass the probe in a list because that it what single_recipe expects.
recipe = single_recipe(cell, [probe])

# (7) Create a simulation (using defaults for context and partition_load_balance)
sim = A.simulation(recipe)

# Instruct the simulation to record the spikes and sample the probe
sim.record(A.spike_recording.all)

probe_id = A.cell_member(0, 0)
handle = sim.sample(probe_id, A.regular_schedule(0.02))

# (8) Run the simulation
sim.run(tfinal=600, dt=0.025)

# (9) Print or display the results
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
    data=df,
    kind="line",
    x="t/ms",
    y="U/mV",
    hue="Location",
    col="Variable",
    errorbar=None,
).savefig("single_cell_bluepyopt_l5pc_bAP_dend1.svg")

#!/usr/bin/env python3

import arbor
import pandas
import seaborn
import sys

try:
    from bluepyopt import ephys
except ImportError:
    raise ImportError("Please install bluepyopt to run this example.")

if len(sys.argv) < 2:
    print("No JSON file passed to the program")
    sys.exit(0)

cell_json_filename = sys.argv[1]
cell_json, morpho, labels, decor = ephys.create_acc.read_acc(cell_json_filename)

# Define the regions and locsets in the model.
defs = {
    "root": "(root)",  # the start of the soma in this morphology is at the root of the cell.
    "stim_site": "(location 0 0.5)",  # site for the stimulus, in the middle of branch 0.
    "det_site": "(location 0 0.8)",
}  # end of the axon.
labels.append(arbor.label_dict(defs))

# Set initial membrane potential to -55 mV
decor.set_property(Vm=-55)
# Attach stimuli that inject 4 nA current for 1 ms, starting at 3 and 8 ms.
decor.place('"root"', arbor.iclamp(10, 1, current=5), "iclamp0")
decor.place('"stim_site"', arbor.iclamp(3, 1, current=0.5), "iclamp1")
decor.place('"stim_site"', arbor.iclamp(10, 1, current=0.5), "iclamp2")
decor.place('"stim_site"', arbor.iclamp(8, 1, current=4), "iclamp3")
# Detect spikes at the soma with a voltage threshold of -10 mV.
decor.place('"det_site"', arbor.spike_detector(-10), "detector")

# Create the policy used to discretise the cell into CVs.
# Use a single CV for the soma, and CVs of maximum length 1 μm elsewhere.
soma_policy = arbor.cv_policy_single('"soma"')
dflt_policy = arbor.cv_policy_max_extent(1.0)
policy = dflt_policy | soma_policy
decor.discretization(policy)

# Combine morphology with region and locset definitions to make a cable cell.
cell = arbor.cable_cell(morpho, labels, decor)

# Make single cell model.
m = arbor.single_cell_model(cell)

# Add catalogues with qualifiers
m.properties.catalogue = arbor.catalogue()
m.properties.catalogue.extend(arbor.default_catalogue(), "default::")
m.properties.catalogue.extend(arbor.bbp_catalogue(), "BBP::")

# Attach voltage probes that sample at 50 kHz.
m.probe("voltage", where='"root"', frequency=50)
m.probe("voltage", where='"stim_site"', frequency=50)
m.probe("voltage", where='"det_site"', frequency=50)

# Simulate the cell for 15 ms.
tfinal = 15
m.run(tfinal)
print("Simulation done.")

# Print spike times.
if len(m.spikes) > 0:
    print("{} spikes:".format(len(m.spikes)))
    for s in m.spikes:
        print("  {:7.4f}".format(s))
else:
    print("no spikes")

# Plot the recorded voltages over time.
print("Plotting results ...")
df_list = []
for t in m.traces:
    df_list.append(
        pandas.DataFrame(
            {
                "t/ms": t.time,
                "U/mV": t.value,
                "Location": str(t.location),
                "Variable": t.variable,
            }
        )
    )

df = pandas.concat(df_list, ignore_index=True)

seaborn.relplot(
    data=df, kind="line", x="t/ms", y="U/mV", hue="Location", col="Variable", ci=None
).savefig("single_cell_bpo_simple.svg")

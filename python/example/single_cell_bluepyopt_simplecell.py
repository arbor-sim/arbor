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
cell_json, morpho, decor, labels = ephys.create_acc.read_acc(cell_json_filename)

# (2) Define labels for stimuli and voltage recordings.

labels["soma_center"] = "(location 0 0.5)"

# (3) Define stimulus and spike detector, adjust discretization

decor.place(
    '"soma_center"', arbor.iclamp(tstart=100, duration=50, current=0.05), "soma_iclamp"
)

# Add spike detector
decor.place('"soma_center"', arbor.threshold_detector(-10), "detector")

# Adjust discretization (single CV on soma, default everywhere else)
decor.discretization(arbor.cv_policy_max_extent(1.0) | arbor.cv_policy_single('"soma"'))

# (4) Create the cell.

cell = arbor.cable_cell(morpho, decor, labels)

# (5) Make the single cell model.

m = arbor.single_cell_model(cell)

# Add catalogues with qualifiers
m.properties.catalogue = arbor.catalogue()
m.properties.catalogue.extend(arbor.default_catalogue(), "default::")
m.properties.catalogue.extend(arbor.bbp_catalogue(), "BBP::")

# (6) Attach voltage probe that samples at 50 kHz.
m.probe("voltage", where='"soma_center"', frequency=50)

# (7) Simulate the cell for 200 ms.
m.run(tfinal=200)
print("Simulation done.")

# (8) Print spike times.
if len(m.spikes) > 0:
    print("{} spikes:".format(len(m.spikes)))
    for s in m.spikes:
        print("  {:7.4f}".format(s))
else:
    print("no spikes")

# (9) Plot the recorded voltages over time.
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
    data=df,
    kind="line",
    x="t/ms",
    y="U/mV",
    hue="Location",
    col="Variable",
    errorbar=None,
).savefig("single_cell_bluepyopt_simplecell.svg")

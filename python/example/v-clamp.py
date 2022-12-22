#!/usr/bin/env python3

import arbor
import pandas  # You may have to pip install these.
import seaborn  # You may have to pip install these.

# (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "midpoint": "(location 0 0.5)"})

# (3) Create and set up a decor object

decor = (
    arbor.decor()
    .set_property(Vm=-40)
    .paint('"soma"', arbor.density("hh"))
    .paint('"soma"', arbor.voltage_process("v_clamp/v0=-42"))
    .place('"midpoint"', arbor.iclamp(10, 2, 0.8), "iclamp")
    .place('"midpoint"', arbor.threshold_detector(-10), "detector")
)

# (4) Create cell and the single cell model based on it
cell = arbor.cable_cell(tree, decor, labels)

# (5) Make single cell model.
m = arbor.single_cell_model(cell)

# (6) Attach voltage probe sampling at 10 kHz (every 0.1 ms).
m.probe("voltage", '"midpoint"', frequency=10)

# (7) Run simulation for 30 ms of simulated activity.
m.run(tfinal=30)

# (8) Print spike times.
if len(m.spikes) > 0:
    print("{} spikes:".format(len(m.spikes)))
    for s in m.spikes:
        print("{:3.3f}".format(s))
else:
    print("no spikes")

# (9) Plot the recorded voltages over time.
print("Plotting results ...")
df = pandas.DataFrame({"t/ms": m.traces[0].time, "U/mV": m.traces[0].value})
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV", errorbar=None).savefig(
    "v-clamp.svg"
)

# (10) Optionally, you can store your results for later processing.
df.to_csv("v-clamp.csv", float_format="%g")

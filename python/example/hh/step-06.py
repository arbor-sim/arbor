#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import matplotlib.pyplot as plt

# Create a single segment morphology
tree = A.segment_tree()
tree.append(A.mnpos, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)

# Create (almost empty) decor
decor = (
    A.decor()
    .paint("(all)", A.density("hh06"))
    .place("(location 0 0.5)", A.iclamp(10 * U.ms, 2 * U.ms, 0.8 * U.nA), "iclamp")
)

# Run the model, extracting the membrane voltage
model = A.single_cell_model(A.cable_cell(tree, decor))
model.probe("voltage", "(location 0 0.5)", tag="Um", frequency=10 * U.kHz)

# add our catalogue
model.properties.catalogue = A.load_catalogue("cat-catalogue.so")

model.run(tfinal=30 * U.ms)

# Create a basic plot
fg, ax = plt.subplots()
ax.plot(model.traces[0].time, model.traces[0].value)
ax.set_xlabel("t/ms")
ax.set_ylabel("U/mV")
plt.savefig("hh-06.pdf")
plt.savefig("hh-06.svg")

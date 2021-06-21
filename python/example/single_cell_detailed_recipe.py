#!/usr/bin/env python3

import arbor
import pandas
import seaborn
import sys
from arbor import mechanism as mech

#(1) Creat a cell.

# Create the morphology

# Read the SWC filename from input
# Example from docs: single_cell_detailed.swc

if len(sys.argv) < 2:
    print("No SWC file passed to the program")
    sys.exit(0)

filename = sys.argv[1]
morph = arbor.load_swc_arbor(filename)

# Create and populate the label dictionary.

labels = arbor.label_dict()

# Regions:

labels['soma'] = '(tag 1)'
labels['axon'] = '(tag 2)'
labels['dend'] = '(tag 3)'
labels['last'] = '(tag 4)'

labels['all'] = '(all)'

labels['gt_1.5'] = '(radius-ge (region "all") 1.5)'
labels['custom'] = '(join (region "last") (region "gt_1.5"))'

# Locsets:

labels['root']     = '(root)'
labels['terminal'] = '(terminal)'
labels['custom_terminal'] = '(restrict (locset "terminal") (region "custom"))'
labels['axon_terminal'] = '(restrict (locset "terminal") (region "axon"))'

# Create and populate the decor.

decor = arbor.decor()

# Set the default properties.

decor.set_property(Vm =-55)

# Override the defaults.

decor.paint('"custom"', tempK=270)
decor.paint('"soma"',   Vm=-50)

# Paint density mechanisms.

decor.paint('"all"', 'pas')
decor.paint('"custom"', 'hh')
decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

# Place stimuli and spike detectors.

decor.place('"root"', arbor.iclamp(10, 1, current=2), "iclamp0")
decor.place('"root"', arbor.iclamp(30, 1, current=2), "iclamp1")
decor.place('"root"', arbor.iclamp(50, 1, current=2), "iclamp2")
decor.place('"axon_terminal"', arbor.spike_detector(-10), "detector")

# Set cv_policy

soma_policy = arbor.cv_policy_single('"soma"')
dflt_policy = arbor.cv_policy_max_extent(1.0)
policy = dflt_policy | soma_policy
decor.discretization(policy)

# Create a cell

cell = arbor.cable_cell(morph, labels, decor)

# (2) Declare a probe.

probe = arbor.cable_probe_membrane_voltage('"custom_terminal"')

# (3) Create a recipe class and instantiate a recipe

class single_recipe (arbor.recipe):

    def __init__(self, cell, probes):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes

        self.the_cat = arbor.default_catalogue()
        self.the_cat.extend(arbor.allen_catalogue(), "")

        self.the_props = arbor.cable_global_properties()
        self.the_props.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
        self.the_props.set_ion(ion='na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
        self.the_props.set_ion(ion='k',  int_con=54.4, ext_con=2.5, rev_pot=-77)
        self.the_props.set_ion(ion='ca', int_con=5e-5, ext_con=2, rev_pot=132.5)

        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def probes(self, gid):
        return self.the_probes

    def connections_on(self, gid):
        return []

    def gap_junction_on(self, gid):
        return []

    def event_generators(self, gid):
        return []

    def global_properties(self, gid):
        return self.the_props

recipe = single_recipe(cell, [probe])

# (4) Create an execution context

context = arbor.context()

# (5) Create a domain decomposition

domains = arbor.partition_load_balance(recipe, context)

# (6) Create a simulation

sim = arbor.simulation(recipe, domains, context)

# Instruct the simulation to record the spikes and sample the probe

sim.record(arbor.spike_recording.all)

probe_id = arbor.cell_member(0,0)
handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

# (7) Run the simulation

sim.run(tfinal=100, dt=0.025)

# (8) Print or display the results

spikes = sim.spikes()
print(len(spikes), 'spikes recorded:')
for s in spikes:
    print(s)

data = []
meta = []
for d, m in sim.samples(handle):
    data.append(d)
    meta.append(m)

df = pandas.DataFrame()
for i in range(len(data)):
    df = df.append(pandas.DataFrame({'t/ms': data[i][:, 0], 'U/mV': data[i][:, 1], 'Location': str(meta[i]), 'Variable':'voltage'}))
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_recipe_result.svg')

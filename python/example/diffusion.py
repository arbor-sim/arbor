#!/usr/bin/env python3

import arbor as A
import pandas as pd
import seaborn as sns

class recipe (A.recipe):
    def __init__(self, cell, probes):
        A.recipe.__init__(self)
        self.the_cell = cell
        self.the_probes = probes
        self.the_props = A.neuron_cable_properties()
        self.the_cat = A.default_catalogue()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

tree = A.segment_tree()
tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(  3, 0, 0, 3), tag=1)
tree.append(A.mnpos, A.mpoint( 3, 0, 0, 1), A.mpoint(303, 0, 0, 1), tag=3)

lbl = A.label_dict({'soma': '(tag 1)',
                    'dend': '(tag 3)',
                    'locs': '''(sum (on-branches 0.00)
                                    (on-branches 0.25)
                                    (on-branches 0.50)
                                    (on-branches 0.75)
                                    (on-branches 1.00))'''})

dec = A.decor()
dec.set_property(Vm=-40)
dec.paint('"soma"', A.density('hh'))

# Set up ion diffusion
# TODO(TH) figure out diff scale
dec.set_ion('na', int_con=10, ext_con=140, rev_pot=50, diff=0.02)
dec.paint('"soma"', ion_name="na", int_con=100.0, diff=0.05)

prb = [A.cable_probe_ion_int_concentration('"locs"', 'na')]
cel = A.cable_cell(tree, lbl, dec)
rec = recipe(cel, prb)
ctx = A.context()
dom = A.partition_load_balance(rec, ctx)
sim = A.simulation(rec, dom, ctx)
hdl = sim.sample((0, 0), A.regular_schedule(0.1))
sim.run(tfinal=30)

# Plot ion concentrations
sns.set_theme()
for d, m in sim.samples(hdl):
    df = pd.DataFrame({'t/ms': d[:, 0], 'Na': d[:, 1]})
    sns.relplot(data=df, kind="line", x="t/ms", y="Na", ci=None)

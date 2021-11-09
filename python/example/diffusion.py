#!/usr/bin/env python3

import arbor as A
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
s = tree.append(A.mnpos, A.mpoint(-3, 0, 0, 3), A.mpoint(  3, 0, 0, 3), tag=1)
_ = tree.append(s,       A.mpoint( 3, 0, 0, 1), A.mpoint(303, 0, 0, 1), tag=3)

dec = A.decor()
dec.set_property(Vm=-40)
dec.paint('(tag 1)', A.density('hh'))

# Set up ion diffusion
# TODO(TH) figure out diff scale
dec.set_ion('na', int_con=10, ext_con=140, rev_pot=50, diff=1e-10)
dec.paint('(tag 1)', ion_name="na", int_con=100.0)

prb = [A.cable_probe_ion_int_concentration('(join (location 0 0) (location 0 0.25) (location 0 0.5) (location 0 0.75) (location 0 1.0))', 'na'), ]
cel = A.cable_cell(tree, A.label_dict(), dec)
rec = recipe(cel, prb)
ctx = A.context()
dom = A.partition_load_balance(rec, ctx)
sim = A.simulation(rec, dom, ctx)
hdl = sim.sample((0, 0), A.regular_schedule(0.1))
sim.run(tfinal=30)

# Plot ion concentrations
sns.set_theme()
fg, ax = plt.subplots()
df = pd.DataFrame()
for d, m in sim.samples(hdl):
    lbl = f'Na_i@{m}'
    nai = d[:, 1]
    df[lbl] = nai
    df['t/ms'] = d[:, 0]
    print(f"{lbl}: {nai.min()}--{nai.max()}")
print(df.columns)
sns.scatterplot(data=df, ax=ax)
fg.savefig('results.pdf')

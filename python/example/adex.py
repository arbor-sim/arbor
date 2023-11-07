#!/usr/bin/env python3

import arbor as A
import pandas as pd
import seaborn as sns

T = 100
dt = 0.05

class recipe(A.recipe):

    def __init__(self):
        A.recipe.__init__(self)
        self.weight = 10

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.adex

    def cell_description(self, _):
        cell = A.adex_cell("src", "tgt")
        return cell

    def event_generators(self, _):
        return [A.event_generator("tgt", self.weight, A.regular_schedule(8))]

    def probes(self, _):
        return [A.adex_probe_voltage("Um"), A.adex_probe_adaption("w")]

rec = recipe()
sim = A.simulation(rec)

sch = A.regular_schedule(dt)
hUm  = sim.sample((0, "Um"), sch)
hw   = sim.sample((0, "w"), sch)

sim.run(T, dt)

df_list = []

Um, _ = sim.samples(hUm)[0]
W, _ = sim.samples(hw)[0]
df = pd.DataFrame(
    {"t/ms": Um[:, 0], "U/mV": Um[:, 1], "W/nA": W[:, 1]}
    )

sns.relplot(data=df, x='t/ms', y='U/mV', kind="line").savefig("adex_results.pdf")

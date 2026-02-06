#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

T = 100 * U.ms
dt = 0.05 * U.ms


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
        cell.b = 0.1 * U.nA
        return cell

    def event_generators(self, _):
        return [A.event_generator("tgt", self.weight, A.regular_schedule(8 * U.ms))]

    def probes(self, _):
        return [A.adex_probe_voltage("Um"), A.adex_probe_adaption("w")]


rec = recipe()
sim = A.simulation(rec)

sch = A.regular_schedule(dt)
hUm = sim.sample((0, "Um"), sch)
hw = sim.sample((0, "w"), sch)

sim.record(A.spike_recording.all)

sim.run(T, dt)

spikes = sim.spikes()

Um, _ = sim.samples(hUm)[0]
W, _ = sim.samples(hw)[0]

colors = sns.color_palette()

fg, ax = plt.subplots()
V_th = rec.cell_description(0).V_th.value_as(U.mV)
ax.plot(Um[:, 0], Um[:, 1], label="$U_m/mV$", color=colors[0])
ax.scatter(spikes["time"], V_th * np.ones_like(spikes["time"]), color=colors[0])
ax.set_ylabel("$U_m/mV$", color=colors[0])
ax.tick_params(axis="y", labelcolor=colors[0])
ax.set_xlabel("$t/ms$")
ax.set_xlim(0, T.value)
ax = ax.twinx()
ax.set_ylabel("$W/nA$", color=colors[2])
ax.plot(W[:, 0], W[:, 1], label="$w/nA$", color=colors[2])
ax.tick_params(axis="y", labelcolor=colors[2])
fg.savefig("adex_results.pdf")

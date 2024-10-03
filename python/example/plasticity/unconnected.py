#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

# global parameters
# total runtime [ms]
T = 100
# one interval [ms]
dT = 10
# number of intervals
nT = int((T + dT - 1)//dT)
# numerical time step [ms]
dt = 0.1

class unconnected(A.recipe):

    def __init__(self, N) -> None:
        super().__init__()
        self.N = N
        # Cell prototype
        self.cell = A.lif_cell("src", "tgt")
        # random seed [0, 100]
        self.seed = 42
        # event generator parameters
        self.gen_weight = 20
        self.gen_freq = 1 * U.kHz

    def num_cells(self) -> int:
        return self.N

    def event_generators(self, gid: int) -> list[Any]:
        return [A.event_generator("tgt",
                                  self.gen_weight,
                                  A.poisson_schedule(freq=self.gen_freq,
                                                     seed=self.cell_seed(gid)))]

    def cell_description(self, gid: int) -> Any:
        return self.cell

    def cell_kind(self, gid: int) -> A.cell_kind:
        return A.cell_kind.lif

    def cell_seed(self, gid):
        return self.seed + gid*100

if __name__ == "__main__":
    rec = unconnected(10)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)

    t = 0
    while t < T:
        t += dT
        sim.run(t * U.ms, dt * U.ms)

    # Extract spikes
    times = []
    gids = []
    rates = np.zeros(shape=(nT, rec.num_cells()))
    for (gid, _), time in sim.spikes():
        times.append(time)
        gids.append(gid)
        it = int(time // dT)
        rates[it, gid] += 1

    fg, ax = plt.subplots()
    ax.scatter(times, gids, c=gids)
    ax.set_xlabel('Time $(t/ms)$')
    ax.set_ylabel('GID')
    ax.set_xlim(0, T)
    fg.savefig('01-raster.pdf')

    fg, ax = plt.subplots()
    ax.plot(np.arange(nT), rates)
    ax.set_xlabel('Interval')
    ax.set_ylabel('Rate $(kHz)$')
    ax.set_xlim(0, nT)
    fg.savefig('01-rates.pdf')

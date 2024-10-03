#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from unconnected import unconnected

# global parameters
# total runtime [ms]
T = 100
# one interval [ms]
dT = 10
# number of intervals
nT = int((T + dT - 1)//dT)
# numerical time step [ms]
dt = 0.1

class random_network(unconnected):

    def __init__(self, N) -> None:
        super().__init__(N)
        self.syn_weight = 80
        self.syn_delay = 0.5 * U.ms
        self.max_inc = 4
        self.max_out = 4
        # format [to, from]
        self.connections = np.zeros(shape=(N, N), dtype=np.uint8)
        self.inc = np.zeros(N, np.uint8)
        self.out = np.zeros(N, np.uint8)

    def connections_on(self, gid: int):
        return [A.connection((source, "src"), "tgt", self.syn_weight, self.syn_delay)
                for source in range(self.N)
                for _ in range(self.connections[gid, source])
                ]

    def add_connection(self, src: int, tgt: int) -> bool:
        if tgt == src or self.inc[tgt] >= self.max_inc or self.out[src] >= self.max_out:
            return False
        self.inc[tgt] += 1
        self.out[src] += 1
        self.connections[tgt, src] += 1
        return True

    def del_connection(self, src: int, tgt: int) -> bool:
        if tgt == src or self.connections[tgt, src] <= 0:
            return False
        self.inc[tgt] -= 1
        self.out[src] -= 1
        self.connections[tgt, src] -= 1
        return True

    def rewire(self):
        tries = self.N*self.N*self.max_inc*self.max_out
        while tries > 0 and self.inc.sum() < self.N*self.max_inc and self.out.sum() < self.N*self.max_out:
            src, tgt = np.random.randint(self.N, size=2, dtype=int)
            self.add_connection(src, tgt)
            tries -= 1


if __name__ == "__main__":
    rec = random(10)
    rec.rewire()
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
    fg.savefig('02-raster.pdf')

    fg, ax = plt.subplots()
    ax.plot(np.arange(nT), rates)
    ax.set_xlabel('Interval')
    ax.set_ylabel('Rate $(kHz)$')
    ax.set_xlim(0, nT)
    fg.savefig('02-rates.pdf')

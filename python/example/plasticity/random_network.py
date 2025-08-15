#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import numpy as np

from util import plot_spikes, plot_network
from unconnected import unconnected

# global parameters
# cell count
N = 10
# total runtime [ms]
T = 1000
# one interval [ms]
t_interval = 10
# numerical time step [ms]
dt = 0.1


class random_network(unconnected):
    def __init__(self, N) -> None:
        super().__init__(N)
        self.syn_weight = 80
        self.syn_delay = 0.5 * U.ms
        # format [to, from]
        self.connections = np.zeros(shape=(N, N), dtype=np.uint8)
        self.inc = np.zeros(N, np.uint8)
        self.out = np.zeros(N, np.uint8)
        self.max_inc = 4
        self.max_out = 4

    def connections_on(self, gid: int):
        return [
            A.connection((source, "src"), "tgt", self.syn_weight, self.syn_delay)
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
        tries = self.N * self.N * self.max_inc * self.max_out
        while (
            tries > 0
            and self.inc.sum() < self.N * self.max_inc
            and self.out.sum() < self.N * self.max_out
        ):
            src, tgt = np.random.randint(self.N, size=2, dtype=int)
            self.add_connection(src, tgt)
            tries -= 1


if __name__ == "__main__":
    rec = random_network(10)
    rec.rewire()
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)
    t = 0
    while t < T:
        t += t_interval
        sim.run(t * U.ms, dt * U.ms)

    plot_network(rec, prefix="02-")
    plot_spikes(sim, N, t_interval, T, prefix="02-")

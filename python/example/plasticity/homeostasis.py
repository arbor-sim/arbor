#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from util import plot_spikes
from random_network import random_network

# global parameters
# total runtime [ms]
T = 10000
# one interval [ms]
t_interval = 100
# numerical time step [ms]
dt = 0.1


def randrange(n: int):
    res = np.arange(n, dtype=int)
    np.random.shuffle(res)
    return res


np.random.seed = 23


class homeostatic_network(random_network):
    def __init__(self, N) -> None:
        super().__init__(N)
        self.max_inc = 8
        self.max_out = 8
        # setpoint rate in kHz
        self.setpoint = 0.1
        # sensitivty towards deviations from setpoint
        self.alpha = 200


if __name__ == "__main__":
    rec = homeostatic_network(10)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)

    print("Initial network:")
    print(rec.inc)
    print(rec.out)
    print(rec.connections)

    t = 0
    while t < T:
        sim.run((t + t_interval) * U.ms, dt * U.ms)
        if t < T / 2:
            t += t_interval
            continue
        n = rec.num_cells()
        rates = np.zeros(n)
        for (gid, _), time in sim.spikes():
            if time < t:
                continue
            rates[gid] += 1
        rates /= t_interval  # kHz
        dC = ((rec.setpoint - rates) * rec.alpha).astype(int)
        unchangeable = set()
        added = []
        deled = []
        for tgt in randrange(n):
            if dC[tgt] == 0:
                continue
            for src in randrange(n):
                if dC[tgt] > 0 and rec.add_connection(src, tgt):
                    added.append((src, tgt))
                    break
                elif dC[tgt] < 0 and rec.del_connection(src, tgt):
                    deled.append((src, tgt))
                    break
                unchangeable.add(tgt)
        sim.update(rec)
        print(f" * t={t:>4} f={rates} [!] {list(unchangeable)} [+] {added} [-] {deled}")
        t += t_interval

    print("Final network:")
    print(rec.inc)
    print(rec.out)
    print(rec.connections)

    plot_spikes(
        sim,
        rec.num_cells(),
    )

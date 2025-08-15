#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import numpy as np

from util import plot_spikes, plot_network, randrange
from random_network import random_network

# global parameters
# cell count
N = 10
# total runtime [ms]
T = 10000
# one interval [ms]
t_interval = 100
# numerical time step [ms]
dt = 0.1
# Set seed for numpy
np.random.seed = 23
# setpoint rate in kHz
setpoint_rate = 0.1
# sensitivty towards deviations from setpoint
sensitivity = 200


class homeostatic_network(random_network):
    def __init__(self, N, setpoint_rate, sensitivity) -> None:
        super().__init__(N)
        self.max_inc = 8
        self.max_out = 8
        self.setpoint = setpoint_rate
        self.alpha = sensitivity


if __name__ == "__main__":
    rec = homeostatic_network(N, setpoint_rate, sensitivity)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)

    plot_network(rec, prefix="03-initial-")

    t = 0
    while t < T:
        sim.run((t + t_interval) * U.ms, dt * U.ms)
        if t < T / 2:
            t += t_interval
            continue
        rates = np.zeros(N)
        for (gid, _), time in sim.spikes():
            if time < t:
                continue
            rates[gid] += 1
        rates /= t_interval  # kHz
        dC = ((rec.setpoint - rates) * rec.alpha).astype(int)
        unchangeable = set()
        added = []
        deled = []
        for tgt in randrange(N):
            if dC[tgt] == 0:
                continue
            for src in randrange(N):
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

    plot_network(rec, prefix="03-final-")
    plot_spikes(sim, N, t_interval, T, prefix="03-")

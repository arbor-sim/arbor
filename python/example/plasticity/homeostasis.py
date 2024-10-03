#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from typing import Any
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np

from random_network import random_network

# global parameters
# total runtime [ms]
T = 10000
# one interval [ms]
dT = 100
# number of intervals
nT = int((T + dT - 1)//dT)
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
        sim.run((t + dT) * U.ms, dt * U.ms)
        if t < T/2:
            t += dT
            continue
        n = rec.num_cells()
        rates = np.zeros(n)
        for (gid, _), time in sim.spikes():
            if time < t:
                continue
            rates[gid] += 1
        rates /= dT # kHz
        dC = ((rec.setpoint - rates)*rec.alpha).astype(int)
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
        t += dT

    print("Final network:")
    print(rec.inc)
    print(rec.out)
    print(rec.connections)

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
    fg.savefig('03-raster.pdf')

    fg, ax = plt.subplots()
    ax.plot(np.arange(nT), rates/dT)
    ax.plot(np.arange(nT), savgol_filter(rates.mean(axis=1)/dT, window_length=5, polyorder=2), color='0.8', lw=4, label='Mean rate')
    ax.axhline(0.1, label='Setpoint', lw=2, c='0.4')
    ax.legend()
    ax.set_xlabel('Interval')
    ax.set_ylabel('Rate $(kHz)$')
    ax.set_xlim(0, nT)
    fg.savefig('03-rates.pdf')
    fg.savefig('03-rates.png')
    fg.savefig('03-rates.svg')

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def plot_spikes(sim, n_cells, t_interval, T, prefix=""):
    # number of intervals
    n_interval = int((T + t_interval - 1) // t_interval)
    print(n_interval, T, t_interval)

    # Extract spikes
    times = []
    gids = []
    rates = np.zeros(shape=(n_interval, n_cells))
    for (gid, _), time in sim.spikes():
        times.append(time)
        gids.append(gid)
        it = int(time // t_interval)
        rates[it, gid] += 1

    fg, ax = plt.subplots()
    ax.scatter(times, gids, c=gids)
    ax.set_xlabel("Time $(t/ms)$")
    ax.set_ylabel("GID")
    ax.set_xlim(0, T)
    fg.savefig(f"{prefix}raster.pdf")
    fg.savefig(f"{prefix}raster.png")
    fg.savefig(f"{prefix}raster.svg")

    ts = np.arange(n_interval) * t_interval
    mean_rate = savgol_filter(
        rates.mean(axis=1) / t_interval, window_length=5, polyorder=2
    )
    fg, ax = plt.subplots()
    ax.plot(ts, rates)
    ax.plot(ts, mean_rate, color="0.8", lw=4, label="Mean rate")
    ax.set_xlabel("Time $(t/ms)$")
    ax.legend()
    ax.set_ylabel("Rate $(kHz)$")
    ax.set_xlim(0, T)
    fg.savefig(f"{prefix}rates.pdf")
    fg.savefig(f"{prefix}rates.png")
    fg.savefig(f"{prefix}rates.svg")

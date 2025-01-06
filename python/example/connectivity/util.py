#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import arbor as A


def plot_network(rec, prefix="", graph=True):
    fg, ax = plt.subplots()
    n = rec.num_cells()
    mat = np.zeros((n, n), dtype=int)

    ctx = A.context()
    net = A.generate_network_connections(rec, ctx)

    for conn in net:
        i = conn.source.gid
        j = conn.target.gid
        mat[i, j] += 1
    ax.matshow(mat)
    fg.savefig(f"{prefix}matrix.pdf")
    fg.savefig(f"{prefix}matrix.png")
    fg.savefig(f"{prefix}matrix.svg")

    if graph:
        fg, ax = plt.subplots()
        g = nx.MultiDiGraph()
        g.add_nodes_from(np.arange(n))
        for i in range(n):
            for j in range(n):
                for _ in range(mat[i, j]):
                    g.add_edge(i, j)
        nx.draw(g, with_labels=True, font_weight="bold")
        fg.savefig(f"{prefix}graph.pdf")
        fg.savefig(f"{prefix}graph.png")
        fg.savefig(f"{prefix}graph.svg")


def plot_spikes(sim, T, N, prefix=""):
    # Extract spikes
    times = []
    gids = []
    for (gid, _), time in sim.spikes():
        times.append(time)
        gids.append(gid + 1)

    fg, ax = plt.subplots()
    ax.scatter(times, gids, c=gids)
    ax.set_xlabel("Time $(t/ms)$")
    ax.set_ylabel("GID")
    ax.set_xlim(0, T)
    ax.set_ylim(0, N + 1)
    fg.savefig(f"{prefix}raster.pdf")
    fg.savefig(f"{prefix}raster.png")
    fg.savefig(f"{prefix}raster.svg")

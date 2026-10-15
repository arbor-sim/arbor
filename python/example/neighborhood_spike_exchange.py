#!/usr/bin/env python3

"""
Neighbourhood spike exchange.

Every rank holds the same number of LIF cells; the ranks form a ring, which is
split into levels by distance. With --widths 1 4 there are
  - the own rank,
  - level 1: the ranks 1 step away on the ring,
  - level 2: the ranks 2 to 4 steps away,
  - far: all other ranks.
Each cell receives --in-degrees connections from cells in each of these
groups, with the respective --delays; delays should grow with distance.

The model is simulated three times: with the default exchange, which runs a
global all-to-all every epoch (half the shortest delay), and twice with
neighbourhood exchange. There, each level is exchanged with a sparse MPI
neighbourhood all-gather and the far ranks with a global all-to-all, each only
as often as the delays from that group require. The levels are either the ones
described above, or chosen by arbor from the connection delays. All runs must
produce the same spikes.

Run with MPI, e.g. `mpirun -n 16 python neighborhood_spike_exchange.py`.
"""

import argparse
import random
import time

import arbor as A
from arbor import units as U


def ring_distance(a, b, num_ranks):
    d = abs(a - b)
    return min(d, num_ranks - d)


def group_of(a, b, num_ranks, widths):
    """Group of rank b as seen from rank a: 0 own, k for level k, len(widths)+1 for far."""
    d = ring_distance(a, b, num_ranks)
    if d == 0:
        return 0
    for level, width in enumerate(widths):
        if d <= width:
            return level + 1
    return len(widths) + 1


class ring_recipe(A.recipe):
    def __init__(self, num_ranks, args):
        super().__init__()
        self.num_ranks = num_ranks
        self.args = args

    def num_cells(self):
        return self.num_ranks * self.args.cells_per_rank

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def cell_description(self, gid):
        return A.lif_cell("src", "tgt")

    def connections_on(self, gid):
        args = self.args
        rank = gid // args.cells_per_rank
        rng = random.Random(gid)

        # The ranks in each group, as seen from this cell's rank.
        groups = [[] for _ in range(len(args.widths) + 2)]
        for r in range(self.num_ranks):
            groups[group_of(rank, r, self.num_ranks, args.widths)].append(r)

        conns = []
        for ranks, in_degree, delay in zip(groups, args.in_degrees, args.delays):
            # Groups can be empty if the ring is small.
            if not ranks:
                continue
            for _ in range(in_degree):
                src = rng.choice(ranks) * args.cells_per_rank + rng.randrange(
                    args.cells_per_rank
                )
                # One in four connections is inhibitory.
                weight = -60 if len(conns) % 4 == 0 else 20
                conns.append(A.connection((src, "src"), "tgt", weight, delay * U.ms))
        return conns

    def event_generators(self, gid):
        sched = A.poisson_schedule(freq=0.8 * U.kHz, seed=gid)
        return [A.event_generator("tgt", 30, sched)]


def ring_neighborhoods(rank, num_ranks, widths):
    """The ranks this rank receives from, one neighbourhood per level."""
    # Every rank must pass all levels, even those that are empty on a small ring.
    levels = [[] for _ in widths]
    for r in range(num_ranks):
        g = group_of(rank, r, num_ranks, widths)
        if 1 <= g <= len(widths):
            levels[g - 1].append(r)
    return A.neighborhood_spike_exchange(levels)


class result:
    """Spikes of this rank, timings, and the exchange levels arbor uses on this rank."""

    def __init__(self, sim, init, run):
        self.spikes = sim.spikes().tolist()
        self.init = init
        self.run = run
        self.levels = sim.spike_exchange_levels()
        # Collective, so every rank calls it.
        self.epoch = sim.max_epoch_length()


def simulate(recipe, context, args, spike_exchange=None):
    t0 = time.perf_counter()
    sim = A.simulation(recipe, context, spike_exchange=spike_exchange)
    sim.record(A.spike_recording.local)
    t1 = time.perf_counter()
    sim.run(args.tfinal * U.ms, args.dt * U.ms)
    t2 = time.perf_counter()
    return result(sim, t1 - t0, t2 - t1)


def describe_levels(title, r):
    """The levels arbor uses on this rank, as text."""
    lines = [f"{title}, in epochs of {r.epoch:g} ms:"]
    for i, level in enumerate(r.levels):
        name = "far" if level.far else f"level {i + 1}"
        ranks = level.ranks
        line = f"  {name + ':':10} {len(ranks)} rank(s)"
        if ranks:
            # Long lists of ranks are cut short.
            line += (
                " ["
                + ", ".join(map(str, ranks[:8]))
                + (", ...]" if len(ranks) > 8 else "]")
            )
        if level.period:
            line += f", every {level.period} epoch(s) ({level.period * r.epoch:g} ms)"
        else:
            line += ", unused"
        lines.append(line)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Neighbourhood spike exchange")
    parser.add_argument("--cells-per-rank", type=int, default=200)
    parser.add_argument(
        "--widths",
        type=int,
        nargs="*",
        default=[1, 4],
        help="outer ring distance of each level, increasing",
    )
    parser.add_argument(
        "--in-degrees",
        type=int,
        nargs="+",
        default=[40, 30, 20, 10],
        help="connections per cell from the own rank, each level, and far ranks",
    )
    parser.add_argument(
        "--delays",
        type=float,
        nargs="+",
        default=[1, 2, 5, 20],
        help="delays [ms] of these connections",
    )
    parser.add_argument("--tfinal", type=float, default=100, help="[ms]")
    parser.add_argument("--dt", type=float, default=0.025, help="[ms]")
    args = parser.parse_args()

    if any(w <= 0 for w in args.widths) or sorted(set(args.widths)) != args.widths:
        parser.error("widths must be positive and strictly increasing")
    num_groups = len(args.widths) + 2
    if len(args.in_degrees) != num_groups or len(args.delays) != num_groups:
        parser.error(
            f"need {num_groups} in-degrees and delays: own rank, each level, and far ranks"
        )
    return args


def main():
    args = parse_args()

    if A.config()["mpi"]:
        A.mpi_init()
        context = A.context(mpi=A.mpi_comm())
    else:
        context = A.context()
    rank, num_ranks = context.rank, context.ranks

    def log(*msg):
        if rank == 0:
            print(*msg)

    recipe = ring_recipe(num_ranks, args)
    exchange = ring_neighborhoods(rank, num_ranks, args.widths)

    log(f"ranks: {num_ranks}, cells: {recipe.num_cells()}\n")

    # The model as seen from rank 0: groups of ranks with their connections.
    log("model:")
    sizes = [1] + [len(level) for level in exchange.neighborhoods]
    sizes.append(num_ranks - sum(sizes))
    names = ["own rank"] + [f"level {i + 1}" for i in range(len(args.widths))] + ["far"]
    for g, (name, size) in enumerate(zip(names, sizes)):
        line = f"  {name + ':':10} {size} rank(s)"
        if size:
            line += f", {args.in_degrees[g]} connections per cell, delay {args.delays[g]:g} ms"
        log(line)
    log()

    default = simulate(recipe, context, args)
    given = simulate(recipe, context, args, exchange)
    automatic = A.neighborhood_spike_exchange(automatic=True)
    auto = simulate(recipe, context, args, automatic)

    log(describe_levels("given levels of rank 0", given))
    log(describe_levels("automatic levels of rank 0", auto))
    log()
    for name, r in [
        ("all-to-all:", default),
        ("neighborhood:", given),
        ("automatic:", auto),
    ]:
        log(
            f"{name:13} {len(r.spikes)} local spikes, init {r.init:.3f} s, run {r.run:.3f} s"
        )

    # Each rank checks its own spikes.
    if default.spikes != given.spikes or default.spikes != auto.spikes:
        raise RuntimeError(
            f"rank {rank}: the exchange methods produced different spikes"
        )

    if A.config()["mpi"]:
        A.mpi_finalize()


if __name__ == "__main__":
    main()

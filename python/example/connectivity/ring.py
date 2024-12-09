#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from util import plot_spikes, plot_network
from unconnected import unconnected


# global parameters
# cell count
N = 5
# total runtime [ms]
T = 1000
# numerical time step [ms]
dt = 0.1


class ring(unconnected):
    def __init__(self, N) -> None:
        super().__init__(N)

    def network_description(self):
        # network structure
        wraps = f"(intersect (source-cell {self.N - 1}) (target-cell 0))"
        cells = f"(gid-range 0 {self.N})"
        chain = f"(chain {cells})"
        ring = f"(join {chain} {wraps})"
        # parameters
        weight = "(scalar 199.99999219)"
        delay = "(scalar 0.5)"
        return A.network_description(ring, weight, delay, {})


if __name__ == "__main__":
    ctx = A.context()
    rec = ring(N)
    sim = A.simulation(rec, ctx)
    sim.record(A.spike_recording.all)
    sim.run(T * U.ms, dt * U.ms)
    plot_spikes(sim, T, N, prefix="02-")
    plot_network(rec, prefix="02-")

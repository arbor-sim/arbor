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


class all2all(unconnected):
    def __init__(self, N) -> None:
        super().__init__(N)

    def network_description(self):
        # network structure
        full = f"(intersect (inter-cell) (source-cell (gid-range 0 {self.N})) (target-cell (gid-range 0 {self.N})))"
        # parameters
        weight = "(scalar 125)"
        delay = "(scalar 0.5)"
        return A.network_description(full, weight, delay, {})


if __name__ == "__main__":
    rec = all2all(N)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)
    sim.run(T * U.ms, dt * U.ms)
    plot_spikes(sim, T, N, prefix="03-")
    plot_network(rec, prefix="03-")

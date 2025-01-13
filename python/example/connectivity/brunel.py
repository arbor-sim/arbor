#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from util import plot_spikes, plot_network
from unconnected import unconnected


# global parameters
# cell count
N = 125
# total runtime [ms]
T = 100
# numerical time step [ms]
dt = 0.1


class brunel(unconnected):
    def __init__(self, N) -> None:
        super().__init__(N)
        # excitatory population: first 80% of the gids
        self.n_exc = int(0.8 * N)
        # inhibitory population: the remainder
        self.n_inh = N - self.n_exc
        # seed for random number generation
        self.seed = 42
        # excitatory weight
        self.weight = 100
        # relative weight of inhibitory connections
        self.g = 0.8
        # probability of connecting any two neurons
        self.p = 0.1

    def network_description(self):
        rand = f"""(intersect (inter-cell)
                                (random {self.seed} {self.p}))"""
        inh = f"(gid-range {self.n_exc} {self.N})"
        weight = f"""(if-else (source-cell {inh})
                              (scalar {self.g * self.weight})
                              (scalar {self.weight}))"""
        delay = "(scalar 0.5)"
        return A.network_description(rand, weight, delay, {})


if __name__ == "__main__":
    rec = brunel(N)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)
    sim.run(T * U.ms, dt * U.ms)
    plot_spikes(sim, T, N, prefix="04-")
    plot_network(rec, prefix="04-", graph=True)

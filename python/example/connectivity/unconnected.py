#!/usr/bin/env python3

import arbor as A
from arbor import units as U
from util import plot_spikes

# global parameters
# cell count
N = 5
# total runtime [ms]
T = 1000
# numerical time step [ms]
dt = 0.1


class unconnected(A.recipe):
    def __init__(self, N) -> None:
        super().__init__()
        self.N = N
        # Cell prototype
        self.cell = A.lif_cell("src", "tgt")
        # random seed [0, 100]
        self.seed = 42
        # event generator parameters
        self.gen_weight = 20
        self.gen_freq = 2.5 * U.kHz

    def num_cells(self) -> int:
        return self.N

    def event_generators(self, gid: int):
        if gid >= 1:
            return []
        seed = self.seed + gid * 100
        return [
            A.event_generator(
                "tgt",
                self.gen_weight,
                A.poisson_schedule(freq=self.gen_freq, seed=seed),
            )
        ]

    def cell_description(self, gid: int):
        return self.cell

    def cell_kind(self, gid: int) -> A.cell_kind:
        return A.cell_kind.lif


if __name__ == "__main__":
    rec = unconnected(N)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)
    sim.run(T * U.ms, dt * U.ms)
    plot_spikes(sim, T, N, prefix="01-")

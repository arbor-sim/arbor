import arbor as A
from arbor import units as U


class recipe(A.recipe):
    def __init__(self, *, n_cell=4):
        A.recipe.__init__(self)
        self.n_cell = n_cell

    def num_cells(self):
        return self.n_cell

    def cell_kind(self, _):
        return A.cell_kind.lif

    def cell_description(self, _):
        return A.lif_cell("src", "tgt")

    def connections_on(self, gid):
        src = (gid - 1) % self.n_cell
        return [A.connection((src, "src"), "tgt", weight=200, delay=0.5 * U.ms)]

    def event_generators(self, gid):
        if gid == 0:
            return [
                A.event_generator(
                    target="tgt", weight=200, sched=A.explicit_schedule([0.1 * U.ms])
                )
            ]
        return []


if __name__ == "__main__":
    rec = recipe(n_cell=4)
    sim = A.simulation(rec)
    sim.record(A.spike_recording.all)
    sim.run(10 * U.ms, 10 * U.us)
    for (gid, _), time in sim.spikes():
        print(f"{time:5.3}ms gid={gid:3d}")

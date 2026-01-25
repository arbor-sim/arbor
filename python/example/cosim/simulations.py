# import our previous code
from wilson_cowan import step, Params
from ring import recipe
from mpi import world, group, inter
import arbor as A
from arbor import units as U
import numpy as np


assert world.size == 2, "Need exactly two ranks"


dt_arb = 0.005  # ms
dt_nmm = 0.01  # ms
dt_com = 0.5  # ms
T = 100  # ms


if __name__ == "__main__":
    if world.rank == 0:
        print(f"[NMM] {world.rank:2d}/{world.size:2d} {group.rank:2d}/{group.size:2d}")

        y = np.array([0.2, 0.1])
        ps = Params()

        t = 0
        ts = [t]
        Es = [y[0]]
        Is = [y[1]]
        while t < T:
            epoch = 0
            while epoch < dt_com:
                y = step(0, dt_nmm, y, ps)
                t += dt_nmm
                epoch += dt_nmm
                ts.append(t)
                Es.append(y[0])
                Is.append(y[1])
    else:  # rank != 0
        print(f"[ARB] {world.rank:2d}/{world.size:2d} {group.rank:2d}/{group.size:2d}")
        rec = recipe(n_cell=4)
        sim = A.simulation(rec)
        sim.record(A.spike_recording.all)
        sim.run(T * U.ms, dt_arb * U.ms)

        rates = np.zeros(int(T / dt_com))
        for _, time in sim.spikes():
            idx = int(time / dt_com)
            rates[idx] += 1
        print(rates)

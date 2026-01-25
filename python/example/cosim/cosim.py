import arbor as A
from arbor import units as U
import numpy as np
# import our previous code
from wilson_cowan import step, Params
from ring import recipe
from mpi import world, group, inter


assert world.size == 2, "Need exactly two ranks"


dt_arb = 0.005  # ms
dt_nmm = 0.01  # ms
dt_com = 0.5  # ms
T = 100  # ms


class corecipe(recipe):
    def __init__(self, *, n_cell=16, ext_weight=0.0):
        recipe.__init__(self, n_cell=n_cell)
        self.ext_weight = ext_weight

    def external_connections_on(self, gid):
        # TODO Customization point: assuming all external gids are connected to all internal gids
        return [
            A.external_connection(
                (ext, 0), "tgt", weight=self.ext_weight, delay=dt_com * U.ms
            )
            for ext in range(2)
        ]


if __name__ == "__main__":
    if world.rank == 0:
        print(f"[NMM] {world.rank:2d}/{world.size:2d} {group.rank:2d}/{group.size:2d}")

        y = np.array([0.2, 0.1])
        ps = Params()

        t = 0
        ts = [t]
        Es = [y[0]]
        Is = [y[1]]
        while True:
            print("[NMM] Ask for CTRL")
            msg = A.remote.exchange_ctrl(A.remote.msg_epoch(t, t + dt_com), inter)
            if isinstance(msg, A.remote.msg_abort):
                print(f"[NMM] Arbor sent an abort {msg.reason}")
                world.Abort()
            elif isinstance(msg, A.remote.msg_done):
                print("[NMM] Done")
                break
            elif isinstance(msg, A.remote.msg_epoch):
                print(f"[NMM] Next epoch {msg}")
                from_arb = A.remote.gather_spikes([], inter)

                nmm_steps_per_epoch = int(dt_com / dt_nmm)
                for_e = np.zeros(nmm_steps_per_epoch)
                for_i = np.zeros(nmm_steps_per_epoch)

                for spk in from_arb:
                    gid = spk.gid
                    time = spk.time

                    idx = int((time - t) / dt_nmm) // 2
                    if 0 <= gid <= 11:
                        for_e[idx] += 1
                    elif 12 <= gid <= 16:
                        for_i[idx] += 1
                    else:
                        pass

                epoch = 0
                idx = 0
                while epoch < dt_com / 2:
                    y[0] += 0.01 * for_e[idx]
                    y[1] += 0.005 * for_i[idx]
                    y = step(t, dt_nmm, y, ps)
                    epoch += dt_nmm
                    t += dt_nmm
                    idx += 1
                    ts.append(t)
                    Es.append(y[0])
                    Is.append(y[1])
            else:
                print("[NMM] Unknown message type")
                world.Abort()

        import matplotlib.pyplot as plt

        fg, ax = plt.subplots()
        ax.plot(ts, Es, label="E")
        ax.plot(ts, Is, label="I")
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Time ($t/ms$)")
        ax.legend()
        fg.savefig("wilson-cowan-cosim.svg")

    else:  # rank != 0
        print(f"[ARB] {world.rank:2d}/{world.size:2d} {group.rank:2d}/{group.size:2d}")
        rec = corecipe(n_cell=16)
        ctx = A.context(mpi=group, inter=inter)
        sim = A.simulation(rec, context=ctx)
        sim.record(A.spike_recording.all)
        sim.run(T * U.ms, dt_arb * U.ms)

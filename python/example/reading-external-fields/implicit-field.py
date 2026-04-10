import arbor as A
from arbor import units as U

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

here = Path(__file__).parent

ctr = "(location 0 0.5)"

mrf = A.load_swc_neuron(here / "Acker2008.swc")

dec = (
    A.decor()
    .set_property(Vm=-55 * U.mV)
    .paint('"soma"', A.density("hh"))
    .paint('"dend"', A.density("pas/e=-70"))
)

cvp = A.cv_policy("(every-segment)")

prp = A.neuron_cable_properties()
cat = A.load_catalogue(here / "efields-catalogue.so")
prp.catalogue.extend(cat, "")

e0 = 30.0e-3
omega = 0.05  # 1/ms

for id, seg in enumerate(mrf.segment_tree.segments):
    loc = f"(support (on-components 0.5 (segment {id})))"
    dec.place(
        loc,
        A.synapse(
            f"efield/e0={e0},omega={omega}",
            xp=seg.prox.x,
            xd=seg.dist.x,
            yp=seg.prox.y,
            yd=seg.dist.y,
            zp=seg.prox.z,
            zd=seg.dist.z,
        ),
        label=f"ef{id}",
    )


class recipe(A.recipe):
    def __init__(self):
        A.recipe.__init__(self)

    def num_cells(self):
        return 1

    def cell_kind(self, _):
        return A.cell_kind.cable

    def global_properties(self, kind):
        return prp

    def cell_description(self, _):
        return A.cable_cell(mrf.morphology, dec, mrf.labels, cvp)

    def probes(self, gid):
        return [A.cable_probe_membrane_voltage(ctr, tag="Um")]


if __name__ == "__main__":
    T = 1000
    dt = 0.01
    rec = recipe()
    sim = A.simulation(rec)
    hdl = sim.sample((0, "Um"), A.regular_schedule(10 * dt * U.ms))
    sim.run(T * U.ms, dt * U.ms)
    fg, ax = plt.subplots()
    for data, meta in sim.samples(hdl):
        ax.plot(data[:, 0], data[:, 1], label=meta)
    ax.legend()
    ax.set_xlabel("Time $(ms)$")
    ax.set_ylabel("Membrane potential $(mV)$")
    ax.set_xlim(0, T)
    ax.set_ylim(-120, -20)

    bx = ax.twinx()
    bx.plot(data[:, 0], e0 * np.sin(omega * data[:, 0]), label="E(t)", color="r")
    bx.set_xlim(0, T)
    bx.set_ylim(-0.05, 0.05)
    bx.set_ylabel("Induced Current $(nA)$")

    fg.savefig("external-fields-implicit-field.svg", bbox_inches="tight")

#!/usr/bin/env python3

import arbor as A
from arbor import units as U
import numpy as np
import pandas as pd
import seaborn as sns  # You may have to pip install these.


class single_recipe(A.recipe):
    def __init__(self, dT, n_pairs):
        A.recipe.__init__(self)
        self.dT = dT
        self.n_pairs = n_pairs

        self.the_props = A.neuron_cable_properties()

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        tree = A.segment_tree()
        tree.append(A.mnpos, (-3, 0, 0, 3), (3, 0, 0, 3), tag=1)

        labels = A.label_dict({"soma": "(tag 1)", "center": "(location 0 0.5)"})

        decor = (
            A.decor()
            .set_property(Vm=-40)
            .paint("(all)", A.density("hh"))
            .place('"center"', A.threshold_detector(-10 * U.mV), "detector")
            .place('"center"', A.synapse("expsyn"), "synapse")
            .place(
                '"center"',
                A.synapse("expsyn_stdp", max_weight=1.0),
                "stpd_synapse",
            )
        )

        return A.cable_cell(tree, decor, labels)

    def event_generators(self, gid):
        """two stimuli: one that makes the cell spike, the other to monitor STDP"""

        stimulus_times = np.linspace(50, 500, self.n_pairs) * U.ms

        # strong enough stimulus
        spike = A.event_generator("synapse", 1.0, A.explicit_schedule(stimulus_times))

        # zero weight -> just modify synaptic weight via stdp
        stdp = A.event_generator(
            "stpd_synapse", 0.0, A.explicit_schedule(stimulus_times - self.dT)
        )

        return [spike, stdp]

    def probes(self, gid):
        def mk(s, t):
            return A.cable_probe_point_state(1, "expsyn_stdp", state=s, tag=t)

        return [
            A.cable_probe_membrane_voltage('"center"', "Um"),
            mk("g", "state-g"),
            mk("apost", "state-apost"),
            mk("apre", "state-apre"),
            mk("weight_plastic", "state-weight"),
        ]

    def global_properties(self, kind):
        return self.the_props


def run(dT, n_pairs=1, do_plots=False):
    recipe = single_recipe(dT, n_pairs)

    sim = A.simulation(recipe)

    sim.record(A.spike_recording.all)

    reg_sched = A.regular_schedule(0.1 * U.ms)
    handles = {
        "U": sim.sample((0, "Um"), reg_sched),
        "g": sim.sample((0, "state-g"), reg_sched),
        "apost": sim.sample((0, "state-apost"), reg_sched),
        "apre": sim.sample((0, "state-apre"), reg_sched),
        "weight_plastic": sim.sample((0, "state-weight"), reg_sched),
    }

    sim.run(tfinal=0.6 * U.s)

    if do_plots:
        print("Plotting detailed results ...")
        for var, handle in handles.items():
            data, _ = sim.samples(handle)[0]

            df = pd.DataFrame({"t/ms": data[:, 0], var: data[:, 1]})
            sns.relplot(data=df, kind="line", x="t/ms", y=var, errorbar=None).savefig(
                f"single_cell_stdp_result_{var}.svg"
            )
    weight_plastic, _ = sim.samples(handles["weight_plastic"])[0]
    return weight_plastic[:, 1][-1]


data = np.array([(dT, run(dT * U.ms)) for dT in np.arange(-20, 20, 0.5)])
df = pd.DataFrame({"t/ms": data[:, 0], "dw": data[:, 1]})
print("Plotting results ...")
sns.relplot(data=df, x="t/ms", y="dw", kind="line", errorbar=None).savefig(
    "single_cell_stdp.svg"
)

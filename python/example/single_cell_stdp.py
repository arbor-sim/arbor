#!/usr/bin/env python3

import arbor
import numpy as np
import pandas as pd
import seaborn as sns  # You may have to pip install these.


class single_recipe(arbor.recipe):
    def __init__(self, dT, n_pairs):
        arbor.recipe.__init__(self)
        self.dT = dT
        self.n_pairs = n_pairs

        self.the_props = arbor.neuron_cable_properties()

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        tree = arbor.segment_tree()
        tree.append(
            arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1
        )

        labels = arbor.label_dict({"soma": "(tag 1)", "center": "(location 0 0.5)"})

        decor = (
            arbor.decor()
            .set_property(Vm=-40)
            .paint("(all)", arbor.density("hh"))
            .place('"center"', arbor.threshold_detector(-10), "detector")
            .place('"center"', arbor.synapse("expsyn"), "synapse")
            .place(
                '"center"',
                arbor.synapse("expsyn_stdp", {"max_weight": 1.0}),
                "stpd_synapse",
            )
        )

        return arbor.cable_cell(tree, decor, labels)

    def event_generators(self, gid):
        """two stimuli: one that makes the cell spike, the other to monitor STDP"""

        stimulus_times = np.linspace(50, 500, self.n_pairs)

        # strong enough stimulus
        spike = arbor.event_generator(
            "synapse", 1.0, arbor.explicit_schedule(stimulus_times)
        )

        # zero weight -> just modify synaptic weight via stdp
        stdp = arbor.event_generator(
            "stpd_synapse", 0.0, arbor.explicit_schedule(stimulus_times - self.dT)
        )

        return [spike, stdp]

    def probes(self, gid):
        def mk(w):
            return arbor.cable_probe_point_state(1, "expsyn_stdp", w)

        return [
            arbor.cable_probe_membrane_voltage('"center"'),
            mk("g"),
            mk("apost"),
            mk("apre"),
            mk("weight_plastic"),
        ]

    def global_properties(self, kind):
        return self.the_props


def run(dT, n_pairs=1, do_plots=False):
    recipe = single_recipe(dT, n_pairs)

    sim = arbor.simulation(recipe)

    sim.record(arbor.spike_recording.all)

    reg_sched = arbor.regular_schedule(0.1)
    handles = {
        "U": sim.sample((0, 0), reg_sched),
        "g": sim.sample((0, 1), reg_sched),
        "apost": sim.sample((0, 2), reg_sched),
        "apre": sim.sample((0, 3), reg_sched),
        "weight_plastic": sim.sample((0, 4), reg_sched),
    }

    sim.run(tfinal=600)

    if do_plots:
        print("Plotting detailed results ...")
        for var, handle in handles.items():
            data, meta = sim.samples(handle)[0]

            df = pd.DataFrame({"t/ms": data[:, 0], var: data[:, 1]})
            sns.relplot(data=df, kind="line", x="t/ms", y=var, errorbar=None).savefig(
                f"single_cell_stdp_result_{var}.svg"
            )

    weight_plastic, _ = sim.samples(handles["weight_plastic"])[0]
    return weight_plastic[:, 1][-1]


data = np.array([(dT, run(dT)) for dT in np.arange(-20, 20, 0.5)])
df = pd.DataFrame({"t/ms": data[:, 0], "dw": data[:, 1]})
print("Plotting results ...")
sns.relplot(data=df, x="t/ms", y="dw", kind="line", errorbar=None).savefig(
    "single_cell_stdp.svg"
)

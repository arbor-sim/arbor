#!/usr/bin/env python3

import arbor
import numpy
import pandas
import seaborn  # You may have to pip install these.


class single_recipe(arbor.recipe):
    def __init__(self, dT, n_pairs):
        arbor.recipe.__init__(self)
        self.dT = dT
        self.n_pairs = n_pairs

        self.the_props = arbor.neuron_cable_properties()
        self.the_cat = arbor.default_catalogue()
        self.the_props.register(self.the_cat)

    def num_cells(self):
        return 1

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        tree = arbor.segment_tree()
        tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3),
                    arbor.mpoint(3, 0, 0, 3), tag=1)

        labels = arbor.label_dict({'soma':   '(tag 1)',
                                   'center': '(location 0 0.5)'})

        decor = arbor.decor()
        decor.set_property(Vm=-40)
        decor.paint('(all)', 'hh')

        decor.place('"center"', arbor.spike_detector(-10), "detector")
        decor.place('"center"', 'expsyn', "synapse")

        mech_syn = arbor.mechanism('expsyn_stdp')
        mech_syn.set("max_weight", 1.)

        decor.place('"center"', mech_syn, "stpd_synapse")

        cell = arbor.cable_cell(tree, labels, decor)

        return cell

    def event_generators(self, gid):
        """two stimuli: one that makes the cell spike, the other to monitor STDP
        """

        stimulus_times = numpy.linspace(50, 500, self.n_pairs)

        # strong enough stimulus
        spike = arbor.event_generator("synapse", 1., arbor.explicit_schedule(stimulus_times))

        # zero weight -> just modify synaptic weight via stdp
        stdp = arbor.event_generator("stpd_synapse", 0., arbor.explicit_schedule(stimulus_times - self.dT))

        return [spike, stdp]

    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('"center"'),
                arbor.cable_probe_point_state(1, "expsyn_stdp", "g"),
                arbor.cable_probe_point_state(1, "expsyn_stdp", "apost"),
                arbor.cable_probe_point_state(1, "expsyn_stdp", "apre"),
                arbor.cable_probe_point_state(
                    1, "expsyn_stdp", "weight_plastic")
                ]

    def global_properties(self, kind):
        return self.the_props


def run(dT, n_pairs=1, do_plots=False):
    recipe = single_recipe(dT, n_pairs)

    context = arbor.context()
    domains = arbor.partition_load_balance(recipe, context)
    sim = arbor.simulation(recipe, domains, context)

    sim.record(arbor.spike_recording.all)

    reg_sched = arbor.regular_schedule(0.1)
    handle_mem = sim.sample((0, 0), reg_sched)
    handle_g = sim.sample((0, 1), reg_sched)
    handle_apost = sim.sample((0, 2), reg_sched)
    handle_apre = sim.sample((0, 3), reg_sched)
    handle_weight_plastic = sim.sample((0, 4), reg_sched)

    sim.run(tfinal=600)

    if do_plots:
        print("Plotting detailed results ...")

        for (handle, var) in [(handle_mem, 'U'),
                              (handle_g, "g"),
                              (handle_apost, "apost"),
                              (handle_apre, "apre"),
                              (handle_weight_plastic, "weight_plastic")]:

            data, meta = sim.samples(handle)[0]

            df = pandas.DataFrame({'t/ms': data[:, 0], var: data[:, 1]})
            seaborn.relplot(data=df, kind="line", x="t/ms", y=var,
                            ci=None).savefig('single_cell_stdp_result_{}.svg'.format(var))

    weight_plastic, meta = sim.samples(handle_weight_plastic)[0]

    return weight_plastic[:, 1][-1]


data = numpy.array([(dT, run(dT)) for dT in numpy.arange(-20, 20, 0.5)])
df = pandas.DataFrame({'t/ms': data[:, 0], 'dw': data[:, 1]})
print("Plotting results ...")
seaborn.relplot(data=df, x="t/ms", y="dw", kind="line",
                ci=None).savefig('single_cell_stdp.svg')

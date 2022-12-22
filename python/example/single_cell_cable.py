#!/usr/bin/env python3

import arbor
import argparse
import numpy as np

import pandas
import seaborn  # You may have to pip install these.


class Cable(arbor.recipe):
    def __init__(
        self,
        probes,
        Vm,
        length,
        radius,
        cm,
        rL,
        g,
        stimulus_start,
        stimulus_duration,
        stimulus_amplitude,
        cv_policy_max_extent,
    ):
        """
        probes -- list of probes

        Vm -- membrane leak potential
        length -- length of cable in μm
        radius -- radius of cable in μm
        cm -- membrane capacitance in F/m^2
        rL -- axial resistivity in Ω·cm
        g -- membrane conductivity in S/cm^2

        stimulus_start -- start of stimulus in ms
        stimulus_duration -- duration of stimulus in ms
        stimulus_amplitude -- amplitude of stimulus in nA

        cv_policy_max_extent -- maximum extent of control volume in μm
        """

        arbor.recipe.__init__(self)

        self.the_probes = probes

        self.Vm = Vm
        self.length = length
        self.radius = radius
        self.cm = cm
        self.rL = rL
        self.g = g

        self.stimulus_start = stimulus_start
        self.stimulus_duration = stimulus_duration
        self.stimulus_amplitude = stimulus_amplitude

        self.cv_policy_max_extent = cv_policy_max_extent

        self.the_props = arbor.neuron_cable_properties()

    def num_cells(self):
        return 1

    def num_sources(self, gid):
        return 0

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def probes(self, gid):
        return self.the_probes

    def global_properties(self, kind):
        return self.the_props

    def cell_description(self, gid):
        """A high level description of the cell with global identifier gid.

        For example the morphology, synapses and ion channels required
        to build a multi-compartment neuron.
        """

        tree = arbor.segment_tree()

        tree.append(
            arbor.mnpos,
            arbor.mpoint(0, 0, 0, self.radius),
            arbor.mpoint(self.length, 0, 0, self.radius),
            tag=1,
        )

        labels = arbor.label_dict({"cable": "(tag 1)", "start": "(location 0 0)"})

        decor = (
            arbor.decor()
            .set_property(Vm=self.Vm, cm=self.cm, rL=self.rL)
            .paint('"cable"', arbor.density(f"pas/e={self.Vm}", {"g": self.g}))
            .place(
                '"start"',
                arbor.iclamp(
                    self.stimulus_start, self.stimulus_duration, self.stimulus_amplitude
                ),
                "iclamp",
            )
        )

        policy = arbor.cv_policy_max_extent(self.cv_policy_max_extent)
        decor.discretization(policy)

        return arbor.cable_cell(tree, decor, labels)


def get_rm(g):
    """Return membrane resistivity in Ohm*m^2
    g -- membrane conductivity in S/m^2
    """
    return 1 / g


def get_taum(cm, rm):
    """Return membrane time constant in seconds
    cm -- membrane capacitance in F/m^2
    rm -- membrane resistivity in Ohm*m^2
    """
    return cm * rm


def get_lambdam(a, rm, rL):
    """Return electronic length in m
    a -- cable radius in m
    rm -- membrane resistivity in Ohm*m^2
    rL -- axial resistivity in Ohm*m
    """
    return np.sqrt(a * rm / (2 * rL))


def get_vcond(lambdam, taum):
    """Return conductance velocity in m/s
    lambda -- electronic length in m
    taum -- membrane time constant
    """
    return 2 * lambdam / taum


def get_tmax(data):
    """Return time of maximum of membrane voltage"""
    return data[np.argmax(data[:, 1])][0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cable")

    parser.add_argument(
        "--Vm", help="membrane leak potential in mV", type=float, default=-65
    )
    parser.add_argument("--length", help="cable length in μm", type=float, default=1000)
    parser.add_argument("--radius", help="cable radius in μm", type=float, default=1)
    parser.add_argument(
        "--cm", help="membrane capacitance in F/m^2", type=float, default=0.01
    )
    parser.add_argument(
        "--rL", help="axial resistivity in Ω·cm", type=float, default=90
    )
    parser.add_argument(
        "--g", help="membrane conductivity in S/cm^2", type=float, default=0.001
    )

    parser.add_argument(
        "--stimulus_start", help="start of stimulus in ms", type=float, default=10
    )
    parser.add_argument(
        "--stimulus_duration",
        help="duration of stimulus in ms",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--stimulus_amplitude",
        help="amplitude of stimulus in nA",
        type=float,
        default=1,
    )

    parser.add_argument(
        "--cv_policy_max_extent",
        help="maximum extent of control volume in μm",
        type=float,
        default=10,
    )

    # parse the command line arguments
    args = parser.parse_args()

    # set up membrane voltage probes equidistantly along the dendrites
    probes = [
        arbor.cable_probe_membrane_voltage(f"(location 0 {r})")
        for r in np.linspace(0, 1, 11)
    ]
    recipe = Cable(probes, **vars(args))

    # configure the simulation and handles for the probes
    sim = arbor.simulation(recipe)
    dt = 0.001
    handles = [
        sim.sample((0, i), arbor.regular_schedule(dt)) for i in range(len(probes))
    ]

    # run the simulation for 30 ms
    sim.run(tfinal=30, dt=dt)

    # retrieve the sampled membrane voltages and convert to a pandas DataFrame
    print("Plotting results ...")
    df_list = []
    for probe in range(len(handles)):
        samples, meta = sim.samples(handles[probe])[0]
        df_list.append(
            pandas.DataFrame(
                {"t/ms": samples[:, 0], "U/mV": samples[:, 1], "Probe": f"{probe}"}
            )
        )

    df = pandas.concat(df_list, ignore_index=True)
    seaborn.relplot(
        data=df, kind="line", x="t/ms", y="U/mV", hue="Probe", errorbar=None
    ).set(xlim=(9, 14)).savefig("single_cell_cable_result.svg")

    # calculcate the idealized conduction velocity, cf. cable equation
    data = [sim.samples(handle)[0][0] for handle in handles]
    rm = get_rm(args.g * 1 / (0.01 * 0.01))
    taum = get_taum(args.cm, rm)
    lambdam = get_lambdam(args.radius * 1e-6, rm, args.rL * 0.01)
    vcond_ideal = get_vcond(lambdam, taum)

    # take the last and second probe
    delta_t = get_tmax(data[-1]) - get_tmax(data[1])

    # 90% because we took the second probe
    delta_x = args.length * 0.9
    vcond = delta_x * 1e-6 / (delta_t * 1e-3)

    print(f"calculated conduction velocity: {vcond_ideal:.2f} m/s")
    print(f"simulated conduction velocity:  {vcond:.2f} m/s")

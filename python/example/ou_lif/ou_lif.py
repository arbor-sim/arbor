#!/bin/python3

# adapted from Jannik Luboeinski's example here: https://github.com/jlubo/arbor_ou_lif_example

import random
import subprocess
import arbor as A
from arbor import units as U
import numpy as np
import matplotlib.pyplot as plt


def make_catalogue():
    # build a new catalogue
    out = subprocess.getoutput("arbor-build-catalogue ou_lif . --cpu True")
    print(out)
    # load the new catalogue and extend it with builtin stochastic catalogue
    cat = A.load_catalogue("./ou_lif-catalogue.so")
    cat.extend(A.stochastic_catalogue(), "")
    return cat


def make_cell():
    # cell morphology
    # ===============

    tree = A.segment_tree()
    radius = 1e-10  # radius of cylinder (in µm)
    height = 2 * radius  # height of cylinder (in µm)
    tree.append(
        A.mnpos,
        A.mpoint(-height / 2, 0, 0, radius),
        A.mpoint(height / 2, 0, 0, radius),
        tag=1,
    )
    labels = A.label_dict({"center": "(location 0 0.5)"})

    # LIF density mechanism
    # =====================

    # surface area of the cylinder in m^2 (excluding the circle-shaped ends, since Arbor does
    # not consider current flux there)
    area_m2 = 2 * np.pi * (radius * 1e-6) * (height * 1e-6)
    area_cm2 = area_m2 * 1e4
    # conversion factor from nA to mA/cm^2; for point neurons
    i_factor = (1e-9 / 1e-3) / area_cm2
    # neuronal capacitance in F
    C_mem = 1e-9
    # specific capacitance in F/m^2, computed from absolute capacitance of point neuron
    c_mem = C_mem / area_m2
    # reversal potential in mV
    V_rev = -65.0
    # spiking threshold in mV
    V_th = -55.0 * U.mV
    # initial synaptic weight in nC
    h_0 = 4.20075
    # leak resistance in MOhm
    R_leak = 10.0
    # membrane time constant in ms
    tau_mem = 2.0

    lif = A.mechanism("lif")
    lif.set("R_leak", R_leak)
    lif.set("R_reset", 1e-10)
    # set to initial value to zero (background input is applied via stochastic ou_bg_mech)
    lif.set("I_0", 0)
    lif.set("i_factor", i_factor)
    lif.set("V_rev", V_rev)
    lif.set("V_reset", -70.0)
    lif.set("V_th", V_th.value_as(U.mV))
    # refractory time in ms
    lif.set("t_ref", tau_mem)

    # stochastic background input current (point mechanism)
    # =====================================================

    # synaptic time constant in ms
    tau_syn = 5.0
    # mean in nA
    mu_bg = 0.15
    # volatility in nA
    sigma_bg = 0.5
    # instantiate mechanism
    ou_bg = A.mechanism("ou_input")
    ou_bg.set("mu", mu_bg)
    ou_bg.set("sigma", sigma_bg)
    ou_bg.set("tau", tau_syn)

    # stochastic stimulus input current (point mechanism)
    # ===================================================

    # number of neurons in the putative population
    N = 25
    # firing rate of the putative neurons in Hz
    f = 100
    # synaptic weight in nC
    w_out = h_0 / R_leak
    # mean in nA
    mu_stim = N * f * w_out
    # volatility in nA
    sigma_stim = np.sqrt((1000.0 * N * f) / (2 * tau_syn)) * w_out
    # instantiate mechanism
    ou_stim = A.mechanism("ou_input")
    ou_stim.set("mu", mu_stim)
    ou_stim.set("sigma", sigma_stim)
    ou_stim.set("tau", tau_syn)

    # paint and place mechanisms
    # ==========================

    decor = A.decor()
    decor.set_property(Vm=V_rev * U.mV, cm=c_mem * U.Ohm * U.cm)
    decor.paint("(all)", A.density(lif))
    decor.place('"center"', A.synapse(ou_stim), "ou_stim")
    decor.place('"center"', A.synapse(ou_bg), "ou_bg")
    decor.place('"center"', A.threshold_detector(V_th), "spike_detector")

    return A.cable_cell(tree, decor, labels)


class ou_recipe(A.recipe):
    def __init__(self, cell, cat):
        A.recipe.__init__(self)

        # simulation runtime parameters in ms
        self.runtime = 20 * U.s
        self.dt = 0.2 * U.ms

        # initialize catalogue and cell properties
        self.the_props = A.neuron_cable_properties()
        self.the_props.catalogue = cat
        self.the_cell = cell

        # protocol for stimulation
        # "time_start": starting time, "scheme": protocol type
        self.bg_prot = {"time_start": 0, "scheme": "FULL", "label": "ou_bg"}
        self.stim1_prot = {"time_start": 10000, "scheme": "TRIPLET", "label": "ou_stim"}
        self.stim2_prot = {
            "time_start": 15000,
            "scheme": "ONEPULSE",
            "label": "ou_stim",
        }

    def cell_kind(self, gid):
        return A.cell_kind.cable

    def cell_description(self, gid):
        return self.the_cell

    def global_properties(self, kind):
        return self.the_props

    def num_cells(self):
        return 1

    def probes(self, gid):
        # probe membrane potential, total current, and external input currents
        return [
            A.cable_probe_membrane_voltage('"center"', "Um"),
            A.cable_probe_total_ion_current_cell("Itot"),
            A.cable_probe_point_state_cell("ou_input", "I_ou", "Iou"),
        ]

    def event_generators(self, gid):
        gens = []
        # OU stimulus input #1
        gens.extend(self.get_generators(self.stim1_prot))
        # OU stimulus input #2
        gens.extend(self.get_generators(self.stim2_prot))
        # OU background input
        gens.extend(self.get_generators(self.bg_prot))
        return gens

    # Returns A.event_generator instances that describe the specifics of a given
    # input/stimulation protocol for a mechanism implementing an Ornstein-Uhlenbeck process.
    # Here, if the value of the 'weight' parameter is 1, stimulation is switched on,
    # whereas if it is -1, stimulation is switched off.
    def get_generators(self, protocol):
        prot_name = protocol["scheme"]  # name of the protocol (defining its structure)
        start_time = (
            protocol["time_start"] * U.ms
        )  # time at which the stimulus starts in ms
        label = protocol["label"]  # target synapse (mechanism label)

        if prot_name == "ONEPULSE":
            # create regular schedules to implement a stimulation pulse that lasts for 0.1 s
            stim_on = A.event_generator(
                label,
                1,
                A.regular_schedule(start_time, self.dt, start_time + self.dt),
            )
            stim_off = A.event_generator(
                label,
                -1,
                A.regular_schedule(
                    start_time + 0.1 * U.s, self.dt, start_time + 0.1 * U.s + self.dt
                ),
            )
            return [stim_on, stim_off]

        elif prot_name == "TRIPLET":
            # create regular schedules to implement pulses that last for 0.1 s each
            stim1_on = A.event_generator(
                label,
                1,
                A.regular_schedule(start_time, self.dt, start_time + self.dt),
            )
            stim1_off = A.event_generator(
                label,
                -1,
                A.regular_schedule(
                    start_time + 0.1 * U.s, self.dt, start_time + 0.1 * U.s + self.dt
                ),
            )
            stim2_on = A.event_generator(
                label,
                1,
                A.regular_schedule(
                    start_time + 0.5 * U.s, self.dt, start_time + 0.5 * U.s + self.dt
                ),
            )
            stim2_off = A.event_generator(
                label,
                -1,
                A.regular_schedule(
                    start_time + 0.6 * U.s, self.dt, start_time + 0.6 * U.s + self.dt
                ),
            )
            stim3_on = A.event_generator(
                label,
                1,
                A.regular_schedule(
                    start_time + 1 * U.s, self.dt, start_time + 1 * U.s + self.dt
                ),
            )
            stim3_off = A.event_generator(
                label,
                -1,
                A.regular_schedule(
                    start_time + 1.1 * U.s, self.dt, start_time + 1.1 * U.s + self.dt
                ),
            )
            return [stim1_on, stim1_off, stim2_on, stim2_off, stim3_on, stim3_off]

        elif prot_name == "FULL":
            # create a regular schedule that lasts for the full runtime
            stim_on = A.event_generator(
                label,
                1,
                A.regular_schedule(start_time, self.dt, start_time + self.dt),
            )
            return [stim_on]

        else:
            return []


if __name__ == "__main__":
    # set up and run simulation
    # =========================

    # create recipe
    cell = make_cell()
    cat = make_catalogue()
    recipe = ou_recipe(cell, cat)

    # get random seed
    random_seed = random.getrandbits(64)
    print("random_seed = " + str(random_seed))

    # select one thread and no GPU
    alloc = A.proc_allocation(threads=1, gpu_id=None)
    context = A.context(alloc, mpi=None)
    domains = A.partition_load_balance(recipe, context)

    # create simulation
    sim = A.simulation(recipe, context, domains, seed=random_seed)

    # create schedule for recording
    reg_sched = A.regular_schedule(0 * U.ms, recipe.dt, recipe.runtime)

    # set handles to probe membrane potential and currents
    gid = 0
    handle_mem = sim.sample((gid, "Um"), reg_sched)  # membrane potential
    handle_tot_curr = sim.sample((gid, "Itot"), reg_sched)  # total current
    handle_curr = sim.sample((gid, "Iou"), reg_sched)  # input current

    sim.record(A.spike_recording.all)
    sim.run(tfinal=recipe.runtime, dt=recipe.dt)

    # get traces and spikes from simulator
    # ====================================

    # there is only one location (single control volume)
    loc = 0
    data_mem, data_curr = [], []

    data, _ = sim.samples(handle_mem)[loc]
    times = data[:, 0]

    # read out neuron data
    data_mem.append(data[:, 1])

    # get total membrane current
    data_tot_curr, _ = sim.samples(handle_tot_curr)[loc]
    data_curr.append(np.negative(data_tot_curr[:, 1]))

    # get Ornstein-Uhlenbeck currents
    data_input_curr, _ = sim.samples(handle_curr)[loc]
    data_curr.append(data_input_curr[:, 1])
    data_curr.append(data_input_curr[:, 2])

    # get spikes
    spike_times = sim.spikes()["time"]

    # assemble, store, and plot data
    # ==============================

    data_header = "Time, V, I_stim, I_bg"
    data_stacked = np.column_stack(
        [times, np.transpose(data_mem), np.transpose(data_curr)]
    )

    np.savetxt("traces.txt", data_stacked, fmt="%.4f", header=data_header)
    np.savetxt("spikes.txt", spike_times, fmt="%.4f")

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, figsize=(10, 10))
    axes[0].set_title("random_seed = " + str(random_seed))
    axes[0].plot(data_stacked[:, 0], data_stacked[:, 1], label="V", color="C3")
    axes[0].plot(
        spike_times, -55 * np.ones(len(spike_times)), ".", color="blue", markersize=1
    )
    axes[1].plot(data_stacked[:, 0], data_stacked[:, 2], label="I_tot", color="C1")
    axes[2].plot(data_stacked[:, 0], data_stacked[:, 3], label="I_stim", color="C2")
    axes[3].plot(data_stacked[:, 0], data_stacked[:, 4], label="I_bg", color="C0")
    axes[3].set_xlabel("Time (ms)")
    axes[0].set_ylabel("V (mV)")
    axes[1].set_ylabel("I_tot (nA)")
    axes[2].set_ylabel("I_stim (nA)")
    axes[3].set_ylabel("I_bg (nA)")
    fig.tight_layout()
    fig.savefig("traces.svg", bbox_inches="tight")

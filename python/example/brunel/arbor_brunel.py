# the code is adapted based on a previous version of Dr. Sebastian Schmidt
# adaptation was made to match brunel network simulation in NEST simulator

# load packages needed for this project
import arbor as A
from arbor import units as U
import numpy as np
from numpy.random import RandomState
from parameters import *


class brunel_recipe(A.recipe):
    # define recipe for brunel network
    def __init__(
        self,
        NE,
        NI,
        epsilon,
        weight,
        delay,
        g,
        rate,
        seed,
        tau_m,
        V_th,
        V_m,
        E_L,
        V_reset,
        C_m,
        t_ref,
    ):
        A.recipe.__init__(self)

        self.seed_ = seed

        # set up neruon parameters
        self.tau_m = neuron_params["tau_m"] * U.ms
        self.V_th = neuron_params["V_th"] * U.mV
        self.V_m = neuron_params["V_m"] * U.mV
        self.E_L = neuron_params["E_L"] * U.mV
        self.E_R = neuron_params["V_reset"] * U.mV
        self.C_m = neuron_params["C_m"] * U.pF
        self.t_ref = neuron_params["t_ref"] * U.ms

        # set up network parameters
        self.ncells_exc_ = NE
        self.ncells_inh_ = NI
        self.in_degree_exc_ = round(
            epsilon * NE
        )  # fixed indegree based on connection probability
        self.in_degree_inh_ = round(
            epsilon * NI
        )  # fixed indegree based on connection probability
        self.delay_ = delay * U.ms

        # set and scale synaptic weights based on C_m due to the conductance based property of LIF neuron in Arbor
        self.weight_exc_ = weight * C_m
        self.weight_inh_ = -1 * g * weight * C_m
        self.weight_ext_ = weight * C_m

        # set the firing rate of the Poissonian input
        self.lambda_ = rate * U.Hz

    def num_cells(self):
        return self.ncells_exc_ + self.ncells_inh_

    def cell_kind(self, gid):
        return A.cell_kind.lif

    def cell_description(self, gid):
        return A.lif_cell(
            "src",
            "tgt",
            tau_m=self.tau_m,
            V_th=self.V_th,
            C_m=self.C_m,
            E_L=self.E_L,
            E_R=self.E_R,
            V_m=self.V_m,
            t_ref=self.t_ref,
        )

    def sample_subset(self, gen, gid, start, end, m):
        # this function is used to generate random connection based on fixed indegree
        idx = np.arange(start, end)
        if start <= gid < end:
            idx = np.delete(idx, gid - start)
        gen.shuffle(idx)
        return idx[:m]

    def connections_on(self, gid):
        gen = RandomState(gid + self.seed_)
        # Add incoming excitatory connections.
        connections = [
            A.connection((i, "src"), "tgt", self.weight_exc_, self.delay_)
            for i in self.sample_subset(
                gen, gid, 0, self.ncells_exc_, self.in_degree_exc_
            )
        ]
        # Add incoming inhibitory connections.
        connections += [
            A.connection((i, "src"), "tgt", self.weight_inh_, self.delay_)
            for i in self.sample_subset(
                gen,
                gid,
                self.ncells_exc_,
                self.ncells_exc_ + self.ncells_inh_,
                self.in_degree_inh_,
            )
        ]

        return connections

    def event_generators(self, gid):
        # add poissonian input to each neuron
        return [
            A.event_generator(
                "tgt",
                self.weight_ext_,
                A.poisson_schedule(
                    freq=self.lambda_,
                    seed=gid + (self.ncells_exc_ + self.ncells_inh_) * self.seed_,
                ),
            )
        ]


if __name__ == "__main__":
    recipe = brunel_recipe(
        NE,
        NI,
        epsilon,
        weight,
        delay,
        g,
        rate,
        seed,
        tau_m,
        V_th,
        V_m,
        E_L,
        V_reset,
        C_m,
        t_ref,
    )

    context = A.context()
    if A.config()["profiling"]:
        A.profiler_initialize(context)
    print(context)
    meters = A.meter_manager()
    meters.start(context)

    hint = A.partition_hint()
    hint.cpu_group_size = 5000
    hints = {A.cell_kind.lif: hint}
    decomp = A.partition_load_balance(recipe, context, hints)
    print(decomp)

    sim = A.simulation(recipe, context, decomp)
    sim.record(A.spike_recording.all)
    sim.run(tfinal * U.ms, dt * U.ms)

    # get data
    spikes = sim.spikes()
    sources = spikes["source"]["gid"]
    times = spikes["time"]

    # save data
    np.save("times.npy", times)
    np.save("sources.npy", sources)

#!/usr/bin/env python3

import arbor
import argparse
import numpy as np
from numpy.random import RandomState

"""
A Brunel network consists of nexc excitatory LIF neurons and ninh inhibitory
LIF neurons. Each neuron in the network receives in_degree_prop * nexc
excitatory connections chosen randomly, in_degree_prop * ninh inhibitory
connections and next (external) Poisson connections. All the connections have
the same delay. The strenght of excitatory and Poisson connections is given by
parameter weight, whereas the strength of inhibitory connections is
rel_inh_strength * weight. Poisson neurons all spike independently with expected
number of spikes given by parameter poiss_lambda. Because of the refractory
period, the activity is mostly driven by Poisson neurons and recurrent
connections have a small effect.

Call with parameters, for example:
./brunel.py -n 400 -m 100 -e 20 -p 0.1 -w 1.2 -d 1 -g 0.5 -l 5 -t 100 -s 1 -G 50 -S 123 -f spikes.txt
"""


# Samples m unique values in interval [start, end) - gid.
# We exclude gid because we don't want self-loops.
def sample_subset(gen, gid, start, end, m):
    idx = np.arange(start, end)
    if start <= gid < end:
        idx = np.delete(idx, gid - start)
    gen.shuffle(idx)
    return idx[:m]


class brunel_recipe(arbor.recipe):
    def __init__(
        self,
        nexc,
        ninh,
        next,
        in_degree_prop,
        weight,
        delay,
        rel_inh_strength,
        poiss_lambda,
        seed=42,
    ):

        arbor.recipe.__init__(self)

        # Make sure that in_degree_prop in the interval (0, 1]
        if not 0.0 < in_degree_prop <= 1.0:
            print(
                "The proportion of incoming connections should be in the interval (0, 1]."
            )
            quit()

        self.ncells_exc_ = nexc
        self.ncells_inh_ = ninh
        self.delay_ = delay
        self.seed_ = seed

        # Set up the parameters.
        self.weight_exc_ = weight
        self.weight_inh_ = -rel_inh_strength * weight
        self.weight_ext_ = weight
        self.in_degree_exc_ = round(in_degree_prop * nexc)
        self.in_degree_inh_ = round(in_degree_prop * ninh)
        # each cell receives next incoming Poisson sources with mean rate poiss_lambda, which is equivalent
        # to a single Poisson source with mean rate next*poiss_lambda
        self.lambda_ = next * poiss_lambda

    def num_cells(self):
        return self.ncells_exc_ + self.ncells_inh_

    def cell_kind(self, gid):
        return arbor.cell_kind.lif

    def connections_on(self, gid):
        gen = RandomState(gid + self.seed_)
        connections = []
        # Add incoming excitatory connections.
        connections = [
            arbor.connection((i, "src"), "tgt", self.weight_exc_, self.delay_)
            for i in sample_subset(gen, gid, 0, self.ncells_exc_, self.in_degree_exc_)
        ]
        # Add incoming inhibitory connections.
        connections += [
            arbor.connection((i, "src"), "tgt", self.weight_inh_, self.delay_)
            for i in sample_subset(
                gen,
                gid,
                self.ncells_exc_,
                self.ncells_exc_ + self.ncells_inh_,
                self.in_degree_inh_,
            )
        ]

        return connections

    def cell_description(self, gid):
        cell = arbor.lif_cell("src", "tgt")
        cell.tau_m = 10
        cell.V_th = 10
        cell.C_m = 20
        cell.E_L = 0
        cell.V_m = 0
        cell.t_ref = 2
        return cell

    def event_generators(self, gid):
        t0 = 0
        sched = arbor.poisson_schedule(t0, self.lambda_, gid + self.seed_)
        return [arbor.event_generator("tgt", self.weight_ext_, sched)]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Brunel model miniapp.")
    parser.add_argument(
        "-n",
        "--n-excitatory",
        dest="nexc",
        type=int,
        default=400,
        help="Number of cells in the excitatory population",
    )
    parser.add_argument(
        "-m",
        "--n-inhibitory",
        dest="ninh",
        type=int,
        default=100,
        help="Number of cells in the inhibitory population",
    )
    parser.add_argument(
        "-e",
        "--n-external",
        dest="next",
        type=int,
        default=40,
        help="Number of incoming Poisson (external) connections per cell",
    )
    parser.add_argument(
        "-p",
        "--in-degree-prop",
        dest="syn_per_cell_prop",
        type=float,
        default=0.05,
        help="Proportion of the connections received per cell",
    )
    parser.add_argument(
        "-w",
        "--weight",
        dest="weight",
        type=float,
        default=1.2,
        help="Weight of excitatory connections",
    )
    parser.add_argument(
        "-d",
        "--delay",
        dest="delay",
        type=float,
        default=0.1,
        help="Delay of all connections",
    )
    parser.add_argument(
        "-g",
        "--rel-inh-w",
        dest="rel_inh_strength",
        type=float,
        default=1,
        help="Relative strength of inhibitory synapses with respect to the excitatory ones",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        dest="poiss_lambda",
        type=float,
        default=1,
        help="Mean firing rate from a single poisson cell (kHz)",
    )
    parser.add_argument(
        "-t",
        "--tfinal",
        dest="tfinal",
        type=float,
        default=100,
        help="Length of the simulation period (ms)",
    )
    parser.add_argument(
        "-s", "--dt", dest="dt", type=float, default=1, help="Simulation time step (ms)"
    )
    parser.add_argument(
        "-G",
        "--group-size",
        dest="group_size",
        type=int,
        default=10,
        help="Number of cells per cell group",
    )
    parser.add_argument(
        "-S",
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Seed for poisson spike generators",
    )
    parser.add_argument(
        "-f",
        "--write-spikes",
        dest="spike_file_output",
        type=str,
        help="Save spikes to file",
    )
    # parser.add_argument('-z', '--profile-rank-zero', dest='profile_only_zero', action='store_true', help='Only output profile information for rank 0')
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print more verbose information to stdout",
    )

    opt = parser.parse_args()
    if opt.verbose:
        print("Running brunel.py with the following settings:")
        for k, v in vars(opt).items():
            print(f"{k} = {v}")

    context = arbor.context("avail_threads")
    print(context)

    meters = arbor.meter_manager()
    meters.start(context)

    recipe = brunel_recipe(
        opt.nexc,
        opt.ninh,
        opt.next,
        opt.syn_per_cell_prop,
        opt.weight,
        opt.delay,
        opt.rel_inh_strength,
        opt.poiss_lambda,
        opt.seed,
    )

    meters.checkpoint("recipe-create", context)

    hint = arbor.partition_hint()
    hint.cpu_group_size = opt.group_size
    hints = {arbor.cell_kind.lif: hint}
    decomp = arbor.partition_load_balance(recipe, context, hints)
    print(decomp)

    meters.checkpoint("load-balance", context)

    sim = arbor.simulation(recipe, context, decomp)
    sim.record(arbor.spike_recording.all)

    meters.checkpoint("simulation-init", context)

    sim.run(opt.tfinal, opt.dt)

    meters.checkpoint("simulation-run", context)

    # Print profiling information
    print(f"{arbor.meter_report(meters, context)}")

    # Print spike times
    print(f"{len(sim.spikes())} spikes generated.")

    if opt.spike_file_output:
        with open(opt.spike_file_output, "w") as the_file:
            for meta, data in sim.spikes():
                the_file.write(f"{meta[0]} {data:3.3f}\n")

#!/usr/bin/env python3

import arbor
import pandas, seaborn
import numpy
from math import sqrt

'''
A Brunel network consists of nexc excitatory LIF neurons and ninh inhibitory LIF neurons.
Each neuron in the network receives in_degree_prop * nexc excitatory connections
chosen randomly, in_degree_prop * ninh inhibitory connections and next (external) Poisson connections.
All the connections have the same delay. The strenght of excitatory and Poisson connections is given by
parameter weight, whereas the strength of inhibitory connections is rel_inh_strength * weight.
Poisson neurons all spike independently with expected number of spikes given by parameter poiss_lambda.
Because of the refractory period, the activity is mostly driven by Poisson neurons and
recurrent connections have a small effect.
'''

class brunel_recipe (arbor.recipe):

    def __init__(nexc, ninh, next, in_degree_prop,
                  weight, delay, rel_inh_strength, poiss_lambda, seed = 42):
        self.ncells_exc_(nexc)
        brunel_recipenexc, ninh, next, in_degree_prop,
                    weight, delay, rel_inh_strength, poiss_lambda, seed = 42):
        ncells_exc_(nexc), ncells_inh_(ninh), delay_(delay), seed_(seed) {
        // Make sure that in_degree_prop in the interval (0, 1]
        if (in_degree_prop <= 0.0 || in_degree_prop > 1.0) {
            throw std::out_of_range("The proportion of incoming connections should be in the interval (0, 1].");
        }

        // Set up the parameters.
        weight_exc_ = weight;
        weight_inh_ = -rel_inh_strength * weight_exc_;
        weight_ext_ =  weight;
        in_degree_exc_ = std::round(in_degree_prop * nexc);
        in_degree_inh_ = std::round(in_degree_prop * ninh);
        // each cell receives next incoming Poisson sources with mean rate poiss_lambda, which is equivalent
        // to a single Poisson source with mean rate next*poiss_lambda
        lambda_ = next * poiss_lambda;
    }

    num_cells() const override {
        return ncells_exc_ + ncells_inh_;
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::lif;
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> connections;
        // Add incoming excitatory connections.
        for (auto i: sample_subset(gid, 0, ncells_exc_, in_degree_exc_)) {
            cell_member_type source{cell_gid_type(i), 0};
            cell_member_type target{gid, 0};
            cell_connection conn(source, target, weight_exc_, delay_);
            connections.push_back(conn);
        }

        // Add incoming inhibitory connections.
        for (auto i: sample_subset(gid, ncells_exc_, ncells_exc_ + ncells_inh_, in_degree_inh_)) {
            cell_member_type source{cell_gid_type(i), 0};
            cell_member_type target{gid, 0};
            cell_connection conn(source, target, weight_inh_, delay_);
            connections.push_back(conn);
        }
        return connections;
    }

    util::unique_any get_cell_description(cell_gid_type gid) const override {
        auto cell = lif_cell();
        cell.tau_m = 10;
        cell.V_th = 10;
        cell.C_m = 20;
        cell.E_L = 0;
        cell.V_m = 0;
        cell.V_reset = 0;
        cell.t_ref = 2;
        return cell;
    }

    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        std::vector<arb::event_generator> gens;

        std::mt19937_64 G;
        G.seed(gid + seed_);

        time_type t0 = 0;
        cell_member_type target{gid, 0};

        gens.emplace_back(poisson_generator(target, weight_ext_, t0, lambda_, G));
        return gens;
    }

    num_sources(cell_gid_type) const override {
         return 1;
    }

    num_targets(cell_gid_type) const override {
        return 1;
    }

private:
    // Number of excitatory cells.
    ncells_exc_;

    // Number of inhibitory cells.
    ncells_inh_;

    // Weight of excitatory synapses.
    weight_exc_;

    // Weight of inhibitory synapses.
    weight_inh_;

    // Weight of external Poisson cell synapses.
    weight_ext_;

    // Delay of all synapses.
    delay_;

    // Number of connections that each neuron receives from excitatory population.
    in_degree_exc_;

    // Number of connections that each neuron receives from inhibitory population.
    in_degree_inh_;

    // Expected number of poisson spikes.
    lambda_;

    // Seed used for the Poisson spikes generation.
    seed_;
};













class ring_recipe (arbor.recipe):

    def __init__(self, n=10):
        # The base C++ class constructor must be called first, to ensure that
        # all memory in the C++ class is initialized correctly.
        arbor.recipe.__init__(self)
        self.ncells = n
        self.props = arbor.neuron_cable_properties()
        self.cat = arbor.default_catalogue()
        self.props.register(self.cat)

    # The num_cells method that returns the total number of cells in the model
    # must be implemented.
    def num_cells(self):
        return self.ncells

    # The cell_description method returns a cell
    def cell_description(self, gid):
        return make_cable_cell(gid)

    def num_targets(self, gid):
        return 1

    def num_sources(self, gid):
        return 1

    # The kind method returns the type of cell with gid.
    # Note: this must agree with the type returned by cell_description.
    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    # Make a ring network
    def connections_on(self, gid):
        src = (gid-1)%self.ncells
        w = 0.01
        d = 5
        return [arbor.connection(arbor.cell_member(src,0), arbor.cell_member(gid,0), w, d)]

    # Attach a generator to the first cell in the ring.
    def event_generators(self, gid):
        if gid==0:
            sched = arbor.explicit_schedule([1])
            return [arbor.event_generator(arbor.cell_member(0,0), 0.1, sched)]
        return []

    def probes(self, gid):
        return [arbor.cable_probe_membrane_voltage('(location 0 0)')]

    def global_properties(self, kind):
        return self.props

context = arbor.context(threads=12, gpu_id=None)
print(context)

meters = arbor.meter_manager()
meters.start(context)

ncells = 4
recipe = ring_recipe(ncells)
print(f'{recipe}')

meters.checkpoint('recipe-create', context)

h= arbor.partition_hint()
hint.prefer_gpu = True
hint.gpu_group_size = 1000
print(f'{hint}')

hints = dict([(arbor.cell_kind.cable, hint)])
decomp = arbor.partition_load_balance(recipe, context, hints)
print(f'{decomp}')

meters.checkpoint('load-balance', context)

sim = arbor.simulation(recipe, decomp, context)
sim.record(arbor.spike_recording.all)

meters.checkpoint('simulation-init', context)

# Attach a sampler to the voltage probe on cell 0.
# Sample rate of 10 sample every ms.
handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

tfinal=100
sim.run(tfinal)
print(f'{sim} finished')

meters.checkpoint('simulation-run', context)

# Prprofiling information
print(f'{arbor.meter_report(meters, context)}')

# Prspike times
print('spikes:')
for sp in sim.spikes():
    print(' ', sp)

# Plot the recorded voltages over time.
print("Plotting results ...")
df_list = []
for gid in range(ncells):
    samples, meta = sim.samples(handles[gid])[0]
    df_list.append(pandas.DataFrame({'t/ms': samples[:, 0], 'U/mV': samples[:, 1], 'Cell': f"cell {gid}"}))

df = pandas.concat(df_list)
seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Cell",ci=None).savefig('network_ring_result.svg')

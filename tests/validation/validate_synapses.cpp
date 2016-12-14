#include <fvm_multicell.hpp>

#include "../gtest.h"
#include "validate_synapses.hpp"

using lowered_cell = nest::mc::fvm::fvm_multicell<nest::mc::multicore::backend>;

TEST(simple_synapse, expsyn_neuron_ref) {
    SCOPED_TRACE("expsyn");
    run_synapse_test<lowered_cell>("expsyn", "neuron_simple_exp_synapse.json");
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    SCOPED_TRACE("exp2syn");
    run_synapse_test<lowered_cell>("exp2syn", "neuron_simple_exp2_synapse.json");
}

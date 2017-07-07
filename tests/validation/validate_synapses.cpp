#include <fvm_multicell.hpp>

#include "../gtest.h"
#include "validate_synapses.hpp"

const auto backend = nest::mc::backend_policy::multicore;

TEST(simple_synapse, expsyn_neuron_ref) {
    SCOPED_TRACE("expsyn");
    run_synapse_test("expsyn", "neuron_simple_exp_synapse.json", backend);
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    SCOPED_TRACE("exp2syn");
    run_synapse_test("exp2syn", "neuron_simple_exp2_synapse.json", backend);
}

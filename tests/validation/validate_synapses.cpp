#include <fvm_multicell.hpp>
#include <hardware/gpu.hpp>

#include "../gtest.h"
#include "validate_synapses.hpp"

using namespace nest::mc;

TEST(simple_synapse, expsyn_neuron_ref) {
    SCOPED_TRACE("expsyn-multicore");
    run_synapse_test("expsyn", "neuron_simple_exp_synapse.json", backend_kind::multicore);
    if (hw::num_gpus()) {
        SCOPED_TRACE("expsyn-gpu");
        run_synapse_test("expsyn", "neuron_simple_exp_synapse.json", backend_kind::gpu);
    }
}

TEST(simple_synapse, exp2syn_neuron_ref) {
    SCOPED_TRACE("exp2syn-multicore");
    run_synapse_test("exp2syn", "neuron_simple_exp2_synapse.json", backend_kind::multicore);
    if (hw::num_gpus()) {
        SCOPED_TRACE("exp2syn-gpu");
        run_synapse_test("exp2syn", "neuron_simple_exp2_synapse.json", backend_kind::gpu);
    }
}

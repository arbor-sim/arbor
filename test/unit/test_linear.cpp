#include <vector>

#include <arbor/mechanism.hpp>
#include <arbor/version.hpp>

#include "backends/multicore/fvm.hpp"

#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif

#include "common.hpp"
#include "mech_private_field_access.hpp"
#include "fvm_lowered_cell.hpp"
#include "fvm_lowered_cell_impl.hpp"
#include "sampler_map.hpp"
#include "simple_recipes.hpp"
#include "unit_test_catalogue.hpp"

using namespace arb;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

using shared_state = backend::shared_state;
ACCESS_BIND(std::unique_ptr<shared_state> fvm_cell::*, private_state_ptr, &fvm_cell::state_)

template <typename backend>
void run_kinetic_test(std::string mech_name,
        std::vector<std::string> variables,
        std::vector<fvm_value_type> values) {

    auto cat = make_unit_test_catalogue();

    fvm_size_type ncell = 1;
    fvm_size_type ncv = 1;
    std::vector<fvm_index_type> cv_to_intdom(ncv, 0);

    std::vector<fvm_gap_junction> gj = {};
    auto instance = cat.instance<backend>(mech_name);
    auto& kinetic_test = instance.mech;

    std::vector<fvm_value_type> temp(ncv, 300.);
    std::vector<fvm_value_type> vinit(ncv, -65);

    auto shared_state = std::make_unique<typename backend::shared_state>(
            ncell, cv_to_intdom, gj, vinit, temp, kinetic_test->data_alignment());

    mechanism_layout layout;
    mechanism_overrides overrides;

    layout.weight.assign(ncv, 1.);
    for (fvm_size_type i = 0; i<ncv; ++i) {
        layout.cv.push_back(i);
    }

    kinetic_test->instantiate(0, *shared_state, overrides, layout);

    kinetic_test->set_parameter("a0", std::vector<fvm_value_type>(ncv,2.5));
    kinetic_test->set_parameter("a1", std::vector<fvm_value_type>(ncv,0.5));
    kinetic_test->set_parameter("a2", std::vector<fvm_value_type>(ncv,3));
    kinetic_test->set_parameter("a3", std::vector<fvm_value_type>(ncv,2.3));

    shared_state->reset();

    kinetic_test->initialize();

    shared_state->update_time_to(0.5, 0.5);
    shared_state->set_dt();

    kinetic_test->nrn_state();

    for (unsigned i = 0; i < variables.size(); i++) {
        for (unsigned j = 0; j < ncv; j++) {
            EXPECT_NEAR(values[i], mechanism_field(kinetic_test.get(), variables[i]).at(j), 1e-6);
        }
    }
}

TEST(mech_linear_cpu, lienar) {
    std::vector<std::string> variables = {"h", "s", "d"};
    std::vector<fvm_value_type> values = {0.5, 0.2, 0.3};

    run_kinetic_test<multicore::backend>("test_linear", variables, values);
}

#ifdef ARB_GPU_ENABLED
TEST(mech_kinetic_gpu, kintetic) {
    run_kinetic_test<gpu::backend>("test_kin_diff");
    run_kinetic_test<gpu::backend>("test_kin_conserve");
}
#endif

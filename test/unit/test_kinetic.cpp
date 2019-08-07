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
void run_kinetic_test(std::string mech_name) {

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
    shared_state->reset();

    kinetic_test->initialize();
    std::vector<fvm_value_type> expected_init_s_values(ncv, 0.5);
    std::vector<fvm_value_type> expected_init_h_values(ncv, 0.2);
    std::vector<fvm_value_type> expected_init_d_values(ncv, 0.3);

    std::vector<fvm_value_type> expected_new_s_values(ncv, 0.380338);
    std::vector<fvm_value_type> expected_new_h_values(ncv, 0.446414);
    std::vector<fvm_value_type> expected_new_d_values(ncv, 0.173247);


    for (unsigned i = 0; i < ncv; i++) {
        EXPECT_NEAR(expected_init_s_values.at(i), mechanism_field(kinetic_test.get(), "s").at(i), 1e-6);
        EXPECT_NEAR(expected_init_h_values.at(i), mechanism_field(kinetic_test.get(), "h").at(i), 1e-6);
        EXPECT_NEAR(expected_init_d_values.at(i), mechanism_field(kinetic_test.get(), "d").at(i), 1e-6);
    }

    shared_state->update_time_to(0.5, 0.5);
    shared_state->set_dt();

    kinetic_test->nrn_state();

    for (unsigned i = 0; i < ncv; i++) {
        EXPECT_NEAR(expected_new_s_values.at(i), mechanism_field(kinetic_test.get(), "s").at(i), 1e-6);
        EXPECT_NEAR(expected_new_h_values.at(i), mechanism_field(kinetic_test.get(), "h").at(i), 1e-6);
        EXPECT_NEAR(expected_new_d_values.at(i), mechanism_field(kinetic_test.get(), "d").at(i), 1e-6);
    }
}

TEST(mech_kinetic, kintetic) {
//    run_kinetic_test<multicore::backend>("test_kin_diff");
    run_kinetic_test<multicore::backend>("test_kin_conserve");
}

#ifdef ARB_GPU_ENABLED
TEST(mech_kinetic_gpu, kintetic) {
    run_kinetic_test<gpu::backend>("test_kin_diff");
    run_kinetic_test<gpu::backend>("test_kin_conserve");
}
#endif

#include <vector>

#include <arbor/mechanism.hpp>
#include <arbor/version.hpp>

#include "backends/multicore/fvm.hpp"
#include "backends/common_types.hpp"
#ifdef ARB_GPU_ENABLED
#include "backends/gpu/fvm.hpp"
#endif

#include "common.hpp"
#include "mech_private_field_access.hpp"
#include "unit_test_catalogue.hpp"

using namespace arb;

template <typename backend>
void run_celsius_test() {
    auto thread_pool = std::make_shared<arb::threading::task_system>();

    auto cat = make_unit_test_catalogue();

    // one cell, three CVs:

    arb_size_type ncell = 1;
    arb_size_type ncv = 3;
    std::vector<arb_index_type> cv_to_cell(ncv, 0);

    auto instance = cat.instance(backend::kind, "celsius_test");
    auto& celsius_test = instance.mech;

    double temperature_K = 300.;
    double temperature_C = temperature_K-273.15;

    std::vector<arb_value_type> temp(ncv, temperature_K);
    std::vector<arb_value_type> diam(ncv, 1.);
    std::vector<arb_value_type> area(ncv, 10.);
    std::vector<arb_value_type> vinit(ncv, -65);
    std::vector<arb_index_type> src_to_spike = {};

    auto shared_state = std::make_unique<typename backend::shared_state>(thread_pool, ncell, ncv, cv_to_cell,
                                                                         vinit, temp, diam, area,
                                                                         src_to_spike,
                                                                         fvm_detector_info{},
                                                                         celsius_test->data_alignment());

    mechanism_layout layout;
    mechanism_overrides overrides;

    layout.weight.assign(ncv, 1.);
    for (arb_size_type i = 0; i<ncv; ++i) {
        layout.cv.push_back(i);
    }

    shared_state->instantiate(*celsius_test, 0, overrides, layout, {});
    shared_state->reset();

    // expect 0 value in state 'c' after init:

    celsius_test->initialize();
    std::vector<arb_value_type> expected_c_values(ncv, 0.);

    EXPECT_EQ(expected_c_values, mechanism_field(celsius_test.get(), "c"));

    // expect temperature_C value in state 'c' after state update:

    celsius_test->update_state();
    expected_c_values.assign(ncv, temperature_C);

    EXPECT_EQ(expected_c_values, mechanism_field(celsius_test.get(), "c"));
}

template <typename backend>
void run_diam_test() {
    auto thread_pool = std::make_shared<arb::threading::task_system>();

    auto cat = make_unit_test_catalogue();

    // one cell, three CVs:

    arb_size_type ncell = 1;
    arb_size_type ncv = 3;
    std::vector<arb_index_type> cv_to_cell(ncv, 0);

    auto instance = cat.instance(backend::kind, "diam_test");
    auto mech = instance.mech.get();

    std::vector<arb_value_type> temp(ncv, 300.);
    std::vector<arb_value_type> vinit(ncv, -65);
    std::vector<arb_value_type> diam(ncv);
    std::vector<arb_value_type> area(ncv);
    std::vector<arb_index_type> src_to_spike;

    mechanism_layout layout;
    layout.weight.assign(ncv, 1.);

    for (arb_size_type i = 0; i < ncv; ++i) {
        diam[i] =   i*2.0 + 0.1;
        area[i] = i*i*4.0 + 0.2;
        layout.cv.push_back(i);
    }

    auto shared_state = std::make_unique<typename backend::shared_state>(thread_pool, ncell, ncv, cv_to_cell,
                                                                         vinit, temp, diam, area,
                                                                         src_to_spike,
                                                                         fvm_detector_info{},
                                                                         mech->data_alignment());

    shared_state->instantiate(*mech, 0, mechanism_overrides{}, layout, {});
    shared_state->reset();

    // expect 0 value in state 'd' after init:
    mech->initialize();
    EXPECT_EQ(std::vector(ncv, -23.0), mechanism_field(mech, "d"));
    EXPECT_EQ(std::vector(ncv, -42.0), mechanism_field(mech, "a"));

    // expect original values in state 'd' and 'a' after state update:
    mech->update_state();
    EXPECT_EQ(diam, mechanism_field(mech, "d"));
    EXPECT_EQ(area, mechanism_field(mech, "a"));
}

TEST(mech_temperature, celsius) {
    run_celsius_test<multicore::backend>();
    run_diam_test<multicore::backend>();
}

#ifdef ARB_GPU_ENABLED
TEST(mech_temperature_gpu, celsius) {
    run_celsius_test<gpu::backend>();
    run_diam_test<gpu::backend>();
}
#endif

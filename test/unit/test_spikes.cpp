#include <cmath>

#include <gtest/gtest.h>

#include <arborio/label_parse.hpp>

#include <arborenv/concurrency.hpp>
#include <arborenv/gpu_env.hpp>

#include <arbor/load_balance.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike.hpp>

#include <backends/multicore/fvm.hpp>
#include <memory/memory.hpp>
#include <util/rangeutil.hpp>

#include "../simple_recipes.hpp"

using namespace arb;
using namespace arborio::literals;

// This source is included in `test_spikes_gpu.cpp`, which defines
// USE_BACKEND to override the default `multicore::backend`
// used for CPU tests.

#ifndef USE_BACKEND
using backend = multicore::backend;
#define SPIKES_TEST_CLASS spikes
#define GPU_ID -1
#else
using backend = USE_BACKEND;
#define SPIKES_TEST_CLASS spikes_gpu
#define GPU_ID 0
#endif

TEST(SPIKES_TEST_CLASS, threshold_watcher) {
    using value_type = backend::value_type;
    using index_type = backend::index_type;
    using array = backend::array;
    using iarray = backend::iarray;
    using list = std::vector<threshold_crossing>;

    // the test creates a watch on 3 values in the array values (which has 10
    // elements in total).
    execution_context context;
    const auto n = 10;

    const std::vector<index_type> index{0, 5, 7};
    const std::vector<value_type> thresh{1., 2., 3.};

    std::vector<int> src_to_spike_vec = {0, 1, 5};
    std::vector<arb_value_type> time_since_spike_vec(10);
    memory::fill(time_since_spike_vec, -1.0);

    // all values are initially 0, except for values[5] which we set
    // to exceed the threshold of 2. for the second watch
    array values(n, 0);
    values[5] = 3.;

    arb_value_type time_before = 0;
    arb_value_type time_after = 0;

    iarray src_to_spike(src_to_spike_vec.size());
    memory::copy(src_to_spike_vec, src_to_spike);

    array time_since_spike(10, -1.0);
    std::vector<unsigned> empty_slots = {2, 3, 4, 6, 7, 8, 9};

    // list for storing expected crossings for validation at the end
    list expected;

    // create the watch
    backend::threshold_watcher watch(
            values.size(),
            src_to_spike.data(),
            index,
            thresh,
            context);

    watch.reset(values);

    // initially the first and third watch should not be spiking
    //           the second is spiking
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));

    // test again at t=1, with unchanged values
    //  - nothing should change
    time_after = 1.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    memory::copy(time_since_spike, time_since_spike_vec);
    for (auto t: time_since_spike_vec) {
        EXPECT_EQ(-1.0, t);
    }

    // test at t=2, with all values set to zero
    //  - 2nd watch should now stop spiking
    memory::fill(values, 0.);
    time_before = time_after;
    time_after = 2.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // test at t=(2.5, 3), with all values set to 4.
    //  - all watches should now be spiking
    memory::fill(values, 4.);
    time_before = time_after;
    time_after = 3.0;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_TRUE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    // record the expected spikes
    expected.push_back({0u, 2.25f}); // 2. + (3-2)*(1./4.)
    expected.push_back({1u, 2.50f}); // 2. + (3-2)*(2./4.)
    expected.push_back({2u, 2.750f}); // 2. + (3-2)*(3./4.)

    memory::copy(time_since_spike, time_since_spike_vec);
    EXPECT_EQ(0.750, time_since_spike_vec[src_to_spike[0]]); // 3.0 - 2.25
    EXPECT_EQ(0.500, time_since_spike_vec[src_to_spike[1]]); // 3.0 - 2.50
    EXPECT_EQ(0.250, time_since_spike_vec[src_to_spike[2]]); // 3.0 - 2.750
    for (auto i: empty_slots) {
        EXPECT_EQ(-1.0, time_since_spike_vec[i]);
    }

    // test at t=4, with all values set to 0.
    //  - all watches should stop spiking
    memory::fill(values, 0.);
    time_before = time_after;
    time_after = 4;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    memory::copy(time_since_spike, time_since_spike_vec);
    for (auto t: time_since_spike_vec) {
        EXPECT_EQ(-1.0, t);
    }

    // test at t=5, with value on 3rd watch set to 6
    //  - watch 3 should be spiking
    values[index[2]] = 6.;
    time_before = time_after;
    time_after = 5;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_TRUE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 4u);
    expected.push_back({2u, 4.5f});

    memory::copy(time_since_spike, time_since_spike_vec);
    EXPECT_EQ(-1.0, time_since_spike_vec[src_to_spike[0]]);
    EXPECT_EQ(-1.0, time_since_spike_vec[src_to_spike[1]]);
    EXPECT_EQ(0.50, time_since_spike_vec[src_to_spike[2]]);
    for (auto i: empty_slots) {
        EXPECT_EQ(-1.0, time_since_spike_vec[i]);
    }

    //
    // test that all generated spikes matched the expected values
    //
    if (expected.size()!=watch.crossings().size()) {
        FAIL() << "count of recorded crosssings did not match expected count";
    }
    auto const& spikes = watch.crossings();
    for (auto i=0u; i<expected.size(); ++i) {
        EXPECT_EQ(expected[i], spikes[i]);
    }

    //
    // test that clearing works
    //
    watch.clear_crossings();
    EXPECT_EQ(watch.crossings().size(), 0u);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_TRUE(watch.is_crossed(2));

    //
    // test that resetting works
    //
    memory::fill(values, 0);
    values[index[0]] = 10.; // first watch should be intialized to spiking state
    time_before = 0;
    watch.reset(values);
    EXPECT_EQ(watch.crossings().size(), 0u);
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
}

namespace {
/// Crossings detected in one step are pushed by whichever thread detected them,
/// so their relative order is not part of the program on every back end. Sort by
/// detector index, then by time, before comparing against an expected list.
std::vector<threshold_crossing> sorted(const std::vector<threshold_crossing>& in) {
    auto out = in;
    util::sort(out, [](const auto& l, const auto& r) {
        return l.index != r.index? l.index < r.index: l.time < r.time;
    });
    return out;
}
} // namespace

// Two watches on the same value must not disturb each other: each keeps its own
// previous value, so both report the interpolated crossing time. If they shared
// one previous value per CV, the second watch would read the value the first has
// just written, interpolate over a zero interval, and record a NaN time.
TEST(SPIKES_TEST_CLASS, threshold_watcher_shared_cv) {
    using value_type = backend::value_type;
    using index_type = backend::index_type;
    using array = backend::array;
    using iarray = backend::iarray;
    using list = std::vector<threshold_crossing>;

    // both watches are on CV 0, at thresholds 1. and 2.
    execution_context context;
    const auto n = 10;

    const std::vector<index_type> index{0, 0};
    const std::vector<value_type> thresh{1., 2.};

    const std::vector<int> src_to_spike_vec{0, 1};

    array values(n, 0);
    array time_since_spike(2, -1.0);
    std::vector<arb_value_type> time_since_spike_vec(2, -1.0);

    iarray src_to_spike(src_to_spike_vec.size());
    memory::copy(src_to_spike_vec, src_to_spike);

    backend::threshold_watcher watch(
            values.size(),
            src_to_spike.data(),
            index,
            thresh,
            context);

    watch.reset(values);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));

    // nothing crosses while the value stays at zero
    arb_value_type time_before = 0;
    arb_value_type time_after = 1.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // one step from 0 to 4 crosses both thresholds: watch 0 a quarter of the way
    // into the step, watch 1 half way. These are the times each watch would
    // report if it were the only one on this CV.
    memory::fill(values, 4.);
    time_before = time_after;
    time_after = 2.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_EQ(watch.crossings().size(), 2u);

    list expected{{0u, 1.25}, {1u, 1.5}};
    EXPECT_EQ(sorted(expected), sorted(watch.crossings()));

    // drop below both thresholds, then cross them again: what each watch carries
    // into the next crossing is the value it saw last step, not the other's
    memory::fill(values, 0.);
    time_before = time_after;
    time_after = 3.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_EQ(watch.crossings().size(), 2u);

    memory::fill(values, 4.);
    time_before = time_after;
    time_after = 4.;
    watch.test(time_since_spike, time_before, time_after);
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_EQ(watch.crossings().size(), 4u);

    expected.push_back({0u, 3.25});
    expected.push_back({1u, 3.5});

    auto got = sorted(watch.crossings());
    EXPECT_EQ(sorted(expected), got);
    for (const auto& c: got) {
        EXPECT_TRUE(std::isfinite(c.time))
            << "detector " << c.index << " reported a crossing time of " << c.time;
    }

    memory::copy(time_since_spike, time_since_spike_vec);
    EXPECT_EQ(0.75, time_since_spike_vec[src_to_spike[0]]); // 4.0 - 3.25
    EXPECT_EQ(0.5,  time_since_spike_vec[src_to_spike[1]]); // 4.0 - 3.5
}

TEST(SPIKES_TEST_CLASS, threshold_watcher_interpolation) {
    auto dt = 0.025*arb::units::ms;
    auto duration = 1*arb::units::ms;

    arb::segment_tree tree;
    tree.append(arb::mnpos, { -6.3, 0.0, 0.0, 6.3}, {  6.3, 0.0, 0.0, 6.3}, 1);
    arb::morphology morpho(tree);

    arb::label_dict dict;
    dict.set("mid", arb::ls::on_branches(0.5));

    arb::proc_allocation resources;
    resources.num_threads = 1;
    resources.gpu_id = GPU_ID;
    auto context = arb::make_context(resources);

    std::vector<arb::spike> spikes;

    for (unsigned i = 0; i < 8; i++) {
        arb::decor decor;
        decor.place("mid"_lab, arb::threshold_detector{10*arb::units::mV}, "detector");
        decor.place("mid"_lab, arb::i_clamp::box(0.01*arb::units::ms + i*dt, duration, 0.5*arb::units::nA), "clamp");
        decor.place("mid"_lab, arb::synapse("expsyn"), "synapse");

        arb::cable_cell cell(morpho, decor, dict, arb::cv_policy_every_segment());
        cable1d_recipe rec({cell});

        auto decomp = arb::partition_load_balance(rec, context);
        arb::simulation sim(rec, context, decomp);

        sim.set_local_spike_callback(
                [&spikes](const std::vector<arb::spike>& recorded_spikes) {
                    spikes.insert(spikes.end(), recorded_spikes.begin(), recorded_spikes.end());
                });

        sim.run(duration, dt);
        ASSERT_EQ(1u, sim.num_spikes());
    }

    for (unsigned i = 1; i < spikes.size(); ++i) {
        EXPECT_NEAR(dt.value(), spikes[i].time - spikes[i-1].time, 1e-4);
    }
}

// v1 places one threshold detector on a region whose support is two locations in
// the same CV, which gives every cable cell two detectors watching one CV. Both
// must record a finite time: a crossing time that is not a number is dropped
// silently when the spike is delivered, so the cell loses that input entirely.
TEST(SPIKES_TEST_CLASS, two_detectors_on_one_cv) {
    arb::segment_tree tree;
    tree.append(arb::mnpos, { -6.3, 0.0, 0.0, 6.3}, {  6.3, 0.0, 0.0, 6.3}, 1);
    arb::morphology morpho(tree);

    arb::decor decor;
    // Two detectors at different locations on a cell whose cable is a single CV,
    // so both of them watch that one CV.
    decor.place(arb::mlocation{0, 0.25}, arb::threshold_detector{10*arb::units::mV}, "det0");
    decor.place(arb::mlocation{0, 0.75}, arb::threshold_detector{10*arb::units::mV}, "det1");
    decor.place(arb::mlocation{0, 0.5}, arb::i_clamp::box(0.01*arb::units::ms, 1*arb::units::ms, 0.5*arb::units::nA), "clamp");

    arb::cable_cell cell(morpho, decor, {}, arb::cv_policy_single());
    cable1d_recipe rec({cell});

    arb::proc_allocation resources;
    resources.num_threads = 1;
    resources.gpu_id = GPU_ID;
    auto context = arb::make_context(resources);

    std::vector<arb::spike> spikes;

    arb::simulation sim(rec, context, arb::partition_load_balance(rec, context));
    sim.set_local_spike_callback(
            [&spikes](const std::vector<arb::spike>& recorded_spikes) {
                spikes.insert(spikes.end(), recorded_spikes.begin(), recorded_spikes.end());
            });

    sim.run(1*arb::units::ms, 0.025*arb::units::ms);

    // each detector reports the crossing once, so neither spike is missing
    ASSERT_EQ(2u, spikes.size());
    EXPECT_EQ(2u, sim.num_spikes());

    // and neither crossing time is a NaN or an infinity
    for (const auto& s: spikes) {
        EXPECT_TRUE(std::isfinite(s.time))
            << "spike from detector " << s.source.index << " has time " << s.time;
    }

    // both detectors watch the same CV at the same threshold, so they agree on
    // when it crossed, and they are two distinct detectors
    EXPECT_NE(spikes[0].source.index, spikes[1].source.index);
    EXPECT_EQ(spikes[0].time, spikes[1].time);
}


#include "../gtest.h"

#include <spike.hpp>
#include <backends/multicore/fvm.hpp>
#include <memory/memory.hpp>
#include <util/rangeutil.hpp>

using namespace nest::mc;

template <typename C>
void dump(const C& c) {
    std::cout << "{ ";
    for (const auto& e: c) std::cout << e << " ";
    std::cout << "}\n";
}

#define DUMP(x) do { std::cout << "line " << __LINE__ << ": " #x << "\n"; dump(x); } while (false)

TEST(spikes, threshold_watcher) {
    using backend = multicore::backend;
    using size_type = backend::size_type;
    using value_type = backend::value_type;
    using array = backend::array;
    using iarray = backend::iarray;
    using list = backend::threshold_watcher::crossing_list;

    // the test creates a watch on 3 values in the array values (which has 10
    // elements in total).
    const auto n = 10;

    const std::vector<size_type> index{0, 5, 7};
    const std::vector<value_type> thresh{1., 2., 3.};

    // all values are initially 0, except for values[5] which we set
    // to exceed the threshold of 2. for the second watch
    array values(n, 0);
    values[5] = 3.;

    // the values are tied to two 'cells' with independent times:
    // compartments [0, 5] -> cell 0
    // compartments [6, 9] -> cell 1
    iarray cell_index(n, 0);
    for (unsigned i = 6; i<n; ++i) {
        cell_index[i] = 1;
    }
    array time_before(2, 0.);
    array time_after(2, 0.);

    // list for storing expected crossings for validation at the end
    list expected;

    // create the watch
    backend::threshold_watcher watch(cell_index, time_before, time_after, values, index, thresh);

    // initially the first and third watch should not be spiking
    //           the second is spiking
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));

    // test again at t=1, with unchanged values
    //  - nothing should change
    util::fill(time_after, 1.);
    watch.test();
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // test at t=2, with all values set to zero
    //  - 2nd watch should now stop spiking
    memory::fill(values, 0.);
    memory::copy(time_after, time_before);
    util::fill(time_after, 2.);
    DUMP(time_before);
    DUMP(time_after);
    watch.test();
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // test at t=(2.5, 3), with all values set to 4.
    //  - all watches should now be spiking
    memory::fill(values, 4.);
    memory::copy(time_after, time_before);
    time_after[0] = 2.5;
    time_after[1] = 3.0;
    watch.test();
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_TRUE(watch.is_crossed(1));
    EXPECT_TRUE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    // record the expected spikes
    expected.push_back({0u, 2.125f}); // 2. + (2.5-2)*(1./4.)
    expected.push_back({1u, 2.250f}); // 2. + (2.5-2)*(2./4.)
    expected.push_back({2u, 2.750f}); // 2. + (3.0-2)*(3./4.)

    // test at t=4, with all values set to 0.
    //  - all watches should stop spiking
    memory::fill(values, 0.);
    memory::copy(time_after, time_before);
    util::fill(time_after, 4.);
    watch.test();
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    // test at t=5, with value on 3rd watch set to 6
    //  - watch 3 should be spiking
    values[index[2]] = 6.;
    memory::copy(time_after, time_before);
    util::fill(time_after, 5.);
    watch.test();
    EXPECT_FALSE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_TRUE(watch.is_crossed(2));
    EXPECT_EQ(watch.crossings().size(), 4u);
    expected.push_back({2u, 4.5f});

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
    util::fill(time_before, 0.);
    watch.reset();
    EXPECT_EQ(watch.crossings().size(), 0u);
    EXPECT_TRUE(watch.is_crossed(0));
    EXPECT_FALSE(watch.is_crossed(1));
    EXPECT_FALSE(watch.is_crossed(2));
}


#include "../gtest.h"

#include <spike.hpp>
#include <backends/fvm_multicore.hpp>

using namespace nest::mc;

TEST(spikes, threshold_watcher) {
    using backend = multicore::backend;
    using size_type = backend::size_type;
    using value_type = backend::value_type;
    using array = backend::array;
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

    // list for storing expected crossings for validation at the end
    list expected;

    // create the watch
    backend::threshold_watcher watch(values, index, thresh, 0.f);

    // initially the first and third watch should not be spiking
    //           the second is spiking
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_TRUE(watch.is_spiking(1));
    EXPECT_FALSE(watch.is_spiking(2));

    // test again at t=1, with unchanged values
    //  - nothing should change
    watch.test(1.);
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_TRUE(watch.is_spiking(1));
    EXPECT_FALSE(watch.is_spiking(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // test at t=2, with all values set to zero
    //  - 2nd watch should now stop spiking
    memory::fill(values, 0.);
    watch.test(2.);
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_FALSE(watch.is_spiking(1));
    EXPECT_FALSE(watch.is_spiking(2));
    EXPECT_EQ(watch.crossings().size(), 0u);

    // test at t=3, with all values set to 4.
    //  - all watches should now be spiking
    memory::fill(values, 4.);
    watch.test(3.);
    EXPECT_TRUE(watch.is_spiking(0));
    EXPECT_TRUE(watch.is_spiking(1));
    EXPECT_TRUE(watch.is_spiking(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    // record the expected spikes
    expected.push_back({0u, 2.25f});
    expected.push_back({1u, 2.50f});
    expected.push_back({2u, 2.75f});

    // test at t=4, with all values set to 0.
    //  - all watches should stop spiking
    memory::fill(values, 0.);
    watch.test(4.);
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_FALSE(watch.is_spiking(1));
    EXPECT_FALSE(watch.is_spiking(2));
    EXPECT_EQ(watch.crossings().size(), 3u);

    // test at t=5, with value on 3rd watch set to 6
    //  - watch 3 should be spiking
    values[index[2]] = 6.;
    watch.test(5.);
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_FALSE(watch.is_spiking(1));
    EXPECT_TRUE(watch.is_spiking(2));
    EXPECT_EQ(watch.crossings().size(), 4u);
    expected.push_back({2u, 4.5f});

    //
    // test that all generated spikes matched the expected values
    //
    if (expected.size()!=watch.crossings().size()) {
        FAIL() << "count of recorded crosssings did not match expected count";
        return;
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
    EXPECT_FALSE(watch.is_spiking(0));
    EXPECT_FALSE(watch.is_spiking(1));
    EXPECT_TRUE(watch.is_spiking(2));

    //
    // test that resetting works
    //
    EXPECT_EQ(watch.last_test_time(), 5);
    memory::fill(values, 0);
    values[index[0]] = 10.; // first watch should be intialized to spiking state
    watch.reset(0);
    EXPECT_EQ(watch.last_test_time(), 0);
    EXPECT_EQ(watch.crossings().size(), 0u);
    EXPECT_TRUE(watch.is_spiking(0));
    EXPECT_FALSE(watch.is_spiking(1));
    EXPECT_FALSE(watch.is_spiking(2));
}


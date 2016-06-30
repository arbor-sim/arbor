#include "gtest.h"

#include <vector>

#include <event_queue.hpp>

TEST(event_queue, push)
{
    using namespace nest::mc;
    using ps_event_queue = event_queue<postsynaptic_spike_event>;

    ps_event_queue q;

    q.push({1u, 2.f, 2.f});
    q.push({4u, 1.f, 2.f});
    q.push({8u, 20.f, 2.f});
    q.push({2u, 8.f, 2.f});

    std::vector<float> times;
    while(q.size()) {
        times.push_back(
            q.pop_if_before(std::numeric_limits<float>::max())->time
        );
    }

    //std::copy(times.begin(), times.end(), std::ostream_iterator<float>(std::cout, ","));
    //std::cout << "\n";
    EXPECT_TRUE(std::is_sorted(times.begin(), times.end()));
}

TEST(event_queue, push_range)
{
    using namespace nest::mc;
    using ps_event_queue = event_queue<postsynaptic_spike_event>;

    postsynaptic_spike_event events[] = {
        {1u, 2.f, 2.f},
        {4u, 1.f, 2.f},
        {8u, 20.f, 2.f},
        {2u, 8.f, 2.f}
    };

    ps_event_queue q;
    q.push(std::begin(events), std::end(events));

    std::vector<float> times;
    while(q.size()) {
        times.push_back(
            q.pop_if_before(std::numeric_limits<float>::max())->time
        );
    }

    EXPECT_TRUE(std::is_sorted(times.begin(), times.end()));
}

TEST(event_queue, pop_if_before)
{
    using namespace nest::mc;
    using ps_event_queue = event_queue<postsynaptic_spike_event>;

    postsynaptic_spike_event events[] = {
        {1u, 1.f, 2.f},
        {2u, 2.f, 2.f},
        {3u, 3.f, 2.f},
        {4u, 4.f, 2.f}
    };

    ps_event_queue q;
    q.push(std::begin(events), std::end(events));

    EXPECT_EQ(q.size(), 4u);

    auto e1 = q.pop_if_before(0.);
    EXPECT_FALSE(e1);
    EXPECT_EQ(q.size(), 4u);

    auto e2 = q.pop_if_before(5.);
    EXPECT_TRUE(e2);
    EXPECT_EQ(e2->target, 1u);
    EXPECT_EQ(q.size(), 3u);

    auto e3 = q.pop_if_before(5.);
    EXPECT_TRUE(e3);
    EXPECT_EQ(e3->target, 2u);
    EXPECT_EQ(q.size(), 2u);

    auto e4 = q.pop_if_before(2.5);
    EXPECT_FALSE(e4);
    EXPECT_EQ(q.size(), 2u);

    auto e5 = q.pop_if_before(5.);
    EXPECT_TRUE(e5);
    EXPECT_EQ(e5->target, 3u);
    EXPECT_EQ(q.size(), 1u);

    q.pop_if_before(5.);
    EXPECT_EQ(q.size(), 0u);

    // empty queue should always return "false"
    auto e6 = q.pop_if_before(100.);
    EXPECT_FALSE(e6);
}

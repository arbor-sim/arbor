#include "../gtest.h"

#include <cmath>
#include <vector>

#include <arbor/spike_event.hpp>

#include "event_queue.hpp"

using namespace arb;

TEST(event_queue, push) {
    using ps_event_queue = event_queue<spike_event>;

    ps_event_queue q;

    q.push({{1u, 0u}, 2.f, 2.f});
    q.push({{4u, 1u}, 1.f, 2.f});
    q.push({{8u, 2u}, 20.f, 2.f});
    q.push({{2u, 3u}, 8.f, 2.f});

    EXPECT_EQ(4u, q.size());

    std::vector<float> times;
    float maxtime(INFINITY);
    while (!q.empty()) {
        times.push_back(q.pop_if_before(maxtime)->time);
    }

    EXPECT_TRUE(std::is_sorted(times.begin(), times.end()));
}

TEST(event_queue, pop_if_before) {
    using ps_event_queue = event_queue<spike_event>;

    cell_member_type target[4] = {
        {1u, 0u},
        {4u, 1u},
        {8u, 2u},
        {2u, 3u}
    };

    spike_event events[] = {
        {target[0], 1.f, 2.f},
        {target[1], 2.f, 2.f},
        {target[2], 3.f, 2.f},
        {target[3], 4.f, 2.f}
    };

    ps_event_queue q;
    for (const auto& ev: events) {
        q.push(ev);
    }

    EXPECT_EQ(4u, q.size());

    auto e1 = q.pop_if_before(0.);
    EXPECT_FALSE(e1);
    EXPECT_EQ(4u, q.size());

    auto e2 = q.pop_if_before(5.);
    EXPECT_TRUE(e2);
    EXPECT_EQ(e2->target, target[0]);
    EXPECT_EQ(3u, q.size());

    auto e3 = q.pop_if_before(5.);
    EXPECT_TRUE(e3);
    EXPECT_EQ(e3->target, target[1]);
    EXPECT_EQ(2u, q.size());

    auto e4 = q.pop_if_before(2.5);
    EXPECT_FALSE(e4);
    EXPECT_EQ(2u, q.size());

    auto e5 = q.pop_if_before(5.);
    EXPECT_TRUE(e5);
    EXPECT_EQ(e5->target, target[2]);
    EXPECT_EQ(1u, q.size());

    q.pop_if_before(5.);
    EXPECT_EQ(0u, q.size());
    EXPECT_TRUE(q.empty());

    // empty queue should always return "false"
    auto e6 = q.pop_if_before(100.);
    EXPECT_FALSE(e6);
}

TEST(event_queue, pop_if_not_after) {
    struct event {
        int time;

        event(int t): time(t) {}
    };

    event_queue<event> queue;

    queue.push(1);
    queue.push(3);
    queue.push(5);

    auto e1 = queue.pop_if_not_after(2);
    EXPECT_TRUE(e1);
    EXPECT_EQ(1, e1->time);

    auto e2 = queue.pop_if_before(3);
    EXPECT_FALSE(e2);

    auto e3 = queue.pop_if_not_after(3);
    EXPECT_TRUE(e3);
    EXPECT_EQ(3, e3->time);

    auto e4 = queue.pop_if_not_after(4);
    EXPECT_FALSE(e4);
}

// Event queues can be defined for arbitrary copy-constructible events
// for which `event_time(ev)` returns the corresponding time. Time values just
// need to be well-ordered on '>'.

struct wrapped_float {
    wrapped_float() {}
    wrapped_float(float f): f(f) {}

    float f;
    bool operator>(wrapped_float x) const { return f>x.f; }
};

struct minimal_event {
    wrapped_float value;
    explicit minimal_event(float x): value(x) {}
};

const wrapped_float& event_time(const minimal_event& ev) { return ev.value; }

TEST(event_queue, minimal_event_impl) {
    minimal_event events[] = {
        minimal_event(3.f),
        minimal_event(2.f),
        minimal_event(2.f),
        minimal_event(10.f)
    };

    std::vector<float> expected;
    for (const auto& ev: events) {
        expected.push_back(ev.value.f);
    }
    std::sort(expected.begin(), expected.end());

    event_queue<minimal_event> q;
    for (auto& ev: events) {
        q.push(ev);
    }

    wrapped_float maxtime(INFINITY);

    std::vector<float> times;
    while (q.size()) {
        times.push_back(q.pop_if_before(maxtime)->value.f);
    }

    EXPECT_EQ(expected, times);
}


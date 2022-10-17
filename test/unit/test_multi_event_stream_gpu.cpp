#include <cstdio>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <backends/event.hpp>
#include <backends/gpu/multi_event_stream.hpp>
#include <memory/wrappers.hpp>
#include <util/rangeutil.hpp>

using namespace arb;

namespace {
    auto evtime = [](deliverable_event e) { return event_time(e); };
    auto evindex = [](deliverable_event e) { return event_index(e); };
}

using deliverable_event_stream = gpu::multi_event_stream<deliverable_event>;

namespace {
    // set up four targets across three streams and two mech ids.

    constexpr cell_local_size_type mech_1 = 10u;
    constexpr cell_local_size_type mech_2 = 13u;
    constexpr cell_size_type cell_1 = 20u;
    constexpr cell_size_type cell_2 = 77u;
    constexpr cell_size_type cell_3 = 33u;
    constexpr cell_size_type n_cell = 100u;

    target_handle handle[4] = {
        target_handle(mech_1, 0u, cell_1),
        target_handle(mech_2, 1u, cell_2),
        target_handle(mech_1, 4u, cell_2),
        target_handle(mech_2, 2u, cell_3)
    };

    // cell_1 (handle 0) has one event at t=3
    // cell_2 (handle 1 and 2) has two events at t=2 and t=5
    // cell_3 (handle 3) has one event at t=3

    std::vector<deliverable_event> common_events = {
        deliverable_event(2.f, handle[1], 2.f),
        deliverable_event(3.f, handle[0], 1.f),
        deliverable_event(3.f, handle[3], 4.f),
        deliverable_event(5.f, handle[2], 3.f)
    };
}

TEST(multi_event_stream_gpu, init) {
    deliverable_event_stream m(n_cell);
    EXPECT_EQ(n_cell, m.n_streams());

    auto events = common_events;
    ASSERT_TRUE(util::is_sorted_by(events, evtime));
    util::stable_sort_by(events, evindex);
    m.init(events);
    EXPECT_FALSE(m.empty());

    m.clear();
    EXPECT_TRUE(m.empty());
}

// CUDA kernel wrapper:
void run_copy_marked_events_kernel(
    unsigned ci,
    deliverable_event_stream::state state,
    deliverable_event_data* store,
    unsigned& count,
    unsigned max_ev);

std::vector<deliverable_event_data> copy_marked_events(int ci, deliverable_event_stream& m) {
    unsigned max_ev = 1000;
    memory::device_vector<deliverable_event_data> store(max_ev);
    memory::device_vector<unsigned> counter(1);

    run_copy_marked_events_kernel(ci, m.marked_events(), store.data(), *counter.data(), max_ev);
    unsigned n_ev = counter[0];
    std::vector<deliverable_event_data> ev(n_ev);
    memory::copy(store(0, n_ev), ev);
    return ev;
}

TEST(multi_event_stream_gpu, mark) {
    deliverable_event_stream m(n_cell);
    ASSERT_EQ(n_cell, m.n_streams());

    auto events = common_events;
    ASSERT_TRUE(util::is_sorted_by(events, evtime));
    util::stable_sort_by(events, evindex);
    m.init(events);

    for (cell_size_type i = 0; i<n_cell; ++i) {
        EXPECT_TRUE(copy_marked_events(i, m).empty());
    }

    memory::device_vector<double> t_until(n_cell);
    t_until[cell_1] = 2.;
    t_until[cell_2] = 2.5;
    t_until[cell_3] = 4.;

    m.mark_until_after(t_until);

    // Only two events should be marked: 
    //     events[0] (with handle 1) at t=2 on cell_2
    //     events[2] (with handle 3) at t=3 on cell_3

    for (cell_size_type i = 0; i<n_cell; ++i) {
        auto evs = copy_marked_events(i, m);
        auto n_marked = evs.size();
        switch (i) {
        case cell_2:
            ASSERT_EQ(1u, n_marked);
            EXPECT_EQ(handle[1].mech_id, evs.front().mech_id);
            EXPECT_EQ(handle[1].mech_index, evs.front().mech_index);
            break;
        case cell_3:
            ASSERT_EQ(1u, n_marked);
            EXPECT_EQ(handle[3].mech_id, evs.front().mech_id);
            EXPECT_EQ(handle[3].mech_index, evs.front().mech_index);
            break;
        default:
            EXPECT_EQ(0u, n_marked);
            break;
        }
    }

    // Drop these events and mark all events up to t=5.
    //     cell_1 should have one marked event (events[1], handle 0)
    //     cell_2 should have one marked event (events[3], handle 2)

    m.drop_marked_events();
    memory::fill(t_until, 5.);
    m.mark_until_after(memory::make_view(t_until));

    for (cell_size_type i = 0; i<n_cell; ++i) {
        auto evs = copy_marked_events(i, m);
        auto n_marked = evs.size();
        switch (i) {
        case cell_1:
            ASSERT_EQ(1u, n_marked);
            EXPECT_EQ(handle[0].mech_id, evs.front().mech_id);
            EXPECT_EQ(handle[0].mech_index, evs.front().mech_index);
            break;
        case cell_2:
            ASSERT_EQ(1u, n_marked);
            EXPECT_EQ(handle[2].mech_id, evs.front().mech_id);
            EXPECT_EQ(handle[2].mech_index, evs.front().mech_index);
            break;
        default:
            EXPECT_EQ(0u, n_marked);
            break;
        }
    }

    // No more events after these.
    EXPECT_FALSE(m.empty());
    m.drop_marked_events();
    EXPECT_TRUE(m.empty());

    // Confirm different semantics of `mark_until`.

    m.init(events);
    t_until[cell_1] = 3.1f;
    t_until[cell_2] = 1.9f;
    t_until[cell_3] = 3.f;
    m.mark_until(t_until);

    // Only one event should be marked: 
    //     events[1] (with handle 0) at t=3.f on cell_1
    //
    // events[2] at 3.f on cell_3 should not be marked (3.f not less than 3.f)
    // events[0] at 2.f on cell_2 should not be marked (2.f not less than 1.9f)
    //     events[2] (with handle 3) at t=3.f on cell_3
    //     events[0] (with handle 1) at t=2.f on cell_2 should _not_ be marked.

    for (cell_size_type i = 0; i<n_cell; ++i) {
        auto evs = copy_marked_events(i, m);
        auto n_marked = evs.size();
        switch (i) {
        case cell_1:
            EXPECT_EQ(1u, n_marked);
            EXPECT_EQ(handle[0].mech_id, evs.front().mech_id);
            EXPECT_EQ(handle[0].mech_index, evs.front().mech_index);
            break;
        default:
            EXPECT_EQ(0u, n_marked);
            break;
        }
    }
}

TEST(multi_event_stream_gpu, time_if_before) {
    deliverable_event_stream m(n_cell);
    ASSERT_EQ(n_cell, m.n_streams());

    auto events = common_events;
    ASSERT_TRUE(util::is_sorted_by(events, evtime));
    util::stable_sort_by(events, evindex);
    m.init(events);

    // Test times less than all event times (first event at t=2).
    std::vector<double> before(n_cell);
    std::vector<double> after;

    for (unsigned i = 0; i<n_cell; ++i) {
        before[i] = 0.1+i/(double)n_cell;
    }

    memory::device_vector<double> t = memory::on_gpu(before);
    m.event_time_if_before(t);
    util::assign(after, memory::on_host(t));

    EXPECT_EQ(before, after);

    // With times between 2 and 3, expect the event at time t=2
    // on cell_2 to restrict corresponding element of t.

    for (unsigned i = 0; i<n_cell; ++i) {
        before[i] = 2.1+0.5*i/(double)n_cell;
    }
    t = memory::make_view(before);
    m.event_time_if_before(t);
    util::assign(after, memory::on_host(t));

    for (unsigned i = 0; i<n_cell; ++i) {
        if (i==cell_2) {
            EXPECT_EQ(2., after[i]);
        }
        else {
            EXPECT_EQ(before[i], after[i]);
        }
    }
}

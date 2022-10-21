#include <gtest/gtest.h>

#include <arbor/spike.hpp>
#include <arborenv/default_env.hpp>

#include "execution_context.hpp"
#include "thread_private_spike_store.hpp"

using arb::spike;

TEST(spike_store, insert)
{
    using store_type = arb::thread_private_spike_store;

    arb::execution_context context({arbenv::default_concurrency(), -1});
    store_type store(context.thread_pool);

    // insert 3 spike events and check that they were inserted correctly
    store.insert({
        {{0,0}, 0.0f},
        {{1,2}, 0.5f},
        {{2,4}, 1.0f}
    });

    {
        EXPECT_EQ(store.get().size(), 3u);
        auto i = 0u;
        for (auto& spike : store.get()) {
            EXPECT_EQ(spike.source.gid,   i);
            EXPECT_EQ(spike.source.index, 2*i);
            EXPECT_EQ(spike.time, float(i)/2.f);
            ++i;
        }
    }

    // insert another 3 events, then check that they were appended to the
    // original three events correctly
    store.insert({
        {{3,6},  1.5f},
        {{4,8},  2.0f},
        {{5,10}, 2.5f}
    });

    {
        EXPECT_EQ(store.get().size(), 6u);
        auto i = 0u;
        for (auto& spike : store.get()) {
            EXPECT_EQ(spike.source.gid,   i);
            EXPECT_EQ(spike.source.index, 2*i);
            EXPECT_EQ(spike.time, float(i)/2.f);
            ++i;
        }
    }
}

TEST(spike_store, clear)
{
    using store_type = arb::thread_private_spike_store;

    arb::execution_context context({arbenv::default_concurrency(), -1});
    store_type store(context.thread_pool);

    // insert 3 spike events
    store.insert({
        {{0,0}, 0.0f}, {{1,2}, 0.5f}, {{2,4}, 1.0f}
    });
    EXPECT_EQ(store.get().size(), 3u);
    store.clear();
    EXPECT_EQ(store.get().size(), 0u);
}

TEST(spike_store, gather)
{
    using store_type = arb::thread_private_spike_store;

    arb::execution_context context({arbenv::default_concurrency(), -1});
    store_type store(context.thread_pool);

    std::vector<spike> spikes =
        { {{0,0}, 0.0f}, {{1,2}, 0.5f}, {{2,4}, 1.0f} };

    store.insert(spikes);
    auto gathered_spikes = store.gather();

    EXPECT_EQ(gathered_spikes.size(), spikes.size());

    for(auto i=0u; i<spikes.size(); ++i) {
        EXPECT_EQ(spikes[i].source.gid, gathered_spikes[i].source.gid);
        EXPECT_EQ(spikes[i].source.index, gathered_spikes[i].source.index);
        EXPECT_EQ(spikes[i].time, gathered_spikes[i].time);
    }
}

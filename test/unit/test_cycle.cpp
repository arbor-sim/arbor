#include "../gtest.h"

#include <algorithm>
#include <iterator>
#include <string>

#include "common.hpp"
#include <util/cycle.hpp>
#include <util/meta.hpp>

using namespace arb;

TEST(cycle_iterator, construct) {
    std::vector<int> values = { 4, 2, 3 };
    auto cycle_iter = util::make_cyclic_iterator(values.cbegin(), values.cend());

    {
        // copy constructor
        auto cycle_iter_copy(cycle_iter);
        EXPECT_EQ(cycle_iter, cycle_iter_copy);
    }

    {
        // copy assignment
        auto cycle_iter_copy = cycle_iter;
        EXPECT_EQ(cycle_iter, cycle_iter_copy);
    }

    {
        // move constructor
        auto cycle_iter_copy(
            util::make_cyclic_iterator(values.cbegin(), values.cend())
        );
        EXPECT_EQ(cycle_iter, cycle_iter_copy);
    }
}


TEST(cycle_iterator, increment) {
    std::vector<int> values = { 4, 2, 3 };

    {
        // test operator++
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());
        auto cycle_iter_copy = cycle_iter;

        auto values_size = values.size();
        for (auto i = 0u; i < 2*values_size; ++i) {
            EXPECT_EQ(values[i % values_size], *cycle_iter);
            EXPECT_EQ(values[i % values_size], *cycle_iter_copy++);
            ++cycle_iter;
        }
    }

    {
        // test operator[]
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());

        for (auto i = 0u; i < values.size(); ++i) {
            EXPECT_EQ(values[i], cycle_iter[values.size() + i]);
        }
    }

    {
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());
        EXPECT_NE(cycle_iter + 1, cycle_iter + 10);
    }
}

TEST(cycle_iterator, decrement) {
    std::vector<int> values = { 4, 2, 3 };

    {
        // test operator--
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());
        auto cycle_iter_copy = cycle_iter;

        auto values_size = values.size();
        for (auto i = 0u; i < 2*values_size; ++i) {
            --cycle_iter;
            cycle_iter_copy--;
            auto val = values[values_size - i%values_size - 1];
            EXPECT_EQ(val, *cycle_iter);
            EXPECT_EQ(val, *cycle_iter_copy);
        }
    }

    {
        // test operator[]
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());
        int values_size = values.size();
        for (int i = 0; i < 2*values_size; ++i) {
            auto pos = i % values_size;
            pos = pos ? values_size - pos : 0;
            EXPECT_EQ(values[pos], cycle_iter[-i]);
        }
    }

    {
        auto cycle_iter = util::make_cyclic_iterator(values.cbegin(),
                                                     values.cend());
        EXPECT_NE(cycle_iter - 2, cycle_iter - 5);
        EXPECT_NE(cycle_iter + 1, cycle_iter - 5);
    }
}

TEST(cycle_iterator, carray) {
    int values[] = { 4, 2, 3 };
    auto cycle_iter = util::make_cyclic_iterator(std::cbegin(values),
                                                 std::cend(values));
    auto values_size = util::size(values);
    for (auto i = 0u; i < 2*values_size; ++i) {
        EXPECT_EQ(values[i % values_size], *cycle_iter++);
    }
}

TEST(cycle_iterator, sentinel) {
    using testing::null_terminated;

    auto msg = "hello";
    auto cycle_iter = util::make_cyclic_iterator(msg, null_terminated);

    auto msg_len = std::string(msg).size();
    for (auto i = 0u; i < 2*msg_len; ++i) {
        EXPECT_EQ(msg[i % msg_len], *cycle_iter++);
    }
}


TEST(cycle, cyclic_view) {
    std::vector<int> values = { 4, 2, 3 };
    std::vector<int> values_new;

    std::copy_n(util::cyclic_view(values).cbegin(), 10,
                std::back_inserter(values_new));

    EXPECT_EQ(10u, values_new.size());

    auto i = 0;
    for (auto const& v : values_new) {
        EXPECT_EQ(values[i++ % values.size()], v);
    }
}

TEST(cycle, cyclic_view_initlist) {
    std::vector<int> values;

    std::copy_n(util::cyclic_view({2., 3., 4.}).cbegin(), 10,
                std::back_inserter(values));

    EXPECT_EQ(10u, values.size());

    auto i = 0;
    for (auto const& v : values) {
        EXPECT_EQ(2 + i++ % 3, v);
    }
}

TEST(cycle_iterator, difference) {
    int values[] = { 4, 2, 3 };

    auto cycle = util::cyclic_view(values);
    auto c1 = cycle.begin();

    auto c2 = c1;
    EXPECT_EQ(0, c2-c1);

    ++c2;
    EXPECT_EQ(1, c2-c1);

    ++c1;
    EXPECT_EQ(0, c2-c1);

    c2 += 6;
    EXPECT_EQ(6, c2-c1);

    c1 += 2;
    EXPECT_EQ(4, c2-c1);

    --c2;
    EXPECT_EQ(3, c2-c1);

    c1 -= 3;
    EXPECT_EQ(6, c2-c1);
}

TEST(cycle_iterator, order) {
    int values[] = { 4, 2, 3 };

    auto cycle = util::cyclic_view(values);
    auto c1 = cycle.begin();
    auto c2 = c1;

    EXPECT_FALSE(c1 < c2);
    EXPECT_FALSE(c2 < c1);
    EXPECT_TRUE(c1 <= c2);
    EXPECT_TRUE(c1 >= c2);

    c2 += util::size(values);

    EXPECT_TRUE(c1 < c2);
    EXPECT_FALSE(c2 < c1);
    EXPECT_TRUE(c1 <= c2);
    EXPECT_FALSE(c1 >= c2);
}

TEST(cycle, cyclic_view_sentinel) {
    const char *msg = "hello";
    auto cycle = util::cyclic_view(
        util::make_range(msg, testing::null_terminated)
    );

    std::string msg_new;
    auto msg_new_size = 2*std::string(msg).size();
    for (auto i = 0u; i < msg_new_size; ++i) {
        msg_new += cycle[i];
    }

    EXPECT_EQ("hellohello", msg_new);
}

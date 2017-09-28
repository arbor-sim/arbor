#ifdef ARB_HAVE_MPI

#include "../gtest.h"

#include <cstring>
#include <vector>

#include <communication/global_policy.hpp>
#include <communication/mpi.hpp>
#include <util/rangeutil.hpp>

using namespace arb;
using namespace arb::communication;

struct big_thing {
    big_thing() {}
    big_thing(int i): value_(i) {}

    bool operator==(const big_thing& other) const {
        return value_==other.value_ && !std::memcmp(salt_, other.salt_, sizeof(salt_));
    }

    bool operator!=(const big_thing& other) const {
        return !(*this==other);
    }

private:
    int value_;
    char salt_[32] = "it's a lovely day for a picnic";
};

TEST(mpi, gather_all) {
    using policy = mpi_global_policy;

    int id = policy::id();

    std::vector<big_thing> data;
    // odd ranks: three items; even ranks: one item.
    if (id%2) {
        data = { id, id+7, id+8 };
    }
    else {
        data = { id };
    }

    std::vector<big_thing> expected;
    for (int i = 0; i<policy::size(); ++i) {
        if (i%2) {
            int rank_data[] = { i, i+7, i+8 };
            util::append(expected, rank_data);
        }
        else {
            int rank_data[] = { i };
            util::append(expected, rank_data);
        }
    }

    auto gathered = mpi::gather_all(data);

    EXPECT_EQ(expected, gathered);
}

TEST(mpi, gather_all_with_partition) {
    using policy = mpi_global_policy;

    int id = policy::id();

    std::vector<big_thing> data;
    // odd ranks: three items; even ranks: one item.
    if (id%2) {
        data = { id, id+7, id+8 };
    }
    else {
        data = { id };
    }

    std::vector<big_thing> expected_values;
    std::vector<unsigned> expected_divisions;

    expected_divisions.push_back(0);
    for (int i = 0; i<policy::size(); ++i) {
        if (i%2) {
            int rank_data[] = { i, i+7, i+8 };
            util::append(expected_values, rank_data);
            expected_divisions.push_back(expected_divisions.back()+util::size(rank_data));
        }
        else {
            int rank_data[] = { i };
            util::append(expected_values, rank_data);
            expected_divisions.push_back(expected_divisions.back()+util::size(rank_data));
        }
    }

    auto gathered = mpi::gather_all_with_partition(data);

    EXPECT_EQ(expected_values, gathered.values());
    EXPECT_EQ(expected_divisions, gathered.partition());
}

TEST(mpi, gather_string) {
    using policy = mpi_global_policy;

    int id = policy::id();

    // Make a string of variable length, with the character
    // in the string distrubuted as follows
    // rank string
    //  0   a
    //  1   bb
    //  2   ccc
    //  3   dddd
    //   ...
    // 25   zzzz...zzz   (26 times z)
    // 26   aaaa...aaaa  (27 times a)
    auto make_string = [](int id) {
        return std::string(id+1, 'a'+char(id%26));};

    auto s = make_string(id);

    auto gathered = mpi::gather(s, 0);

    if (!id) {
        ASSERT_TRUE(policy::size()==(int)gathered.size());
        for (std::size_t i=0; i<gathered.size(); ++i) {
            EXPECT_EQ(make_string(i), gathered[i]);
        }
    }
}

TEST(mpi, gather) {
    using policy = mpi_global_policy;

    int id = policy::id();

    auto gathered = mpi::gather(id, 0);

    if (!id) {
        ASSERT_TRUE(policy::size()==(int)gathered.size());
        for (std::size_t i=0; i<gathered.size(); ++i) {
            EXPECT_EQ(int(i), gathered[i]);
        }
    }
    else {
        EXPECT_EQ(0u, gathered.size());
    }
}

#endif // ARB_HAVE_MPI

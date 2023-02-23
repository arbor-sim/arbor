#ifdef TEST_MPI

#include <gtest/gtest.h>
#include "test.hpp"

#include <cstring>
#include <vector>

#include <communication/mpi.hpp>
#include <util/rangeutil.hpp>

using namespace arb;

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
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

    std::vector<big_thing> data;
    // odd ranks: three items; even ranks: one item.
    if (id%2) {
        data = { id, id+7, id+8 };
    }
    else {
        data = { id };
    }

    std::vector<big_thing> expected;
    for (int i = 0; i<size; ++i) {
        if (i%2) {
            int rank_data[] = { i, i+7, i+8 };
            util::append(expected, rank_data);
        }
        else {
            int rank_data[] = { i };
            util::append(expected, rank_data);
        }
    }

    auto gathered = mpi::gather_all(data, MPI_COMM_WORLD);

    EXPECT_EQ(expected, gathered);
}

TEST(mpi, gather_all_nested_vec) {
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

    std::vector<std::vector<big_thing>> data;
    if (id%2) {
        for (int s = 0; s < (id+1)*2; s++) {
            data.push_back({id + s, id + s + 7, id + s + 8});
        }
    }
    else {
        for (int s = 0; s < (id+1)*2; s++) {
            data.push_back({id + 2*s});
        }
    }

    std::vector<std::vector<big_thing>> expected;
    for (int i = 0; i<size; ++i) {
        if (i%2) {
            for (int s = 0; s < (i+1)*2; s++) {
                expected.push_back({i + s, i + s + 7, i + s + 8});
            }
        }
        else {
            for (int s = 0; s < (i+1)*2; s++) {
                expected.push_back({i + 2*s});
            }
        }
    }

    auto gathered = mpi::gather_all(data, MPI_COMM_WORLD);
    EXPECT_EQ(expected, gathered);
}

TEST(mpi, gather_all_with_partition) {
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

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
    for (int i = 0; i<size; ++i) {
        if (i%2) {
            int rank_data[] = { i, i+7, i+8 };
            util::append(expected_values, rank_data);
            expected_divisions.push_back(expected_divisions.back()+std::size(rank_data));
        }
        else {
            int rank_data[] = { i };
            util::append(expected_values, rank_data);
            expected_divisions.push_back(expected_divisions.back()+std::size(rank_data));
        }
    }

    auto gathered = mpi::gather_all_with_partition(data, MPI_COMM_WORLD);

    EXPECT_EQ(expected_values, gathered.values());
    EXPECT_EQ(expected_divisions, gathered.partition());
}

TEST(mpi, gather_string) {
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

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

    auto gathered = mpi::gather(s, 0, MPI_COMM_WORLD);

    if (!id) {
        ASSERT_TRUE(size==(int)gathered.size());
        for (std::size_t i=0; i<gathered.size(); ++i) {
            EXPECT_EQ(make_string(i), gathered[i]);
        }
    }
}

TEST(mpi, gather_string_vec) {
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

    // Make a vector of strings of variable length:
    // rank strings
    //  0   a
    //  1   b; bb
    //  2   c; cc; ccc
    //  3   d; dd; ddd; dddd
    //   ...
    // 25   z; zz; ...; zzzz...zzz   (26 times z)
    // 26   a; aa; ...; aaaa...aaaa  (27 times a)
    auto make_string = [](int length, int id) {
      return std::string(length, 'a'+char(id%26));};

    std::vector<std::string> string_vec(id+1);
    for (int i = 0; i < id+1; ++i) {
        string_vec[i] = make_string(i+1, id);
    }

    auto gathered = mpi::gather_all(string_vec, MPI_COMM_WORLD);

    int expected_size = size*(size+1)/2;
    ASSERT_TRUE(expected_size==(int)gathered.size());

    int idx = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < i+1; ++j) {
            EXPECT_EQ(make_string(j+1, i), gathered[idx++]);
        }
    }
}

TEST(mpi, gather) {
    int id = mpi::rank(MPI_COMM_WORLD);
    int size = mpi::size(MPI_COMM_WORLD);

    auto gathered = mpi::gather(id, 0, MPI_COMM_WORLD);

    if (!id) {
        ASSERT_TRUE(size==(int)gathered.size());
        for (std::size_t i=0; i<gathered.size(); ++i) {
            EXPECT_EQ(int(i), gathered[i]);
        }
    }
    else {
        EXPECT_EQ(0u, gathered.size());
    }
}

#endif // TEST_MPI

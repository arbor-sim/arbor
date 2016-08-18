#include "gtest.h"

#include <forward_list>
#include <vector>

#include <util/partition.hpp>

using namespace nest::mc;

TEST(partition, partition_view) {
    std::forward_list<int> fl = {1, 4, 6, 8, 10 };

    auto p1 = util::partition_view(fl);
    EXPECT_EQ(std::make_pair(1,4), p1.front());
    EXPECT_EQ(std::make_pair(8,10), p1.back());
    EXPECT_EQ(std::make_pair(1,10), p1.bounds());
    EXPECT_EQ(4u, p1.size());

    std::vector<double> v = {2.0, 3.6, 7.5};

    auto p2 = util::partition_view(v);
    EXPECT_EQ(2u, p2.size());

    std::vector<double> ends;
    std::vector<double> ends_expected = { 2.0, 3.6, 3.6, 7.5 };
    for (auto b: p2) {
        ends.push_back(b.first);
        ends.push_back(b.second);
    }
    EXPECT_EQ(ends_expected, ends);
}

TEST(partition, partition_view_find) {
    std::vector<double> divs = { 1, 2.5, 3, 5.5 };
    double eps = 0.1;
    auto p = util::partition_view(divs);

    EXPECT_EQ(p.end(), p.find(divs.front()-eps));
    EXPECT_NE(p.end(), p.find(divs.front()));
    EXPECT_EQ(divs.front(), p.find(divs.front())->first);

    EXPECT_NE(p.end(), p.find(divs.back()-eps));
    EXPECT_EQ(divs.back(), p.find(divs.back()-eps)->second);
    EXPECT_EQ(p.end(), p.find(divs.back()));
    EXPECT_EQ(p.end(), p.find(divs.back()+eps));

    EXPECT_EQ(divs[1], p.find(divs[1]+eps)->first);
    EXPECT_EQ(divs[2], p.find(divs[1]+eps)->second);
}

TEST(partition, make_partition_in_place) {
    unsigned sizes[] = { 7, 3, 0, 2 };
    unsigned part_store[util::size(sizes)+1];

    auto p = util::make_partition(util::partition_in_place, part_store, sizes, 10u);
    ASSERT_EQ(4u, p.size());
    EXPECT_EQ(std::make_pair(10u, 17u), p[0]);
    EXPECT_EQ(std::make_pair(17u, 20u), p[1]);
    EXPECT_EQ(std::make_pair(20u, 20u), p[2]);
    EXPECT_EQ(std::make_pair(20u, 22u), p[3]);

    // with short sizes sequence
    unsigned short_sizes[] = { 1, 2 };
    p = util::make_partition(util::partition_in_place, part_store, short_sizes, 0u);
    ASSERT_EQ(4u, p.size());
    EXPECT_EQ(std::make_pair(0u, 1u), p[0]);
    EXPECT_EQ(std::make_pair(1u, 3u), p[1]);
    EXPECT_EQ(std::make_pair(3u, 3u), p[2]);
    EXPECT_EQ(std::make_pair(3u, 3u), p[3]);
}

TEST(partition, make_partition) {
    unsigned sizes[] = { 7, 3, 0, 2 };
    std::forward_list<double> part_store = { 100.3 };

    auto p = util::make_partition(part_store, sizes, 10.0);
    ASSERT_EQ(4u, p.size());

    auto pi = p.begin();
    EXPECT_EQ(10.0, pi++->first);
    EXPECT_EQ(17.0, pi++->first);
    EXPECT_EQ(20.0, pi++->first);
    EXPECT_EQ(20.0, pi->first);
    EXPECT_EQ(22.0, pi->second);

    EXPECT_EQ(p.end(), ++pi);
}

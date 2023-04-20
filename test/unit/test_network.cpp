#include <gtest/gtest.h>

#include <arbor/network.hpp>

#include "network_impl.hpp"

#include <tuple>
#include <vector>

using namespace arb;

namespace {
std::vector<network_site_info> test_sites = {
    {0, 0, cell_kind::cable, "a", {1, 0.5}, {0.0, 0.0, 0.0}},
    {1, 0, cell_kind::benchmark, "b", {0, 0.0}, {1.0, 0.0, 0.0}},
    {2, 0, cell_kind::lif, "c", {0, 0.0}, {2.0, 0.0, 0.0}},
    {3, 0, cell_kind::spike_source, "d", {0, 0.0}, {3.0, 0.0, 0.0}},
    {4, 0, cell_kind::cable, "e", {0, 0.2}, {4.0, 0.0, 0.0}},
    {5, 0, cell_kind::cable, "f", {5, 0.1}, {5.0, 0.0, 0.0}},
    {6, 0, cell_kind::cable, "g", {4, 0.3}, {6.0, 0.0, 0.0}},
    {7, 0, cell_kind::cable, "h", {0, 1.0}, {7.0, 0.0, 0.0}},
    {9, 0, cell_kind::cable, "i", {0, 0.1}, {12.0, 3.0, 4.0}},

    {10, 0, cell_kind::cable, "a", {0, 0.1}, {12.0, 15.0, 16.0}},
    {10, 1, cell_kind::cable, "b", {1, 0.1}, {13.0, 15.0, 16.0}},
    {10, 2, cell_kind::cable, "c", {1, 0.5}, {14.0, 15.0, 16.0}},
    {10, 3, cell_kind::cable, "d", {1, 1.0}, {15.0, 15.0, 16.0}},
    {10, 4, cell_kind::cable, "e", {2, 0.1}, {16.0, 15.0, 16.0}},
    {10, 5, cell_kind::cable, "f", {3, 0.1}, {16.0, 16.0, 16.0}},
    {10, 6, cell_kind::cable, "g", {4, 0.1}, {12.0, 17.0, 16.0}},
    {10, 7, cell_kind::cable, "h", {5, 0.1}, {12.0, 18.0, 16.0}},
    {10, 8, cell_kind::cable, "i", {6, 0.1}, {12.0, 19.0, 16.0}},

    {11, 0, cell_kind::cable, "abcd", {0, 0.1}, {-2.0, -5.0, 3.0}},
    {11, 1, cell_kind::cable, "cabd", {1, 0.2}, {-2.1, -5.0, 3.0}},
    {11, 2, cell_kind::cable, "cbad", {1, 0.3}, {-2.2, -5.0, 3.0}},
    {11, 3, cell_kind::cable, "acbd", {1, 1.0}, {-2.3, -5.0, 3.0}},
    {11, 4, cell_kind::cable, "bacd", {2, 0.2}, {-2.4, -5.0, 3.0}},
    {11, 5, cell_kind::cable, "bcad", {3, 0.3}, {-2.5, -5.0, 3.0}},
    {11, 6, cell_kind::cable, "dabc", {4, 0.4}, {-2.6, -5.0, 3.0}},
    {11, 7, cell_kind::cable, "dbca", {5, 0.5}, {-2.7, -5.0, 3.0}},
    {11, 8, cell_kind::cable, "dcab", {6, 0.6}, {-2.8, -5.0, 3.0}},
};
}

TEST(network_selection, all) {
    const auto s = thingify(network_selection::all(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_TRUE(s->select_connection(source, dest)); }
    }
}


TEST(network_selection, none) {
    const auto s = thingify(network_selection::none(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_FALSE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_FALSE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_FALSE(s->select_connection(source, dest)); }
    }
}

TEST(network_selection, source_cell_kind) {
    const auto s =
        thingify(network_selection::source_cell_kind(cell_kind::benchmark), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.kind == cell_kind::benchmark, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(source.kind == cell_kind::benchmark, s->select_connection(source, dest));
        }
    }
}


TEST(network_selection, destination_cell_kind) {
    const auto s =
        thingify(network_selection::destination_cell_kind(cell_kind::benchmark), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.kind == cell_kind::benchmark, s->select_destination(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(dest.kind == cell_kind::benchmark, s->select_connection(source, dest));
        }
    }
}

TEST(network_selection, source_label) {
    const auto s = thingify(network_selection::source_label({"b", "e"}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.label == "b" || site.label == "e",
            s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(
                source.label == "b" || source.label == "e", s->select_connection(source, dest));
        }
    }
}

TEST(network_selection, destination_label) {
    const auto s = thingify(network_selection::destination_label({"b", "e"}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.label == "b" || site.label == "e",
            s->select_destination(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(dest.label == "b" || dest.label == "e", s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, source_cell_vec) {
    const auto s = thingify(network_selection::source_cell({{1, 5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == 1 || src.gid == 5, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, destination_cell_vec) {
    const auto s = thingify(network_selection::destination_cell({{1, 5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_destination(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(dest.gid == 1 || dest.gid == 5, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, source_cell_range) {
    const auto s =
        thingify(network_selection::source_cell(gid_range(1, 6, 4)), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == 1 || src.gid == 5, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, destination_cell_range) {
    const auto s =
        thingify(network_selection::destination_cell(gid_range(1, 6, 4)), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_destination(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(dest.gid == 1 || dest.gid == 5, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, chain) {
    const auto s =
        thingify(network_selection::chain({{0,2,5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 0 || site.gid == 2, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 2 || site.gid == 5, s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ((src.gid == 0 && dest.gid == 2) || (src.gid == 2 && dest.gid == 5),
                s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, chain_range) {
    const auto s =
        thingify(network_selection::chain({gid_range(1,8,3)}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 4 || site.gid == 7, s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ((src.gid == 1 && dest.gid == 4) || (src.gid == 4 && dest.gid == 7),
                s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, chain_range_reverse) {
    const auto s =
        thingify(network_selection::chain_reverse({gid_range(1,8,3)}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 7 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 4 || site.gid == 1, s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ((src.gid == 7 && dest.gid == 4) || (src.gid == 4 && dest.gid == 1),
                s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, inter_cell) {
    const auto s =
        thingify(network_selection::inter_cell(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid != dest.gid, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, named) {
    network_label_dict dict;
    dict.set("mysel", network_selection::inter_cell());
    const auto s =
        thingify(network_selection::named("mysel"), dict);

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid != dest.gid, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, intersect) {
    const auto s = thingify(network_selection::intersect(network_selection::source_cell({1}),
                                network_selection::destination_cell({2})),
        network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.gid == 1, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(site.gid == 2, s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == 1 && dest.gid == 2, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, join) {
    const auto s = thingify(
        network_selection::join(network_selection::intersect(network_selection::source_cell({1}),
                                    network_selection::destination_cell({2})),
            network_selection::intersect(
                network_selection::source_cell({4}), network_selection::destination_cell({5}))),
        network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.gid == 1 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(site.gid == 2 || site.gid == 5, s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ((src.gid == 1 && dest.gid == 2) || (src.gid == 4 && dest.gid == 5),
                s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, difference) {
    const auto s =
        thingify(network_selection::difference(network_selection::source_cell({{0, 1, 2}}),
                     network_selection::source_cell({{1, 3}})),
            network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.gid == 0 || site.gid == 1 || site.gid == 2,
            s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == 0 || src.gid == 2, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, symmetric_difference) {
    const auto s = thingify(
        network_selection::symmetric_difference(
            network_selection::source_cell({{0, 1, 2}}), network_selection::source_cell({{1, 3}})),
        network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.gid == 0 || site.gid == 1 || site.gid == 2 || site.gid == 3,
            s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == 0 || src.gid == 2 || src.gid == 3, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, complement) {
    const auto s = thingify(
        network_selection::complement(network_selection::inter_cell()), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(src.gid == dest.gid, s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, random_p_1) {
    const auto s = thingify(network_selection::random(42, 1.0), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_TRUE(s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, random_p_0) {
    const auto s = thingify(network_selection::random(42, 0.0), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_FALSE(s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, random_seed) {
    const auto s1 = thingify(network_selection::random(42, 0.5), network_label_dict());
    const auto s2 = thingify(network_selection::random(4592304, 0.5), network_label_dict());

    bool all_eq = true;
    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            all_eq &= (s1->select_connection(src, dest) == s2->select_connection(src, dest));
        }
    }
    EXPECT_FALSE(all_eq);
}

TEST(network_selection, random_reproducibility) {
    const auto s = thingify(network_selection::random(42, 0.5), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, 0, cell_kind::cable, "a", {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, 1, cell_kind::cable, "b", {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, 0, cell_kind::benchmark, "c", {0, 0.0}, {20.5, -59.5, 5.0}},
    };
    std::vector<bool> ref = {1, 1, 0, 1, 1, 0, 0, 0, 0};

    std::size_t i = 0;
    for (const auto& src: sites) {
        for (const auto& dest: sites) {
            EXPECT_EQ(ref.at(i), s->select_connection(src, dest));
            ++i;
        }
    };
}

TEST(network_selection, custom) {
    auto inter_cell_func = [](const network_site_info& src, const network_site_info& dest) {
        return src.gid != dest.gid;
    };
    const auto s = thingify(network_selection::custom(inter_cell_func), network_label_dict());
    const auto s_ref = thingify(network_selection::inter_cell(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(s->select_connection(src, dest), s_ref->select_connection(src, dest));
        }
    }
}

TEST(network_selection, distance_lt) {
    const double d = 2.1;
    const auto s =
        thingify(network_selection::distance_lt(d), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(distance(src.global_location, dest.global_location) < d,
                s->select_connection(src, dest));
        }
    }
}

TEST(network_selection, distance_gt) {
    const double d = 2.1;
    const auto s =
        thingify(network_selection::distance_gt(d), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_destination(site.kind, site.gid, site.label));
    }

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_EQ(distance(src.global_location, dest.global_location) > d,
                s->select_connection(src, dest));
        }
    }
}


TEST(network_value, scalar) {
    const auto v = thingify(network_value::scalar(2.0), network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(src, dest)); }
    }
}

TEST(network_value, conversion) {
    const auto v = thingify(static_cast<network_value>(2.0), network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(src, dest)); }
    }
}

TEST(network_value, named) {
    auto dict = network_label_dict();
    dict.set("myval", network_value::scalar(2.0));
    const auto v = thingify(network_value::named("myval"), dict);

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(src, dest)); }
    }
}

TEST(network_value, distance) {
    const auto v = thingify(network_value::distance(), network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_DOUBLE_EQ(
                distance(src.global_location, dest.global_location), v->get(src, dest));
        }
    }
}

TEST(network_value, uniform_distribution) {
    const auto v = thingify(network_value::uniform_distribution(42, {-5.0, 3.0}), network_label_dict());

    double mean = 0.0;
    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { mean += v->get(src, dest); }
    }

    mean /= test_sites.size() * test_sites.size();
    EXPECT_NEAR(mean, -1.0, 1e3);
}

TEST(network_value, uniform_distribution_reproducibility) {
    const auto v = thingify(network_value::uniform_distribution(42, {-5.0, 3.0}), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, 0, cell_kind::cable, "a", {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, 1, cell_kind::cable, "b", {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, 0, cell_kind::benchmark, "c", {0, 0.0}, {20.5, -59.5, 5.0}},
    };
    std::vector<double> ref = {
        1.08007184307616289,
        0.688511962867972116,
        -2.83551807417554347,
        0.688511962867972116,
        0.824599122495063064,
        1.4676501652366376,
        -2.83551807417554347,
        1.4676501652366376,
        -4.89687864740961487,
    };

    std::size_t i = 0;
    for (const auto& src: sites) {
        for (const auto& dest: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(src, dest));
            ++i;
        }
    };
}

TEST(network_value, normal_distribution) {
    const double mean = 5.0;
    const double std_dev = 3.0;
    const auto v = thingify(network_value::normal_distribution(42, mean, std_dev), network_label_dict());

    double sample_mean = 0.0;
    double sample_dev = 0.0;
    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            const auto result = v->get(src, dest);
            sample_mean += result;
            sample_dev += (result - mean) * (result - mean);
        }
    }

    sample_mean /= test_sites.size() * test_sites.size();
    sample_dev = std::sqrt(sample_dev / (test_sites.size() * test_sites.size()));

    EXPECT_NEAR(sample_mean, mean, 1e-1);
    EXPECT_NEAR(sample_dev, std_dev, 1e-1);
}

TEST(network_value, normal_distribution_reproducibility) {
    const double mean = 5.0;
    const double std_dev = 3.0;
    const auto v = thingify(network_value::normal_distribution(42, mean, std_dev), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, 0, cell_kind::cable, "a", {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, 1, cell_kind::cable, "b", {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, 0, cell_kind::benchmark, "c", {0, 0.0}, {20.5, -59.5, 5.0}},
    };
    std::vector<double> ref = {
        9.27330832850693909,
        6.29969914563416733,
        1.81597827782531063,
        6.29969914563416733,
        8.12362497769330183,
        1.52496785710691851,
        1.81597827782531063,
        1.52496785710691851,
        1.49089022270221472,
    };

    std::size_t i = 0;
    for (const auto& src: sites) {
        for (const auto& dest: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(src, dest));
            ++i;
        }
    };
}

TEST(network_value, truncated_normal_distribution) {
    const double mean = 5.0;
    const double std_dev = 3.0;
    // symmtric upper / lower bound around mean for easy check of mean
    const double lower_bound = 1.0;
    const double upper_bound = 9.0;

    const auto v = thingify(
        network_value::truncated_normal_distribution(42, mean, std_dev, {lower_bound, upper_bound}),
        network_label_dict());

    double sample_mean = 0.0;

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            const auto result = v->get(src, dest);
            EXPECT_GT(result, lower_bound);
            EXPECT_LE(result, upper_bound);
            sample_mean += result;
        }
    }

    sample_mean /= test_sites.size() * test_sites.size();

    EXPECT_NEAR(sample_mean, mean, 1e-1);
}

TEST(network_value, truncated_normal_distribution_reproducibility) {
    const double mean = 5.0;
    const double std_dev = 3.0;

    const double lower_bound = 2.0;
    const double upper_bound = 9.0;

    const auto v = thingify(
        network_value::truncated_normal_distribution(42, mean, std_dev, {lower_bound, upper_bound}),
        network_label_dict());

    std::vector<network_site_info> sites = {
        {0, 0, cell_kind::cable, "a", {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, 1, cell_kind::cable, "b", {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, 0, cell_kind::benchmark, "c", {0, 0.0}, {20.5, -59.5, 5.0}},
    };
    std::vector<double> ref = {
        2.81708378066100629,
        4.82619033891918026,
        7.82585873628304096,
        4.82619033891918026,
        3.95914976610015401,
        5.74869285185564216,
        7.82585873628304096,
        5.74869285185564216,
        5.45028211635819293,
    };

    std::size_t i = 0;
    for (const auto& src: sites) {
        for (const auto& dest: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(src, dest));
            ++i;
        }
    };
}

TEST(network_value, custom) {
    auto func = [](const network_site_info& src, const network_site_info& dest) {
        return src.global_location.x + dest.global_location.x;
    };

    const auto v = thingify(network_value::custom(func), network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_DOUBLE_EQ(v->get(src, dest), src.global_location.x + dest.global_location.x);
        }
    }
}

TEST(network_value, add) {
    const auto v =
        thingify(network_value::add(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), 5.0); }
    }
}

TEST(network_value, sub) {
    const auto v =
        thingify(network_value::sub(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), -1.0); }
    }
}

TEST(network_value, mul) {
    const auto v =
        thingify(network_value::mul(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), 6.0); }
    }
}

TEST(network_value, div) {
    const auto v =
        thingify(network_value::div(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), 2.0 / 3.0); }
    }
}

TEST(network_value, exp) {
    const auto v =
        thingify(network_value::exp(network_value::scalar(2.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), std::exp(2.0)); }
    }
}

TEST(network_value, log) {
    const auto v = thingify(network_value::log(network_value::scalar(2.0)), network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) { EXPECT_DOUBLE_EQ(v->get(src, dest), std::log(2.0)); }
    }
}

TEST(network_value, min) {
    const auto v1 =
        thingify(network_value::min(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());
    const auto v2 =
        thingify(network_value::min(network_value::scalar(3.0), network_value::scalar(2.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_DOUBLE_EQ(v1->get(src, dest), 2.0);
            EXPECT_DOUBLE_EQ(v2->get(src, dest), 2.0);
        }
    }
}

TEST(network_value, max) {
    const auto v1 =
        thingify(network_value::max(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());
    const auto v2 =
        thingify(network_value::max(network_value::scalar(3.0), network_value::scalar(2.0)),
            network_label_dict());

    for (const auto& src: test_sites) {
        for (const auto& dest: test_sites) {
            EXPECT_DOUBLE_EQ(v1->get(src, dest), 3.0);
            EXPECT_DOUBLE_EQ(v2->get(src, dest), 3.0);
        }
    }
}

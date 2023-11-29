#include <gtest/gtest.h>

#include <arbor/network.hpp>
#include <arbor/util/hash_def.hpp>

#include "network_impl.hpp"

#include <tuple>
#include <vector>

using namespace arb;

namespace {
std::vector<network_site_info> test_sites = {
    {0, cell_kind::cable, hash_value("a"), {1, 0.5}, {0.0, 0.0, 0.0}},
    {1, cell_kind::benchmark, hash_value("b"), {0, 0.0}, {1.0, 0.0, 0.0}},
    {2, cell_kind::lif, hash_value("c"), {0, 0.0}, {2.0, 0.0, 0.0}},
    {3, cell_kind::spike_source, hash_value("d"), {0, 0.0}, {3.0, 0.0, 0.0}},
    {4, cell_kind::cable, hash_value("e"), {0, 0.2}, {4.0, 0.0, 0.0}},
    {5, cell_kind::cable, hash_value("f"), {5, 0.1}, {5.0, 0.0, 0.0}},
    {6, cell_kind::cable, hash_value("g"), {4, 0.3}, {6.0, 0.0, 0.0}},
    {7, cell_kind::cable, hash_value("h"), {0, 1.0}, {7.0, 0.0, 0.0}},
    {9, cell_kind::cable, hash_value("i"), {0, 0.1}, {12.0, 3.0, 4.0}},

    {10, cell_kind::cable, hash_value("a"), {0, 0.1}, {12.0, 15.0, 16.0}},
    {10, cell_kind::cable, hash_value("b"), {1, 0.1}, {13.0, 15.0, 16.0}},
    {10, cell_kind::cable, hash_value("c"), {1, 0.5}, {14.0, 15.0, 16.0}},
    {10, cell_kind::cable, hash_value("d"), {1, 1.0}, {15.0, 15.0, 16.0}},
    {10, cell_kind::cable, hash_value("e"), {2, 0.1}, {16.0, 15.0, 16.0}},
    {10, cell_kind::cable, hash_value("f"), {3, 0.1}, {16.0, 16.0, 16.0}},
    {10, cell_kind::cable, hash_value("g"), {4, 0.1}, {12.0, 17.0, 16.0}},
    {10, cell_kind::cable, hash_value("h"), {5, 0.1}, {12.0, 18.0, 16.0}},
    {10, cell_kind::cable, hash_value("i"), {6, 0.1}, {12.0, 19.0, 16.0}},

    {11, cell_kind::cable, hash_value("abcd"), {0, 0.1}, {-2.0, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("cabd"), {1, 0.2}, {-2.1, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("cbad"), {1, 0.3}, {-2.2, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("acbd"), {1, 1.0}, {-2.3, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("bacd"), {2, 0.2}, {-2.4, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("bcad"), {3, 0.3}, {-2.5, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("dabc"), {4, 0.4}, {-2.6, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("dbca"), {5, 0.5}, {-2.7, -5.0, 3.0}},
    {11, cell_kind::cable, hash_value("dcab"), {6, 0.6}, {-2.8, -5.0, 3.0}},
};
}

TEST(network_selection, all) {
    const auto s = thingify(network_selection::all(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_TRUE(s->select_connection(source, target)); }
    }
}

TEST(network_selection, none) {
    const auto s = thingify(network_selection::none(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_FALSE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_FALSE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_FALSE(s->select_connection(source, target)); }
    }
}

TEST(network_selection, source_cell_kind) {
    const auto s =
        thingify(network_selection::source_cell_kind(cell_kind::benchmark), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.kind == cell_kind::benchmark, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.kind == cell_kind::benchmark, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, target_cell_kind) {
    const auto s =
        thingify(network_selection::target_cell_kind(cell_kind::benchmark), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.kind == cell_kind::benchmark, s->select_target(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(target.kind == cell_kind::benchmark, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, source_label) {
    const auto s = thingify(network_selection::source_label({"b", "e"}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.label == hash_value("b") || site.label == hash_value("e"),
            s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.label == hash_value("b") || source.label == hash_value("e"),
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, target_label) {
    const auto s = thingify(network_selection::target_label({"b", "e"}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.label == hash_value("b") || site.label == hash_value("e"),
            s->select_target(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(target.label == hash_value("b") || target.label == hash_value("e"),
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, source_cell_vec) {
    const auto s = thingify(network_selection::source_cell({{1, 5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == 1 || source.gid == 5, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, target_cell_vec) {
    const auto s = thingify(network_selection::target_cell({{1, 5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_target(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(target.gid == 1 || target.gid == 5, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, source_cell_range) {
    const auto s =
        thingify(network_selection::source_cell(gid_range(1, 6, 4)), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == 1 || source.gid == 5, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, target_cell_range) {
    const auto s =
        thingify(network_selection::target_cell(gid_range(1, 6, 4)), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 5, s->select_target(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(target.gid == 1 || target.gid == 5, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, chain) {
    const auto s = thingify(network_selection::chain({{0, 2, 5}}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 0 || site.gid == 2, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 2 || site.gid == 5, s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ((source.gid == 0 && target.gid == 2) || (source.gid == 2 && target.gid == 5),
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, chain_range) {
    const auto s = thingify(network_selection::chain({gid_range(1, 8, 3)}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 4 || site.gid == 7, s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ((source.gid == 1 && target.gid == 4) || (source.gid == 4 && target.gid == 7),
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, chain_range_reverse) {
    const auto s =
        thingify(network_selection::chain_reverse({gid_range(1, 8, 3)}), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 7 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 4 || site.gid == 1, s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ((source.gid == 7 && target.gid == 4) || (source.gid == 4 && target.gid == 1),
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, inter_cell) {
    const auto s = thingify(network_selection::inter_cell(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid != target.gid, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, named) {
    network_label_dict dict;
    dict.set("mysel", network_selection::inter_cell());
    const auto s = thingify(network_selection::named("mysel"), dict);

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid != target.gid, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, intersect) {
    const auto s = thingify(network_selection::intersect(network_selection::source_cell({1}),
                                network_selection::target_cell({2})),
        network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(site.gid == 1, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(site.gid == 2, s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == 1 && target.gid == 2, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, join) {
    const auto s = thingify(
        network_selection::join(network_selection::intersect(network_selection::source_cell({1}),
                                    network_selection::target_cell({2})),
            network_selection::intersect(
                network_selection::source_cell({4}), network_selection::target_cell({5}))),
        network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_EQ(
            site.gid == 1 || site.gid == 4, s->select_source(site.kind, site.gid, site.label));
        EXPECT_EQ(
            site.gid == 2 || site.gid == 5, s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ((source.gid == 1 && target.gid == 2) || (source.gid == 4 && target.gid == 5),
                s->select_connection(source, target));
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
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == 0 || source.gid == 2, s->select_connection(source, target));
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
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == 0 || source.gid == 2 || source.gid == 3,
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, complement) {
    const auto s = thingify(
        network_selection::complement(network_selection::inter_cell()), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(source.gid == target.gid, s->select_connection(source, target));
        }
    }
}

TEST(network_selection, random_p_1) {
    const auto s = thingify(network_selection::random(42, 1.0), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_TRUE(s->select_connection(source, target)); }
    }
}

TEST(network_selection, random_p_0) {
    const auto s = thingify(network_selection::random(42, 0.0), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_FALSE(s->select_connection(source, target)); }
    }
}

TEST(network_selection, random_seed) {
    const auto s1 = thingify(network_selection::random(42, 0.5), network_label_dict());
    const auto s2 = thingify(network_selection::random(4592304, 0.5), network_label_dict());

    bool all_eq = true;
    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            all_eq &=
                (s1->select_connection(source, target) == s2->select_connection(source, target));
        }
    }
    EXPECT_FALSE(all_eq);
}

TEST(network_selection, random_reproducibility) {
    const auto s = thingify(network_selection::random(42, 0.5), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, cell_kind::cable, hash_value("a"), {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, cell_kind::cable, hash_value("b"), {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, cell_kind::benchmark, hash_value("c"), {0, 0.0}, {20.5, -59.5, 5.0}},
    };
    std::vector<bool> ref = {1, 1, 0, 1, 1, 0, 0, 0, 0};

    std::size_t i = 0;
    for (const auto& source: sites) {
        for (const auto& target: sites) {
            EXPECT_EQ(ref.at(i), s->select_connection(source, target));
            ++i;
        }
    };
}

TEST(network_selection, custom) {
    auto inter_cell_func = [](const network_site_info& source, const network_site_info& target) {
        return source.gid != target.gid;
    };
    const auto s = thingify(network_selection::custom(inter_cell_func), network_label_dict());
    const auto s_ref = thingify(network_selection::inter_cell(), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(
                s->select_connection(source, target), s_ref->select_connection(source, target));
        }
    }
}

TEST(network_selection, distance_lt) {
    const double d = 2.1;
    const auto s = thingify(network_selection::distance_lt(d), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(distance(source.global_location, target.global_location) < d,
                s->select_connection(source, target));
        }
    }
}

TEST(network_selection, distance_gt) {
    const double d = 2.1;
    const auto s = thingify(network_selection::distance_gt(d), network_label_dict());

    for (const auto& site: test_sites) {
        EXPECT_TRUE(s->select_source(site.kind, site.gid, site.label));
        EXPECT_TRUE(s->select_target(site.kind, site.gid, site.label));
    }

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_EQ(distance(source.global_location, target.global_location) > d,
                s->select_connection(source, target));
        }
    }
}

TEST(network_value, scalar) {
    const auto v = thingify(network_value::scalar(2.0), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(source, target)); }
    }
}

TEST(network_value, conversion) {
    const auto v = thingify(static_cast<network_value>(2.0), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(source, target)); }
    }
}

TEST(network_value, named) {
    auto dict = network_label_dict();
    dict.set("myval", network_value::scalar(2.0));
    const auto v = thingify(network_value::named("myval"), dict);

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(2.0, v->get(source, target)); }
    }
}

TEST(network_value, distance) {
    const auto v = thingify(network_value::distance(), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(
                distance(source.global_location, target.global_location), v->get(source, target));
        }
    }
}

TEST(network_value, uniform_distribution) {
    const auto v =
        thingify(network_value::uniform_distribution(42, {-5.0, 3.0}), network_label_dict());

    double mean = 0.0;
    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { mean += v->get(source, target); }
    }

    mean /= test_sites.size() * test_sites.size();
    EXPECT_NEAR(mean, -1.0, 1e3);
}

TEST(network_value, uniform_distribution_reproducibility) {
    const auto v =
        thingify(network_value::uniform_distribution(42, {-5.0, 3.0}), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, cell_kind::cable, hash_value("a"), {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, cell_kind::cable, hash_value("b"), {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, cell_kind::benchmark, hash_value("c"), {0, 0.0}, {20.5, -59.5, 5.0}},
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
    for (const auto& source: sites) {
        for (const auto& target: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(source, target));
            ++i;
        }
    };
}

TEST(network_value, normal_distribution) {
    const double mean = 5.0;
    const double std_dev = 3.0;
    const auto v =
        thingify(network_value::normal_distribution(42, mean, std_dev), network_label_dict());

    double sample_mean = 0.0;
    double sample_dev = 0.0;
    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            const auto result = v->get(source, target);
            sample_mean += result;
            sample_dev += (result - mean) * (result - mean);
        }
    }

    sample_mean /= test_sites.size() * test_sites.size();
    sample_dev = std::sqrt(sample_dev / (test_sites.size() * test_sites.size()));

    EXPECT_NEAR(sample_mean, mean, 1e-1);
    EXPECT_NEAR(sample_dev, std_dev, 1.5e-1);
}

TEST(network_value, normal_distribution_reproducibility) {
    const double mean = 5.0;
    const double std_dev = 3.0;
    const auto v =
        thingify(network_value::normal_distribution(42, mean, std_dev), network_label_dict());

    std::vector<network_site_info> sites = {
        {0, cell_kind::cable, hash_value("a"), {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, cell_kind::cable, hash_value("b"), {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, cell_kind::benchmark, hash_value("c"), {0, 0.0}, {20.5, -59.5, 5.0}},
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
    for (const auto& source: sites) {
        for (const auto& target: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(source, target));
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

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            const auto result = v->get(source, target);
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
        {0, cell_kind::cable, hash_value("a"), {1, 0.5}, {1.2, 2.3, 3.4}},
        {0, cell_kind::cable, hash_value("b"), {0, 0.1}, {-1.0, 0.5, 0.7}},
        {1, cell_kind::benchmark, hash_value("c"), {0, 0.0}, {20.5, -59.5, 5.0}},
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
    for (const auto& source: sites) {
        for (const auto& target: sites) {
            EXPECT_DOUBLE_EQ(ref.at(i), v->get(source, target));
            ++i;
        }
    };
}

TEST(network_value, custom) {
    auto func = [](const network_site_info& source, const network_site_info& target) {
        return source.global_location.x + target.global_location.x;
    };

    const auto v = thingify(network_value::custom(func), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(
                v->get(source, target), source.global_location.x + target.global_location.x);
        }
    }
}

TEST(network_value, add) {
    const auto v =
        thingify(network_value::add(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(v->get(source, target), 5.0); }
    }
}

TEST(network_value, sub) {
    const auto v =
        thingify(network_value::sub(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(v->get(source, target), -1.0); }
    }
}

TEST(network_value, mul) {
    const auto v =
        thingify(network_value::mul(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) { EXPECT_DOUBLE_EQ(v->get(source, target), 6.0); }
    }
}

TEST(network_value, div) {
    const auto v =
        thingify(network_value::div(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v->get(source, target), 2.0 / 3.0);
        }
    }
}

TEST(network_value, exp) {
    const auto v = thingify(network_value::exp(network_value::scalar(2.0)), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v->get(source, target), std::exp(2.0));
        }
    }
}

TEST(network_value, log) {
    const auto v = thingify(network_value::log(network_value::scalar(2.0)), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v->get(source, target), std::log(2.0));
        }
    }
}

TEST(network_value, min) {
    const auto v1 =
        thingify(network_value::min(network_value::scalar(2.0), network_value::scalar(3.0)),
            network_label_dict());
    const auto v2 =
        thingify(network_value::min(network_value::scalar(3.0), network_value::scalar(2.0)),
            network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v1->get(source, target), 2.0);
            EXPECT_DOUBLE_EQ(v2->get(source, target), 2.0);
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

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v1->get(source, target), 3.0);
            EXPECT_DOUBLE_EQ(v2->get(source, target), 3.0);
        }
    }
}

TEST(network_value, if_else) {
    const auto v1 = network_value::scalar(2.0);
    const auto v2 = network_value::scalar(3.0);

    const auto s = network_selection::inter_cell();

    const auto v = thingify(network_value::if_else(s, v1, v2), network_label_dict());

    for (const auto& source: test_sites) {
        for (const auto& target: test_sites) {
            EXPECT_DOUBLE_EQ(v->get(source, target), source.gid != target.gid ? 2.0 : 3.0);
        }
    }
}

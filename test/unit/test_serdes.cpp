#include <array>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

#include <arbor/serdes.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/simulation.hpp>
#include <arbor/recipe.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <arborio/json_serdes.hpp>

using json = nlohmann::json;
using io = arborio::json_serdes;
using serdes = arb::serdes::serializer;
using arb::serdes::write;
using arb::serdes::read;

TEST(serdes, simple) {
    auto writer = io{};
    auto serializer = serdes{writer};

    arb::serdes::write(serializer, "foo", 42.0);
    arb::serdes::write(serializer, "bar", "bing");

    auto exp = json{};
    exp["foo"] = 42.0;
    exp["bar"] = "bing";

    ASSERT_EQ(exp, writer.data);
}

TEST(serdes, containers) {
    auto writer = io{};
    auto serializer = serdes{writer};

    arb::serdes::write(serializer, "vector", std::vector<float>{1.0, 2.0, 3.0});
    arb::serdes::write(serializer, "umap_s->f", std::unordered_map<std::string, float>{{"a", 1.0}, {"b", 2.0}});
    arb::serdes::write(serializer, "map_s->f", std::map<std::string, double>{{"c", 23.0}, {"d", 42.0}});
    arb::serdes::write(serializer, "array", std::array<int, 3>{1, 2, 3});
    arb::serdes::write(serializer, "bar", "bing");

    auto exp = json{};
    exp["vector"] = std::vector<float>{1.0, 2.0, 3.0};
    exp["umap_s->f"] = std::unordered_map<std::string, float>{{"a", 1.0}, {"b", 2.0}};
    exp["map_s->f"] = std::map<std::string, double>{{"c", 23.0}, {"d", 42.0}};
    exp["array"] = std::array<int, 3>{1, 2, 3};
    exp["bar"] = "bing";

    ASSERT_EQ(exp, writer.data);
}

struct T {
    std::string a;
    double b;
    std::vector<float> vs{1.0, 2.0, 3.0};

    ARB_SERDES_ENABLE(T, a, b, vs);
};


TEST(serdes, macro) {
    auto writer = io{};
    auto serializer = serdes{writer};

    write(serializer, "t", T{"foo", 42});

    auto exp = json{};

    exp["t"]["a"] = "foo";
    exp["t"]["b"] = 42.0;
    exp["t"]["vs"] = std::vector<float>{1.0, 2.0, 3.0};

    ASSERT_EQ(exp, writer.data);
}

struct A {
    std::string s = "baz";
    std::map<int, std::vector<float>> m{{42, {1.0, 2.0}}};
    std::unordered_map<std::string, float> u;
    std::vector<int> a;
    std::vector<int> d {1, 2, 3};
    std::array<unsigned, 3> k{0,0,0};
    bool b = false;

    ARB_SERDES_ENABLE(A, s, u, m, a, k, b, d);
};


TEST(serdes, round_trip) {
    auto writer = io{};
    auto serializer = serdes{writer};

    A a;
    a.s = "bar";
    a.u = {{"a", 1.0}, {"b", 2.0}};
    a.m = {{23, {2.0, 3.0}}, {42, {4.0, 2.0}}};
    a.a = {1,2,3};
    a.k = {1,2,3};
    a.d = {4,5,6,7};
    a.b = true;

    write(serializer, "A", a);

    A b;
    read(serializer, "A", b);

    ASSERT_EQ(a.s, b.s);
    ASSERT_EQ(a.m, b.m);
    ASSERT_EQ(a.u, b.u);
    ASSERT_EQ(a.a, b.a);
    ASSERT_EQ(a.k, b.k);
    ASSERT_EQ(a.b, b.b);
    ASSERT_EQ(a.d, b.d);
}

struct serdes_recipe: public arb::recipe {
    arb::cell_size_type num_cells() const override { return num; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override {
        return {{arb::cable_probe_membrane_voltage{arb::ls::location(0, 0.5)}, 0}};
    }
    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind) const override {
        auto prop = arb::cable_cell_global_properties{};
        prop.default_parameters = arb::neuron_parameter_defaults;
        prop.default_parameters.discretization = arb::cv_policy_max_extent(1.0);
        return prop;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {
        auto decor = arb::decor{}
            .paint(arb::reg::tagged(1),
                   arb::density("hh"))
            .paint(arb::join(arb::reg::tagged(2), arb::reg::tagged(3)),
                   arb::density("pas"))
            .place(arb::mlocation{0, 0.0},
                   arb::threshold_detector{10},
                   "detector")
            .place(arb::mlocation{0, 1.0},
                   arb::synapse("exp2syn"),
                   "synapse");
        double l = 5;
        arb::segment_tree tree;
        tree.append(arb::mnpos, { -l, 0, 0, 3}, {l, 0, 0, 3}, 1);
        tree.append(0, { -l, 0, 0, 3}, {l, 0, 0, 3}, 2);

        return arb::cable_cell({tree}, decor);
    }

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        if (num <= 1) return {};
        auto src = (gid ? gid : num) - 1;
        return {{{src, "detector"},
                 {"synapse"},
                 0.5,
                 0.125}};
    }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        std::vector<arb::event_generator> res;
        if (!gid) res.push_back(arb::regular_generator({"synapse"}, 1, 0.5, 0.73));
        return {};
    }

    arb::cell_size_type num = 1;
};

static std::vector<arb_value_type>* output;

void sampler(arb::probe_metadata pm,
             std::size_t n,
             const arb::sample_record* samples) {
    auto* loc = arb::util::any_cast<const arb::mlocation*>(pm.meta);
    auto* point_info = arb::util::any_cast<const arb::cable_probe_point_info*>(pm.meta);
    loc = loc ? loc : &(point_info->loc);

    for (std::size_t i = 0; i<n; ++i) {
        auto* value = arb::util::any_cast<const double*>(samples[i].data);
        output->push_back(*value);
    }
}

TEST(serdes, single_cell) {
    double dt = 0.5;
    double T  = 5;

    // Result
    std::vector<double> result_pre;
    std::vector<double> result_v1;
    std::vector<double> result_v2;

    // Storage
    auto writer = io{};
    auto serializer = serdes{writer};

    // Set up the simulation.
    auto model = serdes_recipe{};
    auto simulation = arb::simulation{model};
    simulation.add_sampler(arb::all_probes,
                           arb::regular_schedule(dt),
                           sampler,
                           arb::sampling_policy::lax);

    // Run simulation forward && snapshot
    output = &result_pre;
    simulation.run(T, dt);
    write(serializer, "sim", simulation);

    // Then run some more, ...
    output = &result_v1;
    simulation.run(2*T, dt);

    // ... rewind ...
    read(serializer, "sim", simulation);

    // ... and run the same segment again.
    output = &result_v2;
    simulation.run(2*T, dt);

    // Now compare the two segments [T, 2T)
    ASSERT_EQ(result_v1, result_v2);
}

TEST(serdes, network) {
    double dt = 0.5;
    double T  = 5;

    // Result
    std::vector<double> result_pre;
    std::vector<double> result_v1;
    std::vector<double> result_v2;

    // Storage
    auto writer = io{};
    auto serializer = serdes{writer};

    // Set up the simulation.
    auto model = serdes_recipe{};
    model.num = 10;
    auto simulation = arb::simulation{model};
    simulation.add_sampler(arb::all_probes,
                           arb::regular_schedule(dt),
                           sampler,
                           arb::sampling_policy::lax);

    // Run simulation forward && snapshot
    output = &result_pre;
    simulation.run(T, dt);
    write(serializer, "sim", simulation);

    // Then run some more, ...
    output = &result_v1;
    simulation.run(2*T, dt);

    // ... rewind ...
    read(serializer, "sim", simulation);

    // ... and run the same segment again.
    output = &result_v2;
    simulation.run(2*T, dt);

    // Now compare the two segments [T, 2T)
    ASSERT_EQ(result_v1, result_v2);
}

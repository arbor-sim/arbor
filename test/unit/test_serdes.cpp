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

using json = nlohmann::json;

TEST(serdes, simple) {
    auto writer = arb::serdes::json_serdes{};
    auto serializer = arb::serdes::serializer{writer};

    serializer.write("foo", 42.0);
    serializer.write("bar", "bing");

    auto exp = json{};
    exp["foo"] = 42.0;
    exp["bar"] = "bing";

    ASSERT_EQ(exp, writer.data);
}

TEST(serdes, containers) {
    auto writer = arb::serdes::json_serdes{};
    auto serializer = arb::serdes::serializer{writer};

    serializer.write("vector", std::vector<float>{1.0, 2.0, 3.0});
    serializer.write("umap_s->f", std::unordered_map<std::string, float>{{"a", 1.0}, {"b", 2.0}});
    serializer.write("map_s->f", std::map<std::string, double>{{"c", 23.0}, {"d", 42.0}});
    serializer.write("array", std::array<int, 3>{1, 2, 3});
    serializer.write("bar", "bing");

    auto exp = json{};
    exp["vector"] = std::vector<float>{1.0, 2.0, 3.0};
    exp["umap_s->f"] = std::unordered_map<std::string, float>{{"a", 1.0}, {"b", 2.0}};
    exp["map_s->f"] = std::map<std::string, double>{{"c", 23.0}, {"d", 42.0}};
    exp["array"] = std::array<int, 3>{1, 2, 3};
    exp["bar"] = "bing";

    ASSERT_EQ(exp, writer.data);
}

TEST(serdes, macro) {
    struct A {
        std::string a;
        double b;
        std::vector<float> vs{1.0, 2.0, 3.0};

        ARB_SERDES_ENABLE(a, b, vs);
    };

    auto writer = arb::serdes::json_serdes{};
    auto serializer = arb::serdes::serializer{writer};

    A{"foo", 42}.serialize(serializer);

    auto exp = json{};
    exp["a"] = "foo";
    exp["b"] = 42.0;
    exp["vs"] = std::vector<float>{1.0, 2.0, 3.0};

    ASSERT_EQ(exp, writer.data);
}

TEST(serdes, round_trip) {
    auto serdes = arb::serdes::json_serdes{};
    auto serializer = arb::serdes::serializer{serdes};

    struct A {
        std::string s;
        std::map<int, float> m;
        std::unordered_map<std::string, float> u;
        std::vector<int> a;
        std::array<unsigned, 3> k{0,0,0};

        ARB_SERDES_ENABLE(s, u, m, a, k);
    };

    A a;
    a.s = "bar";
    a.u = {{"a", 1.0}, {"b", 2.0}};
    a.m = {{23, 1.0}, {42, 2.0}};
    a.a = {1,2,3};
    a.k = {1,2,3};

    serializer.write("A", a);

    A b;
    serializer.read("A", b);

    ASSERT_EQ(a.s, b.s);
    ASSERT_EQ(a.m, b.m);
    ASSERT_EQ(a.u, b.u);
    ASSERT_EQ(a.a, b.a);
    ASSERT_EQ(a.k, b.k);
}

struct the_recipe: public arb::recipe {
    the_recipe() {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        arb::segment_tree tree;
        double l = 5;
        tree.append(arb::mnpos, { -l, 0, 0, 3}, {l, 0, 0, 3}, 1);
        tree.append(0, { -l, 0, 0, 3}, {l, 0, 0, 3}, 2);
        morpho = arb::morphology{tree};
        gprop.default_parameters.discretization = arb::cv_policy_max_extent(1.0);
    }

    arb::cell_size_type num_cells() const override { return 1; }

    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override {
        arb::mlocation mid_soma = {0, 0.5};
        arb::cable_probe_membrane_voltage probe = {mid_soma};
        return {arb::probe_info{probe, 0}};
    }

    arb::cell_kind get_cell_kind(arb::cell_gid_type) const override {
        return arb::cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop;
    }

    arb::util::unique_any get_cell_description(arb::cell_gid_type) const override {

        auto soma = arb::reg::tagged(1);
        auto dend = arb::join(arb::reg::tagged(2),
                              arb::reg::tagged(3));
        auto decor = arb::decor{}
            // Add HH mechanism to soma, passive channels to dendrites.
            .paint(soma, arb::density("hh"))
            .paint(dend, arb::density("pas"))
            // Add synapse to last branch.
            .place(arb::mlocation{ morpho.num_branches()-1, 1. }, arb::synapse("exp2syn"), "synapse");

        return arb::cable_cell(morpho, decor);
    }

    arb::morphology morpho;
    arb::cable_cell_global_properties gprop;
};


TEST(serdes, simulation) {
    auto serdes = arb::serdes::json_serdes{};
    auto serializer = arb::serdes::serializer{serdes};

    {
        auto recipe = the_recipe{};
        auto simulation = arb::simulation{recipe};
        simulation.serialize(serializer);
        simulation.run(50, 0.5);
    }
    std::cerr << serdes.data.dump(4) << '\n';
}

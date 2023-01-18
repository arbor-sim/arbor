#include <array>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>


#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "serdes.hpp"

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
        std::string s{"bar"};
        std::unordered_map<std::string, float> u{{"a", 1.0}, {"b", 2.0}};
        std::map<std::string, float> m{{"a", 1.0}, {"b", 2.0}};
        std::vector<int> a{1,2,3};
        std::array<unsigned, 3> k{1,2,3};

        ARB_SERDES_ENABLE(s, u, m, a, k);
    };

    A a;

    serializer.write("A", a);

    A b;
    b.s.clear();
    b.u.clear();
    b.m.clear();
    b.a.clear();
    b.k = std::array<unsigned, 3>{0, 0, 0};

    serializer.read("A", b);

    ASSERT_EQ(a.s, b.s);
    ASSERT_EQ(a.m, b.m);
    ASSERT_EQ(a.u, b.u);
    ASSERT_EQ(a.a, b.a);
    ASSERT_EQ(a.k, b.k);
}
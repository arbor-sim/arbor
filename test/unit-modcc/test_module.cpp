#include "common.hpp"
#include "io/bulkio.hpp"
#include "module.hpp"
#include <unordered_map>

TEST(Module, open) {
    Module m(io::read_all(DATADIR "/mod_files/test0.mod"), "test0.mod");
    if (!m.buffer().size()) {
        std::cout << "skipping Module.open test because unable to open input file" << std::endl;
        return;
    }
    Lexer lexer(m.buffer());
    auto t = lexer.parse();
    while (t.type != tok::eof) {
        t = lexer.parse();
        EXPECT_NE(t.type, tok::reserved);
    }
}

TEST(Module, ion_deps) {
    Module m(io::read_all(DATADIR "/mod_files/test0.mod"), "test0.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());

    EXPECT_TRUE(m.has_ion("k"));
    auto k_dep = m.find_ion("k");
    EXPECT_TRUE(k_dep->writes_current());
    EXPECT_TRUE(k_dep->uses_current());
    EXPECT_FALSE(k_dep->uses_rev_potential());
    EXPECT_TRUE(k_dep->uses_concentration_int());
    EXPECT_FALSE(k_dep->uses_concentration_ext());
    EXPECT_TRUE(k_dep->writes_current());
    EXPECT_FALSE(k_dep->writes_concentration_int());
    EXPECT_FALSE(k_dep->writes_concentration_ext());
    EXPECT_FALSE(k_dep->writes_rev_potential());
    EXPECT_FALSE(k_dep->uses_valence());
    EXPECT_FALSE(k_dep->verifies_valence());

    EXPECT_TRUE(m.has_ion("ca"));
    auto ca_dep = m.find_ion("ca");
    EXPECT_FALSE(ca_dep->writes_current());
    EXPECT_FALSE(ca_dep->uses_current());
    EXPECT_FALSE(ca_dep->uses_rev_potential());
    EXPECT_TRUE(ca_dep->uses_concentration_int());
    EXPECT_FALSE(ca_dep->uses_concentration_ext());
    EXPECT_FALSE(ca_dep->writes_current());
    EXPECT_FALSE(ca_dep->writes_concentration_int());
    EXPECT_FALSE(ca_dep->writes_concentration_ext());
    EXPECT_FALSE(ca_dep->writes_rev_potential());
    EXPECT_FALSE(ca_dep->uses_valence());
    EXPECT_FALSE(ca_dep->verifies_valence());
}

TEST(Module, identifiers) {
    Module m(io::read_all(DATADIR "/mod_files/test0.mod"), "test0.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());

    std::unordered_map<std::string, std::pair<std::string, bool>> expected{
        {"cai", {"", false}},
        {"vhalfh", {"mV", true}},
        {"q10", {"", true}},
        {"gkbar", {"mho / cm2", false}}};

    for (const auto& parm: m.parameter_block().parameters) {
        auto it = expected.find(parm.name());
        if (it != expected.end()) {
            const auto& [units, rangep] = it->second;
            EXPECT_EQ(units, parm.unit_string());
            EXPECT_EQ(rangep, parm.has_range());
        }
    }
}

TEST(Module, linear_mechanisms) {
    for (int i = 1; i < 6; i++) {
        auto file_name = "test" + std::to_string(i) + ".mod";

        Module m(io::read_all(DATADIR "/mod_files/" + file_name), file_name);
        if (!m.buffer().size()) {
            std::cout << "skipping Module.open test because unable to open input file" << std::endl;
            return;
        }

        Parser p(m, false);
        if (!p.parse()) {
            std::cout << "problem with parsing input file" << std::endl;
            return;
        }

        m.semantic();

        if (i < 3) {
            EXPECT_TRUE(m.is_linear());
        }
        else {
            EXPECT_FALSE(m.is_linear());
        }
    }
}

TEST(Module, breakpoint) {
    // Test function call in BREAKPOINT block
    Module m(io::read_all(DATADIR "/mod_files/test8.mod"), "test8.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());

    EXPECT_TRUE(m.semantic());
}

TEST(Module, read_write_ion) {
    Module m(io::read_all(DATADIR "/mod_files/test-rw-ion.mod"), "test-rw-ion.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());
    EXPECT_TRUE(m.semantic());
}

TEST(Module, v_process) {
    Module m(io::read_all(DATADIR "/mod_files/test_v_process.mod"), "test_v_process.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());
    EXPECT_TRUE(m.semantic());
}

// Regression test in #1893 we found that the solver segfaults when handed a
// naked comparison statement.
TEST(Module, solver_bug_1893) {
    {
        Module m(io::read_all(DATADIR "/mod_files/bug-1893.mod"), "bug-1893.mod");
        EXPECT_NE(m.buffer().size(), 0u);

        Parser p(m, false);
        EXPECT_TRUE(p.parse());
        EXPECT_TRUE(m.semantic());
    }
    {
        Module m(io::read_all(DATADIR "/mod_files/bug-1893-bad.mod"), "bug-1893.mod");
        EXPECT_NE(m.buffer().size(), 0u);

        Parser p(m, false);
        EXPECT_TRUE(p.parse());
        EXPECT_FALSE(m.semantic());
    }
}

TEST(Module, stochastic_point) {
    Module m(io::read_all(DATADIR "/mod_files/test9.mod"), "test9.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());

    auto const & wnb = m.white_noise_block();

    EXPECT_EQ(wnb.parameters.size(), 2u);
    EXPECT_EQ(wnb.used.size(), 0u);

    EXPECT_TRUE(m.semantic());

    EXPECT_EQ(wnb.parameters.size(), 2u);
    EXPECT_EQ(wnb.used.size(), 1u);
}

TEST(Module, stochastic_density) {
    Module m(io::read_all(DATADIR "/mod_files/test10.mod"), "test10.mod");
    EXPECT_NE(m.buffer().size(), 0u);

    Parser p(m, false);
    EXPECT_TRUE(p.parse());

    auto const & wnb = m.white_noise_block();

    EXPECT_EQ(wnb.parameters.size(), 2u);
    EXPECT_EQ(wnb.used.size(), 0u);

    EXPECT_TRUE(m.semantic());

    EXPECT_EQ(wnb.parameters.size(), 2u);
    EXPECT_EQ(wnb.used.size(), 2u);
}

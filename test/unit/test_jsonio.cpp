#include <regex>
#include <sstream>

#include <arborio/jsonio.hpp>

#include <arbor/morph/region.hpp>
#include <arbor/util/any_visitor.hpp>

#include <sup/json_params.hpp>

#include "../gtest.h"

TEST(cable_cell_parameter_set_reader, valid) {
    {
        std::stringstream ss(
            "{\n"
            "  \"celsius\": 6.3,\n"
            "  \"Vm\": -60,\n"
            "  \"cm\": 0.01,\n"
            "  \"Ra\": 35.4,\n"
            "  \"ions\": {\n"
            "    \"ca\": {\n"
            "      \"internal-concentration\": 5e-5,\n"
            "      \"external-concentration\": 2.0,\n"
            "      \"reversal-potential\": 132.4579341637009,\n"
            "      \"method\": \"nernst\"\n"
            "    },\n"
            "    \"k\": {\n"
            "      \"internal-concentration\": 54.4,\n"
            "      \"external-concentration\": 2.5,\n"
            "      \"reversal-potential\": -77,\n"
            "      \"method\": \"constant\"\n"
            "    },\n"
            "    \"na\": {\n"
            "      \"internal-concentration\":  10,\n"
            "      \"external-concentration\": 140,\n"
            "      \"reversal-potential\": 50,\n"
            "      \"method\": \"constant\"\n"
            "    }\n"
            "  }\n"
            "}");

        auto params = arborio::load_cable_cell_parameter_set(ss);

        EXPECT_EQ(6.3 + 273.15, params.temperature_K.value());
        EXPECT_EQ(-60, params.init_membrane_potential.value());
        EXPECT_EQ(0.01, params.membrane_capacitance.value());
        EXPECT_EQ(35.4, params.axial_resistivity.value());

        EXPECT_EQ(5e-5, params.ion_data["ca"].init_int_concentration.value());
        EXPECT_EQ(2.0,  params.ion_data["ca"].init_ext_concentration.value());
        EXPECT_EQ(132.4579341637009, params.ion_data["ca"].init_reversal_potential.value());
        EXPECT_EQ("nernst/ca", params.reversal_potential_method["ca"].name());

        EXPECT_EQ(54.4, params.ion_data["k"].init_int_concentration.value());
        EXPECT_EQ(2.5,  params.ion_data["k"].init_ext_concentration.value());
        EXPECT_EQ(-77,  params.ion_data["k"].init_reversal_potential.value());
        EXPECT_FALSE(params.reversal_potential_method.count("k"));

        EXPECT_EQ(10,  params.ion_data["na"].init_int_concentration.value());
        EXPECT_EQ(140, params.ion_data["na"].init_ext_concentration.value());
        EXPECT_EQ(50,  params.ion_data["na"].init_reversal_potential.value());
        EXPECT_FALSE(params.reversal_potential_method.count("na"));
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"Vm\": -65,\n"
            "  \"cm\": 0.02,\n"
            "  \"Ra\": 100,\n"
            "  \"ions\": {\n"
            "    \"na\": {\n"
            "      \"external-concentration\": 140,\n"
            "      \"reversal-potential\": 50\n"
            "    }\n"
            "  }\n"
            "}");

        auto params = arborio::load_cable_cell_parameter_set(ss);

        EXPECT_EQ(-65,  params.init_membrane_potential.value());
        EXPECT_EQ(0.02, params.membrane_capacitance.value());
        EXPECT_EQ(100,  params.axial_resistivity.value());

        EXPECT_EQ(140, params.ion_data["na"].init_ext_concentration.value());
        EXPECT_EQ(50,  params.ion_data["na"].init_reversal_potential.value());

        EXPECT_FALSE(params.temperature_K);

        EXPECT_FALSE(params.ion_data["ca"].init_int_concentration);
        EXPECT_FALSE(params.ion_data["ca"].init_ext_concentration);
        EXPECT_FALSE(params.ion_data["ca"].init_reversal_potential);
        EXPECT_FALSE(params.reversal_potential_method.count("ca"));

        EXPECT_FALSE(params.ion_data["k"].init_int_concentration);
        EXPECT_FALSE(params.ion_data["k"].init_ext_concentration);
        EXPECT_FALSE(params.ion_data["k"].init_reversal_potential);
        EXPECT_FALSE(params.reversal_potential_method.count("k"));

        EXPECT_FALSE(params.ion_data["na"].init_int_concentration);
        EXPECT_FALSE(params.reversal_potential_method.count("na"));
    }
}

TEST(cable_cell_parameter_set_reader, invalid) {
    {
        std::stringstream ss(
                "{\n"
                "  \"Vm\": -65,\n"
                "  \"cm\": 0.02,\n"
                "  \"Ga\": 100,\n"
                "  \"ions\": {\n"
                "    \"na\": {\n"
                "      \"external-concentration\": 140,\n"
                "      \"reversal-potential\": 50\n"
                "    }\n"
                "  }\n"
                "}");

        EXPECT_THROW(arborio::load_cable_cell_parameter_set(ss), arborio::jsonio_unused_input);
    }
    {
        std::stringstream ss(
                "{\n"
                "  \"Vm\": -65,\n"
                "  \"cm\": 0.02,\n"
                "  \"Ga\": hundred,\n"
                "  \"ions\": {\n"
                "    \"na\": {\n"
                "      \"external-concentration\": 140,\n"
                "      \"reversal-potential\": 50\n"
                "    }\n"
                "  }\n"
                "}");

        EXPECT_THROW(arborio::load_cable_cell_parameter_set(ss), arborio::jsonio_json_parse_error);
    }
}

TEST(decor_reader, valid) {
    {
        std::stringstream  ss(
            "{\n"
            "  \"global\": {\n"
            "    \"celsius\": 34,\n"
            "    \"cm\": 0.01,\n"
            "    \"Ra\": 100\n"
            "  },\n"
            "  \"local\": [\n"
            "    {\n"
            "      \"region\": \"apic\",\n"
            "      \"cm\": 0.02,\n"
            "      \"ions\": {\n"
            "        \"na\": {\"reversal-potential\":  50},\n"
            "        \"k\":  {\"reversal-potential\": -85}\n"
            "      }\n"
            "    },\n"
            "    {\n"
            "      \"region\": \"dend\",\n"
            "      \"cm\": 0.02\n"
            "    }\n"
            "  ],\n"
            "  \"mechanisms\": [\n"
            "    {\n"
            "      \"region\": \"all\",\n"
            "      \"mechanism\": \"pas\"\n"
            "    },\n"
            "    {\n"
            "      \"region\": \"soma\",\n"
            "      \"mechanism\": \"CaDynamics_E2\",\n"
            "      \"parameters\": {\"gamma\": 0.000609, \"decay\": 210.485284}\n"
            "    }\n"
            "  ]\n"
            "}"
        );

        auto decor = arborio::load_decor(ss);

        EXPECT_EQ(34 + 273.15, decor.defaults().temperature_K.value());
        EXPECT_EQ(0.01, decor.defaults().membrane_capacitance.value());
        EXPECT_EQ(100, decor.defaults().axial_resistivity.value());

        EXPECT_TRUE(decor.defaults().reversal_potential_method.empty());
        EXPECT_TRUE(decor.defaults().ion_data.empty());
        EXPECT_FALSE(decor.defaults().init_membrane_potential);
        EXPECT_FALSE(decor.defaults().discretization);

        auto paintings = decor.paintings();

        for (auto& p: paintings) {
            if (to_string(p.first) == "(region \"apic\")") {
                auto apic_visitor = arb::util::overload(
                    [&](const arb::membrane_capacitance& a) {
                        EXPECT_EQ(0.02, a.value);
                    },
                    [&](const arb::init_reversal_potential& a) {
                        if (a.ion == "na") {
                            EXPECT_EQ(50, a.value);
                        }
                        else if (a.ion == "k") {
                            EXPECT_EQ(-85, a.value);
                        }
                        else {
                            FAIL() << "Unexpected ion";
                        }
                    },
                    [&](const arb::paintable&) {
                        FAIL() << "unexpected paintable";
                    });
                std::visit(apic_visitor, p.second);
            }
            else if (to_string(p.first) == "(region \"dend\")") {
                auto dend_visitor = arb::util::overload(
                    [&](const arb::membrane_capacitance& a) {
                      EXPECT_EQ(0.02, a.value);
                    },
                    [&](const arb::paintable&) {
                      FAIL() << "unexpected paintable";
                    });
                std::visit(dend_visitor, p.second);
            }
            else if (to_string(p.first) == "(region \"all\")") {
                auto all_visitor = arb::util::overload(
                    [&](const arb::mechanism_desc& a) {
                      EXPECT_EQ("pas", a.name());
                      EXPECT_TRUE(a.values().empty());
                    },
                    [&](const arb::paintable&) {
                      FAIL() << "unexpected paintable";
                    });
                std::visit(all_visitor, p.second);
            }
            else if (to_string(p.first) == "(region \"soma\")") {
                auto soma_visitor = arb::util::overload(
                    [&](const arb::mechanism_desc& a) {
                      EXPECT_EQ("CaDynamics_E2", a.name());
                      EXPECT_EQ(2u, a.values().size());
                      EXPECT_EQ(0.000609, a.values().at("gamma"));
                      EXPECT_EQ(210.485284, a.values().at("decay"));
                    },
                    [&](const arb::paintable&) {
                      FAIL() << "unexpected paintable";
                    });
                std::visit(soma_visitor, p.second);
            }
            else {
                FAIL() << "Unexpected region";
            }
        }
    }
}

TEST(decor_reader, invalid) {
    {
        std::stringstream ss(
            "{\n"
            "  \"global\": {\n"
            "    \"celsius\": 34,\n"
            "    \"cm\": cap,\n"
            "    \"Ra\": 100\n"
            "  }\n"
            "}");

        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_json_parse_error);
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"global\": {\n"
            "    \"celsius\": 34,\n"
            "    \"pm\": 22,\n"
            "    \"Ra\": 100\n"
            "  }\n"
            "}");
        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_decor_global_load_error);
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"global\": {\n"
            "    \"celsius\": 34,\n"
            "    \"cm\": 0.01,\n"
            "    \"Ra\": 100\n"
            "  },\n"
            "  \"local\": [\n"
            "    {\n"
            "      \"cm\": 0.02,\n"
            "      \"ions\": {\n"
            "        \"na\": {\"reversal-potential\":  50},\n"
            "        \"k\":  {\"reversal-potential\": -85}\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}");
        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_decor_local_missing_region);
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"local\": [\n"
            "    {\n"
            "      \"region\": \"apic\",\n"
            "      \"cm\": 0.02,\n"
            "      \"ions\": {\n"
            "        \"na\": {\"reversal-potential\":  50},\n"
            "        \"k\":  {\"method\": \"nernst\"}\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}");
        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_decor_local_revpot_mech);
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"mechanisms\": [\n"
            "    {\n"
            "      \"mechanism\": \"pas\"\n"
            "    }\n"
            "  ]\n"
            "}");
        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_decor_mech_missing_region);
    }
    {
        std::stringstream ss(
            "{\n"
            "  \"mechanisms\": [\n"
            "    {\n"
            "      \"region\": \"all\"\n"
            "    }\n"
            "  ]\n"
            "}");
        EXPECT_THROW(arborio::load_decor(ss), arborio::jsonio_decor_mech_missing_name);
    }
}

TEST(cable_cell_parameter_set_writer, valid) {
    std::string input(
        "{\n"
        "  \"celsius\": 6.0,\n"
        "  \"Vm\": -60.0,\n"
        "  \"cm\": 0.01,\n"
        "  \"Ra\": 35.4,\n"
        "  \"ions\": {\n"
        "    \"ca\": {\n"
        "      \"internal-concentration\": 5e-5,\n"
        "      \"external-concentration\": 2.0,\n"
        "      \"reversal-potential\": 132.4579341637009,\n"
        "      \"method\": \"nernst/ca\"\n"
        "    },\n"
        "    \"k\": {\n"
        "      \"internal-concentration\": 54.4,\n"
        "      \"external-concentration\": 2.5,\n"
        "      \"reversal-potential\": -77.0,\n"
        "      \"method\": \"constant\"\n"
        "    },\n"
        "    \"na\": {\n"
        "      \"internal-concentration\":  10.0,\n"
        "      \"external-concentration\": 140.0,\n"
        "      \"reversal-potential\": 50.0,\n"
        "      \"method\": \"constant\"\n"
        "    }\n"
        "  }\n"
        "}");

    std::stringstream ss_in, ss_in_cp, ss_out;
    nlohmann::json j_in, j_out;

    ss_in << input;
    auto params = arborio::load_cable_cell_parameter_set(ss_in);

    ss_in_cp << input;
    ss_in_cp >> j_in;

    arborio::store_cable_cell_parameter_set(params, ss_out);
    ss_out >> j_out;

    EXPECT_TRUE(j_in == j_out);
}

TEST(decor_writer, valid) {
    std::string input(
        "{\n"
        "  \"global\": {\n"
        "    \"celsius\": 34.0,\n"
        "    \"cm\": 0.01,\n"
        "    \"Ra\": 100.0\n"
        "  },\n"
        "  \"local\": [\n"
        "    {\n"
        "      \"region\": \"(region \\\"apic\\\")\",\n"
        "      \"cm\": 0.02,\n"
        "      \"ions\": {\n"
        "        \"na\": {\"reversal-potential\":  50.0},\n"
        "        \"k\":  {\"reversal-potential\": -85.0}\n"
        "      }\n"
        "    },\n"
        "    {\n"
        "      \"region\": \"(region \\\"dend\\\")\",\n"
        "      \"cm\": 0.02\n"
        "    },\n"
        "    {\n"
        "      \"region\": \"(region \\\"apic\\\")\",\n"
        "      \"cm\": 0.03\n"
        "    }\n"
        "  ],\n"
        "  \"mechanisms\": [\n"
        "    {\n"
        "      \"region\": \"(region \\\"all\\\")\",\n"
        "      \"mechanism\": \"pas\"\n"
        "    },\n"
        "    {\n"
        "      \"region\": \"(region \\\"soma\\\")\",\n"
        "      \"mechanism\": \"CaDynamics_E2\",\n"
        "      \"parameters\": {\"gamma\": 0.000609, \"decay\": 210.485284}\n"
        "    },\n"
        "    {\n"
        "      \"region\": \"(region \\\"all\\\")\",\n"
        "      \"mechanism\": \"hh\"\n"
        "    }\n"
        "  ]\n"
        "}"
    );

    std::stringstream ss_in, ss_in_cp, ss_out;
    nlohmann::json j_in, j_out;

    ss_in << input;
    auto decor = arborio::load_decor(ss_in);

    ss_in_cp << input;
    ss_in_cp >> j_in;

    arborio::store_decor(decor, ss_out);
    ss_out >> j_out;

    EXPECT_TRUE(j_in == j_out);
}
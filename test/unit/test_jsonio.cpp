#include <regex>
#include <sstream>

#include <arborio/jsonio.hpp>

#include <arbor/morph/region.hpp>
#include <arbor/util/any_visitor.hpp>

#include <sup/json_params.hpp>

#include "../gtest.h"

TEST(load_json, invalid) {
    {
        nlohmann::json j= R"~({ "type" : "global-parameters", "data" : {} })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_missing_field);
    }
    {
        nlohmann::json j= R"~({ "version" : 1, "data" : {} })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_missing_field);
    }
    {
        nlohmann::json j= R"~({ "version" : 1, "type" : "global-parameters" })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_missing_field);
    }
    {
        nlohmann::json j = R"~({ "version" : 2, "type" : "global-parameters", "data" : {} })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_version_error);
    }
    {
        nlohmann::json j = R"~({ "version" : 1, "type" : "global-properties", "data" : {} })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_type_error);
    }
}

TEST(load_cable_cell_parameter_set, valid) {
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "global-parameters",
            "data" :
              {
                "temperature-K": 279.45,
                "init-membrane-potential": -60,
                "membrane-capacitance": 0.01,
                "axial-resistivity": 35.4,
                "ions": {
                  "ca": {
                    "internal-concentration": 5e-5,
                    "external-concentration": 2.0,
                    "reversal-potential": 132.4579341637009,
                    "method": "nernst"
                  },
                  "k": {
                    "internal-concentration": 54.4,
                    "external-concentration": 2.5,
                    "reversal-potential": -77,
                    "method": "constant"
                  },
                  "na": {
                    "internal-concentration":  10,
                    "external-concentration": 140,
                    "reversal-potential": 50,
                    "method": "constant"
                  }
                }
              }
            })~"_json;
        auto params = std::get<arb::cable_cell_parameter_set>(arborio::load_json(j));

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
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "global-parameters",
            "data" :
              {
                "init-membrane-potential": -65,
                "membrane-capacitance": 0.02,
                "axial-resistivity": 100,
                "ions": {
                  "na": {
                    "external-concentration": 140,
                    "reversal-potential": 50
                  }
                }
              }
            })~"_json;
        auto params = std::get<arb::cable_cell_parameter_set>(arborio::load_json(j));

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

TEST(load_cable_cell_parameter_set, invalid) {
    nlohmann::json j = R"~(
        {
        "version" : 1,
        "type" : "global-parameters",
        "data" :
          {
            "init-membrane-potential": -65,
            "membrane-capacitance": 0.02,
            "Ga": 100,
            "ions": {
              "na": {
                "external-concentration": 140,
                "reversal-potential": 50
              }
            }
          }
        })~"_json;
    EXPECT_THROW(std::get<arb::cable_cell_parameter_set>(arborio::load_json(j)), arborio::jsonio_unused_input);
}

TEST(decor_reader, valid) {
    nlohmann::json j = R"~(
        {
        "version" : 1,
        "type" : "decor",
        "data" :
          {
            "global": {
              "temperature-K": 307.15,
              "membrane-capacitance": 0.01,
              "axial-resistivity": 100
            },
            "local": [
              {
                "region": "\"apic\"",
                "membrane-capacitance": 0.02,
                "ions": {
                  "na": {"reversal-potential":  50},
                  "k":  {"reversal-potential": -85}
                }
              },
              {
                "region": "\"dend\"",
                "membrane-capacitance": 0.02
              }
            ],
            "mechanisms": [
              {
                "region": "\"all\"",
                "mechanism": "pas"
              },
              {
                "region": "\"soma\"",
                "mechanism": "CaDynamics_E2",
                "parameters": {"gamma": 0.000609, "decay": 210.485284 }
              }
            ]
          }
        })~"_json;
    auto decor = std::get<arb::decor>(arborio::load_json(j));

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

TEST(decor_reader, invalid) {
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "decor",
            "data" :
              {
                "global": {
                  "temperature-K": 307.15,
                  "pm": 22,
                  "axial-resistivity": 100
                }
              }
            })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_decor_global_load_error);
    }
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "decor",
            "data" :
              {
                "global": {
                  "temperature-K": 307.15,
                  "membrane-capacitance": 0.01,
                  "axial-resistivity": 100
                },
                "local": [
                  {
                    "membrane-capacitance": 0.02,
                    "ions": {
                      "na": {"reversal-potential":  50},
                      "k":  {"reversal-potential": -85}
                    }
                  }
                ]
              }
            })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_decor_local_missing_region);
    }
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "decor",
            "data" :
              {
                "local": [
                  {
                    "region": "\"apic\"",
                    "membrane-capacitance": 0.02,
                    "ions": {
                      "na": {"reversal-potential":  50},
                      "k":  {"method": "nernst"}
                    }
                  }
                ]
              }
            })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_decor_local_revpot_mech);
    }
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "decor",
            "data" :
              {
                "mechanisms": [
                  {
                    "mechanism": "pas"
                  }
                ]
              }
            })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_decor_mech_missing_region);
    }
    {
        nlohmann::json j = R"~(
            {
            "version" : 1,
            "type" : "decor",
            "data" :
              {
                "mechanisms": [
                  {
                    "region": "\"all\""
                  }
                ]
              }
            })~"_json;
        EXPECT_THROW(arborio::load_json(j), arborio::jsonio_decor_mech_missing_name);
    }
}

TEST(cable_cell_parameter_set_writer, valid) {
    nlohmann::json j_in = R"~(
        {
        "version" : 1,
        "type" : "global-parameters",
        "data" :
          {
            "temperature-K": 279.15,
            "init-membrane-potential": -60.0,
            "membrane-capacitance": 0.01,
            "axial-resistivity": 35.4,
            "ions": {
              "ca": {
                "internal-concentration": 5e-5,
                "external-concentration": 2.0,
                "reversal-potential": 132.4579341637009,
                "method": "nernst/ca"
              },
              "k": {
                "internal-concentration": 54.4,
                "external-concentration": 2.5,
                "reversal-potential": -77.0,
                "method": "constant"
              },
              "na": {
                "internal-concentration":  10.0,
                "external-concentration": 140.0,
                "reversal-potential": 50.0,
                "method": "constant"
              }
            }
          }
        })~"_json;

    auto params = std::get<arb::cable_cell_parameter_set>(arborio::load_json(j_in));
    auto j_out = arborio::write_json(params);
    EXPECT_TRUE(j_in == j_out);
}

TEST(decor_writer, valid) {
    nlohmann::json j_in = R"~(
        {
        "version" : 1,
        "type" : "decor",
        "data" :
          {
            "global": {
              "temperature-K": 307.15,
              "membrane-capacitance": 0.01,
              "axial-resistivity": 100.0
            },
            "local": [
              {
                "region": "(region \"apic\")",
                "membrane-capacitance": 0.02,
                "ions": {
                  "na": {"reversal-potential":  50.0},
                  "k":  {"reversal-potential": -85.0}
                }
              },
              {
                "region": "(region \"dend\")",
                "membrane-capacitance": 0.02
              },
              {
                "region": "(region \"apic\")",
                "membrane-capacitance": 0.03
              }
            ],
            "mechanisms": [
              {
                "region": "(region \"all\")",
                "mechanism": "pas"
              },
              {
                "region": "(region \"soma\")",
                "mechanism": "CaDynamics_E2",
                "parameters": {"gamma": 0.000609, "decay": 210.485284}
              },
              {
                "region": "(region \"all\")",
                "mechanism": "hh"
              }
            ]
          }
        })~"_json;

    auto decor = std::get<arb::decor>(arborio::load_json(j_in));
    auto j_out = arborio::write_json(decor);
    EXPECT_TRUE(j_in == j_out);
}
#include <sstream>

#include <arborio/jsonio.hpp>

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

        EXPECT_THROW(arborio::load_cable_cell_parameter_set(ss), arborio::jsonio_error);
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

        EXPECT_THROW(arborio::load_cable_cell_parameter_set(ss), arborio::jsonio_error);
    }
}

TEST(decor_reader, valid) {}

TEST(decor_reader, invalid) {}

TEST(cable_cell_parameter_set_writer, valid) {}

TEST(cable_cell_parameter_set_writer, invalid) {}

TEST(decor_writer, valid) {}

TEST(decor_writer, invalid) {}
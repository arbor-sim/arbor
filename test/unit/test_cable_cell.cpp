#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "../common_cells.hpp"

#include <arbor/util/hash_def.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/units.hpp>
#include <arbor/cable_cell_param.hpp>

#include <arborio/label_parse.hpp>

using namespace arb;
using namespace arborio::literals;

namespace U = arb::units;

TEST(cable_cell, lid_ranges) {

    // Create a morphology with two branches:
    //   * branch 0 is a single segment soma followed by a single segment dendrite.
    //   * branch 1 is an axon attached to the root.
    segment_tree tree;
    tree.append(mnpos, {0, 0,  0, 10}, {0, 0,  10, 10}, 1);
    tree.append(0,     {0, 0, 10,  1}, {0, 0, 100,  1}, 3);
    tree.append(mnpos, {0, 0,  0,  2}, {0, 0, -20,  2}, 2);

    arb::morphology morph(tree);

    label_dict dict;
    dict.set("term", "(terminal)"_ls);

    decor decorations;

    mlocation_list empty_sites = {};
    mlocation_list three_sites = {{0, 0.1}, {1, 0.2}, {1, 0.7}};

    // Place synapses and threshold detectors in interleaved order.
    // Note: there are 2 terminal points.
    decorations.place("term"_lab, synapse("expsyn"), "t0");
    decorations.place("term"_lab, synapse("expsyn"), "t1");
    decorations.place("term"_lab, threshold_detector{-10*arb::units::mV}, "s0");
    decorations.place(empty_sites, synapse("expsyn"), "t2");
    decorations.place("term"_lab, threshold_detector{-20*arb::units::mV}, "s1");
    decorations.place(three_sites, synapse("expsyn"), "t3");
    decorations.place("term"_lab, synapse("exp2syn"), "t3");

    cable_cell cell(morph, decorations, dict);

    // Get the assigned lid ranges for each placement
    const auto& src_ranges = cell.detector_ranges();
    const auto& tgt_ranges = cell.synapse_ranges();

    EXPECT_EQ(1u, tgt_ranges.count(hash_value("t0")));
    EXPECT_EQ(1u, tgt_ranges.count(hash_value("t1")));
    EXPECT_EQ(1u, src_ranges.count(hash_value("s0")));
    EXPECT_EQ(1u, tgt_ranges.count(hash_value("t2")));
    EXPECT_EQ(1u, src_ranges.count(hash_value("s1")));
    EXPECT_EQ(2u, tgt_ranges.count(hash_value("t3")));

    auto r1 = tgt_ranges.equal_range(hash_value("t0")).first->second;
    auto r2 = tgt_ranges.equal_range(hash_value("t1")).first->second;
    auto r3 = src_ranges.equal_range(hash_value("s0")).first->second;
    auto r4 = tgt_ranges.equal_range(hash_value("t2")).first->second;
    auto r5 = src_ranges.equal_range(hash_value("s1")).first->second;

    auto r6_range = tgt_ranges.equal_range(hash_value("t3"));
    auto r6_0 = r6_range.first;
    auto r6_1 = std::next(r6_range.first);
    if (r6_0->second.begin != 4u) {
        std::swap(r6_0, r6_1);
    }

    EXPECT_EQ(r1.begin, 0u); EXPECT_EQ(r1.end, 2u);
    EXPECT_EQ(r2.begin, 2u); EXPECT_EQ(r2.end, 4u);
    EXPECT_EQ(r3.begin, 0u); EXPECT_EQ(r3.end, 2u);
    EXPECT_EQ(r4.begin, 4u); EXPECT_EQ(r4.end, 4u);
    EXPECT_EQ(r5.begin, 2u); EXPECT_EQ(r5.end, 4u);
    EXPECT_EQ(r6_0->second.begin, 4u); EXPECT_EQ(r6_0->second.end, 7u);
    EXPECT_EQ(r6_1->second.begin, 7u); EXPECT_EQ(r6_1->second.end, 9u);
}

namespace {

constexpr double nan_d = std::numeric_limits<double>::quiet_NaN();
constexpr double inf_d = std::numeric_limits<double>::infinity();

// Call `f` and require a std::domain_error whose message mentions `expect`.
//
// Each parameter type below has two independent rejections: a sign constraint
// in the constructor body and a finiteness constraint in U::unit_of, which also
// catches a quantity of the wrong dimension. Both are std::domain_error, so a
// test for one would be satisfied by the other unless it says which fired.
template <typename F>
void expect_domain_error(const F& f, const char* expect) {
    try {
        f();
    }
    catch (const std::domain_error& e) {
        std::string msg{e.what()};
        EXPECT_NE(std::string::npos, msg.find(expect)) << "expected \"" << expect << "\" in: " << msg;
        return;
    }
    catch (const std::exception& e) {
        ADD_FAILURE() << "expected std::domain_error, got another type: " << e.what();
        return;
    }
    ADD_FAILURE() << "expected std::domain_error mentioning \"" << expect << "\"; nothing thrown";
}

} // namespace

// These are value types: the constraints are enforced in the constructors, so
// neither a morphology nor a discretization is needed to test them.
TEST(cable_cell, strictly_positive_parameters) {
    // Zero is rejected, not just negative: a 0 K cell makes the thermal voltage
    // zero and a zero capacitance or resistivity has no physical meaning.
    expect_domain_error([]{ temperature{0*U::Kelvin}; },              "positive");
    expect_domain_error([]{ temperature{-1*U::Kelvin}; },             "positive");
    expect_domain_error([]{ axial_resistivity{0*U::Ohm*U::cm}; },     "strictly positive");
    expect_domain_error([]{ axial_resistivity{-35.4*U::Ohm*U::cm}; }, "strictly positive");
    expect_domain_error([]{ membrane_capacitance{0*U::F/U::m2}; },    "strictly positive");
    expect_domain_error([]{ membrane_capacitance{-0.01*U::F/U::m2}; },"strictly positive");

    // Non-finite values are rejected by the conversion, with a different message.
    expect_domain_error([]{ temperature{nan_d*U::Kelvin}; },            "finite");
    expect_domain_error([]{ temperature{inf_d*U::Kelvin}; },            "finite");
    expect_domain_error([]{ axial_resistivity{nan_d*U::Ohm*U::cm}; },   "finite");
    expect_domain_error([]{ membrane_capacitance{inf_d*U::F/U::m2}; },  "finite");

    // A quantity of the wrong dimension converts to NaN and is caught the same way.
    expect_domain_error([]{ membrane_capacitance{1*U::mV}; },         "finite");
    expect_domain_error([]{ axial_resistivity{1*U::mV}; },            "finite");

    // Legal values, converted into the unit the type stores.
    EXPECT_NEAR(300.0, temperature{300*U::Kelvin}.value, 1e-9);
    EXPECT_NEAR(303.15, temperature{30*U::Celsius}.value, 1e-9);
    EXPECT_NEAR(0.01, membrane_capacitance{1*U::uF/U::cm2}.value, 1e-15);
    EXPECT_NEAR(1000.0, axial_resistivity{1*U::kOhm*U::cm}.value, 1e-9);

    // There is no magnitude bound, only a sign and a finiteness check, so a
    // positive-but-vanishing capacitance is still accepted here. What happens
    // to it further down is a separate question (see test_fvm_layout).
    EXPECT_LT(0.0, membrane_capacitance{1e-300*U::F/U::m2}.value);
}

TEST(cable_cell, concentration_and_diffusivity_constraints) {
    expect_domain_error([]{ init_int_concentration{"na", -1*U::mM}; }, "positive");
    expect_domain_error([]{ init_ext_concentration{"na", -1*U::mM}; }, "positive");
    expect_domain_error([]{ init_int_concentration{"na", nan_d*U::mM}; },"finite");
    expect_domain_error([]{ init_ext_concentration{"na", inf_d*U::mM}; },"finite");

    // Conversion to the stored unit, and the ion label, are unaffected.
    auto iconc = init_int_concentration{"na", 0.001*U::M};
    auto econc = init_ext_concentration{"ca", 2*U::mM};
    EXPECT_NEAR(1.0, iconc.value, 1e-12);
    EXPECT_NEAR(2.0, econc.value, 1e-12);
    EXPECT_EQ("na", iconc.ion);

    auto d_zero = ion_diffusivity{"na", 0*U::m2/U::s};
    auto d_real = ion_diffusivity{"ca", 5e-10*U::m2/U::s};
    EXPECT_DOUBLE_EQ(0.0, d_zero.value);
    EXPECT_NEAR(5e-10, d_real.value, 1e-25);
    expect_domain_error([]{ ion_diffusivity{"na", -1*U::m2/U::s}; },    "positive");
    expect_domain_error([]{ ion_diffusivity{"na", nan_d*U::m2/U::s}; }, "finite");
    expect_domain_error([]{ ion_diffusivity{"na", 1*U::mV}; },          "finite");
}

// Potentials carry no sign constraint — -65 mV has to stay constructible — and
// neither does a threshold. Only finiteness is policed.
TEST(cable_cell, potentials_require_only_finiteness) {
    EXPECT_NEAR(-65.0, threshold_detector{-65*U::mV}.threshold, 1e-12);
    EXPECT_NEAR(0.0, threshold_detector{0*U::mV}.threshold, 0.0);
    EXPECT_NEAR(-65.0, init_membrane_potential{-65*U::mV}.value, 1e-12);
    auto erevk = init_reversal_potential{"k", -77*U::mV};
    EXPECT_NEAR(-77.0, erevk.value, 1e-12);

    expect_domain_error([]{ threshold_detector{nan_d*U::mV}; },                "finite");
    expect_domain_error([]{ threshold_detector::from_raw_millivolts(nan_d); }, "finite");
    expect_domain_error([]{ threshold_detector{inf_d*U::mV}; },                "finite");
    expect_domain_error([]{ init_membrane_potential{inf_d*U::mV}; },           "finite");
    expect_domain_error([]{ init_reversal_potential{"ca", nan_d*U::mV}; },     "finite");
    expect_domain_error([]{ threshold_detector{1*U::mM}; },                    "mV");
}

// The default constructor means "unset", not "illegal": these values are what a
// cell or global parameter set holds when the user never supplied one, and they
// must stay constructible.
TEST(cable_cell, default_constructed_parameters_are_unset) {
    EXPECT_TRUE(std::isnan(membrane_capacitance{}.value));
    EXPECT_TRUE(std::isnan(axial_resistivity{}.value));
    EXPECT_TRUE(std::isnan(temperature{}.value));
    EXPECT_TRUE(std::isnan(init_int_concentration{}.value));
    EXPECT_TRUE(std::isnan(init_ext_concentration{}.value));
    EXPECT_TRUE(std::isnan(init_reversal_potential{}.value));
    EXPECT_TRUE(std::isnan(init_membrane_potential{}.value));

    EXPECT_EQ("", init_int_concentration{}.ion);
    EXPECT_EQ("", ion_diffusivity{}.ion);
    EXPECT_NO_THROW(membrane_capacitance{});
    EXPECT_NO_THROW(init_int_concentration{});
}


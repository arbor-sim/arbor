#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mechinfo.hpp>

#include "common.hpp"

using namespace std::string_literals;
using namespace arb;

// Set up a small system of mechanisms and backends for testing,
// comprising:
//
// * Two mechanisms: burble and fleeb.
//
// * Two backends: foo and bar.
//
// * Three implementations of fleeb:
//   - Two for the foo backend: fleeb_foo and special_fleeb_foo.
//   - One for the bar backend: fleeb_bar.
//
// * One implementation of burble for the bar backend, with
//   a mismatched fingerprint: burble_bar.

// Mechanism info:

using field_kind = mechanism_field_spec::field_kind;

mechanism_info burble_info = {
    {{"quux",  {field_kind::global, "nA", 2.3,   0, 10.}},
     {"xyzzy", {field_kind::global, "mV", 5.1, -20, 20.}}},
    {},
    {},
    {{"x", {}}},
    "burbleprint"
};

mechanism_info fleeb_info = {
    {{"plugh", {field_kind::global, "C",   2.3,  0, 10.}},
     {"norf",  {field_kind::global, "mGy", 0.1,  0, 5000.}}},
    {},
    {},
    {{"a", {}}, {"b", {}}, {"c", {}}, {"d", {}}},
    "fleebprint"
};

// Backend classes:

template <typename B>
struct common_impl: concrete_mechanism<B> {
    void instantiate(fvm_size_type id, typename B::shared_state& state, const mechanism_overrides& o, const mechanism_layout& l) override {
        width_ = l.cv.size();
        // Write mechanism global values to shared state to test instatiation call and catalogue global
        // variable overrides.
        for (auto& kv: o.globals) {
            state.overrides.insert(kv);
        }

        for (auto& ion: mech_ions) {
            if (o.ion_rebind.count(ion)) {
                ion_bindings_[ion] = state.ions.at(o.ion_rebind.at(ion));
            }
            else {
                ion_bindings_[ion] = state.ions.at(ion);
            }
        }
    }

    std::size_t memory() const override { return 10u; }
    std::size_t size() const override { return width_; }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& vs) override {}

    void initialize() override {}
    void nrn_state() override {}
    void nrn_current() override {}
    void deliver_events() override {}
    void write_ions() override {}

    std::size_t width_ = 0;

    std::vector<std::string> mech_ions;

    std::unordered_map<std::string, std::string> ion_bindings_;
};

template <typename B>
std::string ion_binding(const std::unique_ptr<concrete_mechanism<B>>& mech, const char* ion) {
    const common_impl<B>& impl = dynamic_cast<const common_impl<B>&>(*mech.get());
    return impl.ion_bindings_.count(ion)? impl.ion_bindings_.at(ion): "";
}


struct foo_backend {
    struct shared_state {
        std::unordered_map<std::string, fvm_value_type> overrides;
        std::unordered_map<std::string, std::string> ions = {
            { "a", "foo_ion_a" },
            { "b", "foo_ion_b" },
            { "c", "foo_ion_c" },
            { "d", "foo_ion_d" },
            { "e", "foo_ion_e" },
            { "f", "foo_ion_f" }
        };
    };
};

using foo_mechanism = common_impl<foo_backend>;

struct bar_backend {
    struct shared_state {
        std::unordered_map<std::string, fvm_value_type> overrides;
        std::unordered_map<std::string, std::string> ions = {
            { "a", "bar_ion_a" },
            { "b", "bar_ion_b" },
            { "c", "bar_ion_c" },
            { "d", "bar_ion_d" },
            { "e", "bar_ion_e" },
            { "f", "bar_ion_f" }
        };
    };
};

using bar_mechanism = common_impl<bar_backend>;

// Fleeb implementations:

struct fleeb_foo: foo_mechanism {
    fleeb_foo() {
        this->mech_ions = {"a", "b", "c", "d"};
    }

    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fleebprint";
        return hash;
    }

    std::string internal_name() const override { return "fleeb"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new fleeb_foo()); }
};

struct special_fleeb_foo: foo_mechanism {
    special_fleeb_foo() {
        this->mech_ions = {"a", "b", "c", "d"};
    }

    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fleebprint";
        return hash;
    }

    std::string internal_name() const override { return "special fleeb"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new special_fleeb_foo()); }
};

struct fleeb_bar: bar_mechanism {
    fleeb_bar() {
        this->mech_ions = {"a", "b", "c", "d"};
    }

    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fleebprint";
        return hash;
    }

    std::string internal_name() const override { return "fleeb"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new fleeb_bar()); }
};

// Burble implementation:

struct burble_bar: bar_mechanism {
    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fnord";
        return hash;
    }

    std::string internal_name() const override { return "burble"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new burble_bar()); }
};

// Implementation register helper:

template <typename B, typename M>
std::unique_ptr<concrete_mechanism<B>> make_mech() {
    return std::unique_ptr<concrete_mechanism<B>>(new M());
}

// Mechinfo equality test:

namespace arb {
static bool operator==(const mechanism_field_spec& a, const mechanism_field_spec& b) {
    return a.kind==b.kind && a.units==b.units && a.default_value==b.default_value && a.lower_bound==b.lower_bound && a.upper_bound==b.upper_bound;
}

static bool operator==(const ion_dependency& a, const ion_dependency& b) {
    return a.write_concentration_int==b.write_concentration_int && a.write_concentration_ext==b.write_concentration_ext;
}

static bool operator==(const mechanism_info& a, const mechanism_info& b) {
    return a.globals==b.globals && a.parameters==b.parameters && a.state==b.state && a.ions==b.ions && a.fingerprint==b.fingerprint;
}
}

mechanism_catalogue build_fake_catalogue() {
    mechanism_catalogue cat;

    cat.add("fleeb", fleeb_info);
    cat.add("burble", burble_info);

    // Add derived versions with global overrides:

    cat.derive("fleeb1",        "fleeb",         {{"plugh", 1.0}}, {{"a", "b"}, {"b", "a"}});
    cat.derive("special_fleeb", "fleeb",         {{"plugh", 2.0}});
    cat.derive("fleeb2",        "special_fleeb", {{"norf", 11.0}});
    cat.derive("fleeb3",        "fleeb1",        {}, {{"b", "c"}, {"c", "b"}});
    cat.derive("bleeble",       "burble",        {{"quux",  10.}, {"xyzzy", -20.}});

    // Attach implementations:

    cat.register_implementation<bar_backend>("fleeb", make_mech<bar_backend, fleeb_bar>());
    cat.register_implementation<foo_backend>("fleeb", make_mech<foo_backend, fleeb_foo>());
    cat.register_implementation<foo_backend>("special_fleeb", make_mech<foo_backend, special_fleeb_foo>());

    return cat;
}

TEST(mechcat, fingerprint) {
    auto cat = build_fake_catalogue();

    EXPECT_EQ("fleebprint", cat.fingerprint("fleeb"));
    EXPECT_EQ("fleebprint", cat.fingerprint("special_fleeb"));
    EXPECT_EQ("burbleprint", cat.fingerprint("burble"));
    EXPECT_EQ("burbleprint", cat.fingerprint("bleeble"));

    EXPECT_THROW(cat.register_implementation<bar_backend>("burble", make_mech<bar_backend, burble_bar>()),
        arb::fingerprint_mismatch);
}

TEST(mechcat, derived_info) {
    auto cat = build_fake_catalogue();

    EXPECT_EQ(fleeb_info,  cat["fleeb"]);
    EXPECT_EQ(burble_info, cat["burble"]);

    mechanism_info expected_special_fleeb = fleeb_info;
    expected_special_fleeb.globals["plugh"].default_value = 2.0;
    EXPECT_EQ(expected_special_fleeb, cat["special_fleeb"]);

    mechanism_info expected_fleeb2 = fleeb_info;
    expected_fleeb2.globals["plugh"].default_value = 2.0;
    expected_fleeb2.globals["norf"].default_value = 11.0;
    EXPECT_EQ(expected_fleeb2, cat["fleeb2"]);
}

TEST(mechcat, queries) {
    auto cat = build_fake_catalogue();

    EXPECT_TRUE(cat.has("fleeb"));
    EXPECT_TRUE(cat.has("special_fleeb"));
    EXPECT_TRUE(cat.has("fleeb1"));
    EXPECT_TRUE(cat.has("fleeb2"));
    EXPECT_TRUE(cat.has("burble"));
    EXPECT_TRUE(cat.has("bleeble"));
    EXPECT_FALSE(cat.has("corge"));

    EXPECT_TRUE(cat.is_derived("special_fleeb"));
    EXPECT_TRUE(cat.is_derived("fleeb1"));
    EXPECT_TRUE(cat.is_derived("fleeb2"));
    EXPECT_TRUE(cat.is_derived("bleeble"));
    EXPECT_FALSE(cat.is_derived("fleeb"));
    EXPECT_FALSE(cat.is_derived("burble"));
}

TEST(mechcat, remove) {
    auto cat = build_fake_catalogue();

    cat.remove("special_fleeb");
    EXPECT_TRUE(cat.has("fleeb"));
    EXPECT_TRUE(cat.has("fleeb1"));
    EXPECT_FALSE(cat.has("special_fleeb"));
    EXPECT_FALSE(cat.has("fleeb2")); // fleeb2 derived from special_fleeb.
}

TEST(mechcat, instance) {
    auto cat = build_fake_catalogue();

    EXPECT_THROW(cat.instance<bar_backend>("burble"), arb::no_such_implementation);

    // All fleebs on the bar backend have the same implementation:

    auto fleeb_bar_inst = cat.instance<bar_backend>("fleeb");
    auto fleeb1_bar_inst = cat.instance<bar_backend>("fleeb1");
    auto special_fleeb_bar_inst = cat.instance<bar_backend>("special_fleeb");
    auto fleeb2_bar_inst = cat.instance<bar_backend>("fleeb2");

    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb1_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*special_fleeb_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb2_bar_inst.mech.get()));

    EXPECT_EQ("fleeb"s, fleeb2_bar_inst.mech->internal_name());

    // special_fleeb and fleeb2 (deriving from special_fleeb) have a specialized
    // implementation:

    auto fleeb_foo_inst = cat.instance<foo_backend>("fleeb");
    auto fleeb1_foo_inst = cat.instance<foo_backend>("fleeb1");
    auto special_fleeb_foo_inst = cat.instance<foo_backend>("special_fleeb");
    auto fleeb2_foo_inst = cat.instance<foo_backend>("fleeb2");

    EXPECT_EQ(typeid(fleeb_foo), typeid(*fleeb_foo_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_foo), typeid(*fleeb1_foo_inst.mech.get()));
    EXPECT_EQ(typeid(special_fleeb_foo), typeid(*special_fleeb_foo_inst.mech.get()));
    EXPECT_EQ(typeid(special_fleeb_foo), typeid(*fleeb2_foo_inst.mech.get()));

    EXPECT_EQ("fleeb"s, fleeb1_foo_inst.mech->internal_name());
    EXPECT_EQ("special fleeb"s, fleeb2_foo_inst.mech->internal_name());
}

TEST(mechcat, instantiate) {
    // Note: instantiating a mechanism doesn't normally have that mechanism
    // write its specialized global variables to shared state, but we do in
    // these tests for testing purposes.

    mechanism_layout layout = {{0u, 1u, 2u}, {1., 2., 1.}, {1u, 1u, 1u}};
    bar_backend::shared_state bar_state;

    auto cat = build_fake_catalogue();

    auto fleeb = cat.instance<bar_backend>("fleeb");
    fleeb.mech->instantiate(0, bar_state, fleeb.overrides, layout);
    EXPECT_TRUE(bar_state.overrides.empty());

    bar_state.overrides.clear();
    auto fleeb2 = cat.instance<bar_backend>("fleeb2");
    fleeb2.mech->instantiate(0, bar_state, fleeb2.overrides, layout);
    EXPECT_EQ(2.0,  bar_state.overrides.at("plugh"));
    EXPECT_EQ(11.0, bar_state.overrides.at("norf"));

    // Check ion rebinding:
    // fleeb1 should have ions 'a' and 'b' swapped;
    // fleeb2 should swap 'b' and 'c' relative to fleeb1, so that
    // 'b' maps to the state 'c' ion, 'c' maps to the state 'a' ion,
    // and 'a' maps to the state 'b' ion.

    EXPECT_EQ("bar_ion_a", ion_binding(fleeb.mech, "a"));
    EXPECT_EQ("bar_ion_b", ion_binding(fleeb.mech, "b"));
    EXPECT_EQ("bar_ion_c", ion_binding(fleeb.mech, "c"));
    EXPECT_EQ("bar_ion_d", ion_binding(fleeb.mech, "d"));

    auto fleeb3 = cat.instance<bar_backend>("fleeb3");
    fleeb3.mech->instantiate(0, bar_state, fleeb3.overrides, layout);

    foo_backend::shared_state foo_state;
    auto fleeb1 = cat.instance<foo_backend>("fleeb1");
    fleeb1.mech->instantiate(0, foo_state, fleeb1.overrides, layout);

    EXPECT_EQ("foo_ion_b", ion_binding(fleeb1.mech, "a"));
    EXPECT_EQ("foo_ion_a", ion_binding(fleeb1.mech, "b"));
    EXPECT_EQ("foo_ion_c", ion_binding(fleeb1.mech, "c"));
    EXPECT_EQ("foo_ion_d", ion_binding(fleeb1.mech, "d"));

    EXPECT_EQ("bar_ion_c", ion_binding(fleeb3.mech, "a"));
    EXPECT_EQ("bar_ion_a", ion_binding(fleeb3.mech, "b"));
    EXPECT_EQ("bar_ion_b", ion_binding(fleeb3.mech, "c"));
    EXPECT_EQ("bar_ion_d", ion_binding(fleeb3.mech, "d"));
}

TEST(mechcat, bad_ion_rename) {
    auto cat = build_fake_catalogue();

    // missing ion
    EXPECT_THROW(cat.derive("ono", "fleeb", {}, {{"nosuchion", "x"}}), invalid_ion_remap);

    // two ions with the same name, the original 'b', and the renamed 'a'
    EXPECT_THROW(cat.derive("alas", "fleeb", {}, {{"a", "b"}}), invalid_ion_remap);
}

TEST(mechcat, implicit_deriv) {
    auto cat = build_fake_catalogue();

    mechanism_info burble_derived_info = cat["burble/quux=3,xyzzy=4"];
    EXPECT_EQ(3, burble_derived_info.globals["quux"].default_value);
    EXPECT_EQ(4, burble_derived_info.globals["xyzzy"].default_value);

    // If the mechanism is already in the catalogue though, don't make a new derivation.
    cat.derive("fleeb/plugh=5", "fleeb", {{"plugh", 7.0}}, {});
    mechanism_info deceptive = cat["fleeb/plugh=5"];
    EXPECT_EQ(7, deceptive.globals["plugh"].default_value);

    // Check ion rebinds, too.
    mechanism_info fleeb_derived_info = cat["fleeb/plugh=2,a=foo,b=bar"];
    EXPECT_EQ(2, fleeb_derived_info.globals["plugh"].default_value);
    EXPECT_FALSE(fleeb_derived_info.ions.count("a"));
    EXPECT_FALSE(fleeb_derived_info.ions.count("b"));
    EXPECT_TRUE(fleeb_derived_info.ions.count("foo"));
    EXPECT_TRUE(fleeb_derived_info.ions.count("bar"));

    // If only one ion, don't need to give lhs in reassignment.
    mechanism_info bleeble_derived_info = cat["bleeble/fish,quux=9"];
    EXPECT_EQ(9, bleeble_derived_info.globals["quux"].default_value);
    EXPECT_EQ(-20, bleeble_derived_info.globals["xyzzy"].default_value);
    EXPECT_EQ(1u, bleeble_derived_info.ions.size());
    EXPECT_TRUE(bleeble_derived_info.ions.count("fish"));

    // Can't omit lhs if there is more than one ion though.
    EXPECT_THROW(cat["fleeb/fish"], invalid_ion_remap);

    // Implicitly derived mechanisms should inherit implementations.
    auto fleeb2 = cat.instance<foo_backend>("fleeb2");
    auto fleeb2_derived = cat.instance<foo_backend>("fleeb2/plugh=4.5");
    EXPECT_EQ("special fleeb", fleeb2.mech->internal_name());
    EXPECT_EQ("special fleeb", fleeb2_derived.mech->internal_name());
    EXPECT_EQ(4.5, fleeb2_derived.overrides.globals.at("plugh"));

    // Requesting an implicitly derived instance with improper parameters should throw.
    EXPECT_THROW(cat.instance<foo_backend>("fleeb2/fidget=7"), no_such_parameter);

    // Testing for implicit derivation though should not throw.
    EXPECT_TRUE(cat.has("fleeb2/plugh=7"));
    EXPECT_FALSE(cat.has("fleeb2/fidget=7"));
    EXPECT_TRUE(cat.is_derived("fleeb2/plugh=7"));
    EXPECT_FALSE(cat.is_derived("fleeb2/fidget=7"));
}

TEST(mechcat, copy) {
    auto cat = build_fake_catalogue();
    mechanism_catalogue cat2 = cat;

    EXPECT_EQ(cat["fleeb2"], cat2["fleeb2"]);

    auto fleeb2_inst = cat.instance<foo_backend>("fleeb2");
    auto fleeb2_inst2 = cat2.instance<foo_backend>("fleeb2");

    EXPECT_EQ(typeid(*fleeb2_inst.mech.get()), typeid(*fleeb2_inst2.mech.get()));
}



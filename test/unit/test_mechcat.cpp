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
    {},
    "burbleprint"
};

mechanism_info fleeb_info = {
    {{"plugh", {field_kind::global, "C",   2.3,  0, 10.}},
     {"norf",  {field_kind::global, "mGy", 0.1,  0, 5000.}}},
    {},
    {},
    {},
    "fleebprint"
};

// Backend classes:

template <typename B>
struct common_impl: concrete_mechanism<B> {
    void instantiate(fvm_size_type id, typename B::shared_state& state, const mechanism::layout& l) override {
        width_ = l.cv.size();
        // Write mechanism global values to shared state to test instatiation call and catalogue global
        // variable overrides.
        for (auto& kv: overrides_) {
            state.overrides.insert(kv);
        }
    }

    std::size_t memory() const override { return 10u; }
    std::size_t size() const override { return width_; }

    void set_parameter(const std::string& key, const std::vector<fvm_value_type>& vs) override {}

    void set_global(const std::string& key, fvm_value_type v) override {
        overrides_[key] = v;
    }

    void nrn_init() override {}
    void nrn_state() override {}
    void nrn_current() override {}
    void deliver_events() override {}
    void write_ions() override {}

    std::unordered_map<std::string, fvm_value_type> overrides_;
    std::size_t width_ = 0;
};

struct foo_backend {
    struct shared_state {
        std::unordered_map<std::string, fvm_value_type> overrides;
    };
};

using foo_mechanism = common_impl<foo_backend>;

struct bar_backend {
    struct shared_state {
        std::unordered_map<std::string, fvm_value_type> overrides;
    };
};

using bar_mechanism = common_impl<bar_backend>;

// Fleeb implementations:

struct fleeb_foo: foo_mechanism {
    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fleebprint";
        return hash;
    }

    std::string internal_name() const override { return "fleeb"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new fleeb_foo()); }
};

struct special_fleeb_foo: foo_mechanism {
    const mechanism_fingerprint& fingerprint() const override {
        static mechanism_fingerprint hash = "fleebprint";
        return hash;
    }

    std::string internal_name() const override { return "special fleeb"; }
    mechanismKind kind() const override { return mechanismKind::density; }
    mechanism_ptr clone() const override { return mechanism_ptr(new special_fleeb_foo()); }
};

struct fleeb_bar: bar_mechanism {
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

    cat.derive("fleeb1",        "fleeb",         {{"plugh", 1.0}});
    cat.derive("special_fleeb", "fleeb",         {{"plugh", 2.0}});
    cat.derive("fleeb2",        "special_fleeb", {{"norf", 11.0}});
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

    auto fleeb_bar_mech = cat.instance<bar_backend>("fleeb");
    auto fleeb1_bar_mech = cat.instance<bar_backend>("fleeb1");
    auto special_fleeb_bar_mech = cat.instance<bar_backend>("special_fleeb");
    auto fleeb2_bar_mech = cat.instance<bar_backend>("fleeb2");

    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb_bar_mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb1_bar_mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*special_fleeb_bar_mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb2_bar_mech.get()));

    EXPECT_EQ("fleeb"s, fleeb2_bar_mech->internal_name());

    // special_fleeb and fleeb2 (deriving from special_fleeb) have a specialized
    // implementation:

    auto fleeb_foo_mech = cat.instance<foo_backend>("fleeb");
    auto fleeb1_foo_mech = cat.instance<foo_backend>("fleeb1");
    auto special_fleeb_foo_mech = cat.instance<foo_backend>("special_fleeb");
    auto fleeb2_foo_mech = cat.instance<foo_backend>("fleeb2");

    EXPECT_EQ(typeid(fleeb_foo), typeid(*fleeb_foo_mech.get()));
    EXPECT_EQ(typeid(fleeb_foo), typeid(*fleeb1_foo_mech.get()));
    EXPECT_EQ(typeid(special_fleeb_foo), typeid(*special_fleeb_foo_mech.get()));
    EXPECT_EQ(typeid(special_fleeb_foo), typeid(*fleeb2_foo_mech.get()));

    EXPECT_EQ("fleeb"s, fleeb1_foo_mech->internal_name());
    EXPECT_EQ("special fleeb"s, fleeb2_foo_mech->internal_name());
}

TEST(mechcat, instantiate) {
    // Note: instantiating a mechanism doesn't normally have that mechanism
    // write its specialized global variables to shared state, but we do in
    // these tests for testing purposes.

    mechanism::layout layout = {{0u, 1u, 2u}, {1., 2., 1.}};
    bar_backend::shared_state bar_state;

    auto cat = build_fake_catalogue();

    cat.instance<bar_backend>("fleeb")->instantiate(0, bar_state, layout);
    EXPECT_TRUE(bar_state.overrides.empty());

    bar_state.overrides.clear();
    cat.instance<bar_backend>("fleeb2")->instantiate(0, bar_state, layout);
    EXPECT_EQ(2.0,  bar_state.overrides.at("plugh"));
    EXPECT_EQ(11.0, bar_state.overrides.at("norf"));
}

TEST(mechcat, copy) {
    auto cat = build_fake_catalogue();
    mechanism_catalogue cat2 = cat;

    EXPECT_EQ(cat["fleeb2"], cat2["fleeb2"]);

    auto fleeb2_instance = cat.instance<foo_backend>("fleeb2");
    auto fleeb2_instance2 = cat2.instance<foo_backend>("fleeb2");

    EXPECT_EQ(typeid(*fleeb2_instance.get()), typeid(*fleeb2_instance.get()));
}



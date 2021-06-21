#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mechinfo.hpp>

#include "common.hpp"

#ifndef LIBDIR
#warning "LIBDIR not set; defaulting to '.'"
#define LIBDIR "."
#endif

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

struct test_mechanism: mechanism {
    std::vector<std::string> mech_ions;
    std::unordered_map<std::string, std::string> ion_bindings_;
};

// Backend classes:
struct test_backend {
    using iarray = std::vector<fvm_index_type>;
    using array  = std::vector<fvm_value_type>;

    test_backend(const std::unordered_map<std::string, std::string>& ions_): shared_{ions_} {}

    struct shared_state {
        shared_state(const std::unordered_map<std::string, std::string>& ions_): ions{ions_} {}

        void instantiate(mechanism& m_, fvm_size_type id, const mechanism_overrides& o, const mechanism_layout& l) {
            test_mechanism& m = dynamic_cast<test_mechanism&>(m_);
            m.ppack_.width = l.cv.size();
            // Write mechanism global values to shared state to test instatiation call and catalogue global
            // variable overrides.
            for (auto& kv: o.globals) {
                overrides.insert(kv);
            }

            for (auto& ion: m.mech_ions) {
                if (o.ion_rebind.count(ion)) {
                    m.ion_bindings_[ion] = ions.at(o.ion_rebind.at(ion));
                } else {
                    m.ion_bindings_[ion] = ions.at(ion);
                }
            }
        }

        std::unordered_map<std::string, fvm_value_type> overrides;
        std::unordered_map<std::string, std::string> ions;
    };

    shared_state shared_;

    struct deliverable_event_stream {
        struct state {
            void* ev_data;
            int* begin_offset;
            int* end_offset;
            int n;
        };
        state& marked_events() { return state_; }
        state state_;
    };

    static void multiply_in_place(fvm_value_type* s, const fvm_index_type* p, int n) {}
};

struct foo_backend: test_backend {
    static constexpr arb_backend_kind kind = 42;
    foo_backend(): test_backend{{{ "a", "foo_ion_a" },
                                 { "b", "foo_ion_b" },
                                 { "c", "foo_ion_c" },
                                 { "d", "foo_ion_d" },
                                 { "e", "foo_ion_e" },
                                 { "f", "foo_ion_f" }}} {}
};

struct bar_backend: test_backend {
    static constexpr arb_backend_kind kind = 23;
    bar_backend(): test_backend{{{ "a", "bar_ion_a" },
                                 { "b", "bar_ion_b" },
                                 { "c", "bar_ion_c" },
                                 { "d", "bar_ion_d" },
                                 { "e", "bar_ion_e" },
                                 { "f", "bar_ion_f" }}} {}
};

std::string ion_binding(const mechanism& mech, const char* ion) {
    const test_mechanism& tm = dynamic_cast<const test_mechanism&>(mech);
    return tm.ion_bindings_.count(ion)? tm.ion_bindings_.at(ion): "";
}

// Fleeb implementations:

struct fleeb_foo: test_mechanism {
    fleeb_foo() {
        this->mech_ions = {"a", "b", "c", "d"};
        mech_.fingerprint = "fleebprint";
        mech_.name        = "fleeb";
        mech_.kind        = arb_mechanism_kind_density;
        iface_.backend    = foo_backend::kind;
    }
};

struct special_fleeb_foo: test_mechanism {
    special_fleeb_foo() {
        this->mech_ions   = {"a", "b", "c", "d"};
        mech_.fingerprint = "fleebprint";
        mech_.name        = "special fleeb";
        mech_.kind        = arb_mechanism_kind_density;
        iface_.backend    = foo_backend::kind;
    }
};

struct fleeb_bar: test_mechanism {
    fleeb_bar() {
        this->mech_ions = {"a", "b", "c", "d"};
        mech_.fingerprint = "fleebprint";
        mech_.name        = "fleeb";
        mech_.kind        = arb_mechanism_kind_density;
        iface_.backend    = bar_backend::kind;
    }
};

// Burble implementation:

struct burble_bar: test_mechanism {
    burble_bar() {
        mech_.fingerprint = "fnord";
        mech_.name        = "burble";
        mech_.kind        = arb_mechanism_kind_density;
        iface_.backend    = bar_backend::kind;
    }
};

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

    cat.register_implementation("fleeb",         std::unique_ptr<mechanism>{new fleeb_bar});
    cat.register_implementation("fleeb",         std::unique_ptr<mechanism>{new fleeb_foo});
    cat.register_implementation("special_fleeb", std::unique_ptr<mechanism>{new special_fleeb_foo});

    return cat;
}

TEST(mechcat, fingerprint) {
    auto cat = build_fake_catalogue();

    EXPECT_EQ("fleebprint", cat.fingerprint("fleeb"));
    EXPECT_EQ("fleebprint", cat.fingerprint("special_fleeb"));
    EXPECT_EQ("burbleprint", cat.fingerprint("burble"));
    EXPECT_EQ("burbleprint", cat.fingerprint("bleeble"));

    EXPECT_THROW(cat.register_implementation("burble", std::unique_ptr<mechanism>{new burble_bar}),
        arb::fingerprint_mismatch);
}

TEST(mechcat, names) {
    // All names are caught; covers `add' and `derive'
    {
        auto cat = build_fake_catalogue();
        auto names  = cat.mechanism_names();
        auto expect = std::vector<std::string>{"bleeble", "burble", "fleeb", "fleeb1", "fleeb2", "fleeb3", "special_fleeb"};
        std::sort(names.begin(), names.end());
        EXPECT_EQ(names, expect);
    }

    // Deriving names does not add to catalogue
    {
        auto cat = build_fake_catalogue();
        auto info   = cat["burble/quux=3,xyzzy=4"];
        auto names  = cat.mechanism_names();
        auto expect = std::vector<std::string>{"bleeble", "burble", "fleeb", "fleeb1", "fleeb2", "fleeb3", "special_fleeb"};
        std::sort(names.begin(), names.end());
        EXPECT_EQ(names, expect);
    }

    // Deleting a mechanism removes it and all derived from it.
    {
        auto cat = build_fake_catalogue();
        cat.remove("fleeb");
        auto names  = cat.mechanism_names();
        auto expect = std::vector<std::string>{"bleeble", "burble"};
        std::sort(names.begin(), names.end());
        EXPECT_EQ(names, expect);
    }

    // Empty means empty.
    {
        auto cat = build_fake_catalogue();
        cat.remove("fleeb");
        cat.remove("burble");
        auto names  = cat.mechanism_names();
        auto expect = std::vector<std::string>{};
        std::sort(names.begin(), names.end());
        EXPECT_EQ(names, expect);
    }
}

#ifdef USE_DYNAMIC_CATALOGUES
TEST(mechcat, loading) {
    EXPECT_THROW(load_catalogue(LIBDIR "/does-not-exist-catalogue.so"), bad_catalogue_error);
    EXPECT_THROW(load_catalogue(LIBDIR "/libarbor.a"), bad_catalogue_error);
    const mechanism_catalogue* cat = nullptr;
    EXPECT_NO_THROW(cat = &load_catalogue(LIBDIR "/dummy-catalogue.so"));
    ASSERT_NE(cat, nullptr);
    EXPECT_EQ(std::vector<std::string>{"dummy"}, cat->mechanism_names());

}
#endif

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

    EXPECT_THROW(cat.instance(bar_backend::kind, "burble"), arb::no_such_implementation);

    // All fleebs on the bar backend have the same implementation:

    auto fleeb_bar_inst = cat.instance(bar_backend::kind, "fleeb");
    auto fleeb1_bar_inst = cat.instance(bar_backend::kind, "fleeb1");
    auto special_fleeb_bar_inst = cat.instance(bar_backend::kind, "special_fleeb");
    auto fleeb2_bar_inst = cat.instance(bar_backend::kind, "fleeb2");

    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb1_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*special_fleeb_bar_inst.mech.get()));
    EXPECT_EQ(typeid(fleeb_bar), typeid(*fleeb2_bar_inst.mech.get()));

    EXPECT_EQ("fleeb"s, fleeb2_bar_inst.mech->internal_name());

    // special_fleeb and fleeb2 (deriving from special_fleeb) have a specialized
    // implementation:

    auto fleeb_foo_inst = cat.instance(foo_backend::kind, "fleeb");
    auto fleeb1_foo_inst = cat.instance(foo_backend::kind, "fleeb1");
    auto special_fleeb_foo_inst = cat.instance(foo_backend::kind, "special_fleeb");
    auto fleeb2_foo_inst = cat.instance(foo_backend::kind,"f leeb2");

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
    bar_backend bar;

    auto cat = build_fake_catalogue();

    auto fleeb = cat.instance(bar_backend::kind, "fleeb");
    bar.shared_.instantiate(*fleeb.mech, 0, fleeb.overrides, layout);
    EXPECT_TRUE(bar.shared_.overrides.empty());

    bar.shared_.overrides.clear();
    auto fleeb2 = cat.instance(bar_backend::kind, "fleeb2");
    bar.shared_.instantiate(*fleeb2.mech, 0, fleeb2.overrides, layout);
    EXPECT_EQ(2.0,  bar.shared_.overrides.at("plugh"));
    EXPECT_EQ(11.0, bar.shared_.overrides.at("norf"));

    // Check ion rebinding:
    // fleeb1 should have ions 'a' and 'b' swapped;
    // fleeb2 should swap 'b' and 'c' relative to fleeb1, so that
    // 'b' maps to the state 'c' ion, 'c' maps to the state 'a' ion,
    // and 'a' maps to the state 'b' ion.

    EXPECT_EQ("bar_ion_a", ion_binding(*fleeb.mech, "a"));
    EXPECT_EQ("bar_ion_b", ion_binding(*fleeb.mech, "b"));
    EXPECT_EQ("bar_ion_c", ion_binding(*fleeb.mech, "c"));
    EXPECT_EQ("bar_ion_d", ion_binding(*fleeb.mech, "d"));

    auto fleeb3 = cat.instance(bar_backend::kind, "fleeb3");
    bar.shared_.instantiate(*fleeb3.mech, 0, fleeb3.overrides, layout);

    foo_backend foo;
    auto fleeb1 = cat.instance(foo_backend::kind, "fleeb1");
    foo.shared_.instantiate(*fleeb1.mech, 0, fleeb1.overrides, layout);

    EXPECT_EQ("foo_ion_b", ion_binding(*fleeb1.mech, "a"));
    EXPECT_EQ("foo_ion_a", ion_binding(*fleeb1.mech, "b"));
    EXPECT_EQ("foo_ion_c", ion_binding(*fleeb1.mech, "c"));
    EXPECT_EQ("foo_ion_d", ion_binding(*fleeb1.mech, "d"));

    EXPECT_EQ("bar_ion_c", ion_binding(*fleeb3.mech, "a"));
    EXPECT_EQ("bar_ion_a", ion_binding(*fleeb3.mech, "b"));
    EXPECT_EQ("bar_ion_b", ion_binding(*fleeb3.mech, "c"));
    EXPECT_EQ("bar_ion_d", ion_binding(*fleeb3.mech, "d"));
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
    auto fleeb2 = cat.instance(foo_backend::kind, "fleeb2");
    auto fleeb2_derived = cat.instance(foo_backend::kind, "fleeb2/plugh=4.5");
    EXPECT_EQ("special fleeb", fleeb2.mech->internal_name());
    EXPECT_EQ("special fleeb", fleeb2_derived.mech->internal_name());
    EXPECT_EQ(4.5, fleeb2_derived.overrides.globals.at("plugh"));

    // Requesting an implicitly derived instance with improper parameters should throw.
    EXPECT_THROW(cat.instance(foo_backend::kind, "fleeb2/fidget=7"), no_such_parameter);

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

    auto fleeb2_inst = cat.instance(foo_backend::kind, "fleeb2");
    auto fleeb2_inst2 = cat2.instance(foo_backend::kind, "fleeb2");

    EXPECT_EQ(typeid(*fleeb2_inst.mech.get()), typeid(*fleeb2_inst2.mech.get()));
}

TEST(mechcat, import) {
    auto cat = build_fake_catalogue();
    mechanism_catalogue cat2;
    cat2.import(cat, "fake_");

    EXPECT_TRUE(cat.has("fleeb2"));
    EXPECT_FALSE(cat.has("fake_fleeb2"));

    EXPECT_TRUE(cat2.has("fake_fleeb2"));
    EXPECT_FALSE(cat2.has("fleeb2"));

    EXPECT_EQ(cat["fleeb2"], cat2["fake_fleeb2"]);

    auto fleeb2_inst  = cat.instance(foo_backend::kind, "fleeb2");
    auto fleeb2_inst2 = cat2.instance(foo_backend::kind, "fake_fleeb2");

    EXPECT_EQ(typeid(*fleeb2_inst.mech.get()), typeid(*fleeb2_inst2.mech.get()));
}

TEST(mechcat, import_collisions) {
    {
        auto cat = build_fake_catalogue();

        mechanism_catalogue cat2;
        EXPECT_NO_THROW(cat2.import(cat, "prefix:")); // Should have no collisions.
        EXPECT_NO_THROW(cat.import(cat2, "prefix:")); // Should have no collisions here either.

        // cat should have both original entries and copies with 'prefix:prefix:' prefixed.
        ASSERT_TRUE(cat.has("fleeb2"));
        ASSERT_TRUE(cat.has("prefix:prefix:fleeb2"));
    }

    // We should throw if there any collisions between base or derived mechanism
    // names between the catalogues. If the import fails, the catalogue should
    // remain unchanged.
    {
        // Collision between two base mechanisms.
        {
            auto cat = build_fake_catalogue();

            mechanism_catalogue other;
            other.add("fleeb", burble_info); // Note different mechanism info!

            EXPECT_THROW(cat.import(other, ""), arb::duplicate_mechanism);
            ASSERT_EQ(cat["fleeb"], fleeb_info);
        }

        // Collision derived vs base.
        {
            auto cat = build_fake_catalogue();

            mechanism_catalogue other;
            other.add("fleeb2", burble_info);

            auto fleeb2_info = cat["fleeb2"];
            EXPECT_THROW(cat.import(other, ""), arb::duplicate_mechanism);
            EXPECT_EQ(cat["fleeb2"], fleeb2_info);
        }

        // Collision base vs derived.
        {
            auto cat = build_fake_catalogue();

            mechanism_catalogue other;
            other.add("zonkers", fleeb_info);
            other.derive("fleeb", "zonkers", {{"plugh", 8.}});
            ASSERT_FALSE(other["fleeb"]==fleeb_info);

            ASSERT_FALSE(cat.has("zonkers"));
            EXPECT_THROW(cat.import(other, ""), arb::duplicate_mechanism);
            EXPECT_EQ(cat["fleeb"], fleeb_info);
            EXPECT_FALSE(cat.has("zonkers"));
        }

        // Collision derived vs derived.
        {
            auto cat = build_fake_catalogue();

            mechanism_catalogue other;
            other.add("zonkers", fleeb_info);
            other.derive("fleeb2", "zonkers", {{"plugh", 8.}});

            auto fleeb2_info = cat["fleeb2"];
            ASSERT_FALSE(other["fleeb2"]==fleeb2_info);

            ASSERT_FALSE(cat.has("zonkers"));
            EXPECT_THROW(cat.import(other, ""), arb::duplicate_mechanism);
            EXPECT_EQ(cat["fleeb2"], fleeb2_info);
            EXPECT_FALSE(cat.has("zonkers"));
        }
    }
}

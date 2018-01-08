#pragma once

#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include <mechinfo.hpp>
#include <mechanism.hpp>

// Mechanism catalogue maintains:
//
// 1. Collection of mechanism metadata indexed by name.
//
// 2. A further hierarchy of 'derived' mechanisms, that allow specialization of
//    global parameters and implementations.
//
// 3. A map taking mechanism names x back-end class -> mechanism implementation
//    prototype object.
//
// Implementations for a backend `B` are represented by a pointer to a 
// `concrete_mechanism<B>` object.
//
// References to mechanism_info and mechanism_fingerprint objects are invalidated
// after any modification to the catalogue.
//
// There is in addition a global default mechanism catalogue object that is
// populated with any builtin mechanisms and mechanisms generated from
// module files included with arbor.

namespace arb {

class mechanism_catalogue {
public:
    using value_type = double;

    mechanism_catalogue() = default;
    mechanism_catalogue(mechanism_catalogue&& other) = default;
    mechanism_catalogue& operator=(mechanism_catalogue&& other) = default;

    // Copying a catalogue requires cloning the prototypes.
    mechanism_catalogue(const mechanism_catalogue& other) {
        copy_impl(other);
    }

    mechanism_catalogue& operator=(const mechanism_catalogue& other) {
        copy_impl(other);
        return *this;
    }

    void add(const std::string& name, mechanism_info info);

    bool has(const std::string& name) const {
        return info_map_.count(name) || is_derived(name);
    }

    bool is_derived(const std::string& name) const {
        return derived_map_.count(name);
    }

    // Read-only access to mechanism info.
    const mechanism_info& operator[](const std::string& name) const;

    // Read-only access to mechanism fingerprint.
    const mechanism_fingerprint& fingerprint(const std::string& name) const;

    // Construct a schema for a mechanism derived from an existing entry,
    // with a sequence of overrides for global scalar parameter settings.
    void derive(const std::string& name, const std::string& parent, const std::vector<std::pair<std::string, double>>& global_params);

    // Remove mechanism from catalogue, together with any derived.
    void remove(const std::string& name);

    // Clone the implementation associated with name (search derivation hierarchy starting from
    // most derived) and set global parameters according to derivations.
    template <typename B>
    std::unique_ptr<concrete_mechanism<B>> instance(const std::string& name) const {
        mechanism_ptr mech = instance_impl(std::type_index(typeid(B)), name);

        return std::unique_ptr<concrete_mechanism<B>>(dynamic_cast<concrete_mechanism<B>*>(mech.release()));
    }

    // Associate a concrete (prototype) mechanism for a given back-end B with a (possibly derived)
    // mechanism name.
    template <typename B>
    void register_implementation(const std::string& name, std::unique_ptr<concrete_mechanism<B>> proto) {
        mechanism_ptr generic_proto = mechanism_ptr(proto.release());
        register_impl(std::type_index(typeid(B)), name, std::move(generic_proto));
    }

private:
    using mechanism_info_ptr = std::unique_ptr<mechanism_info>;

    template <typename V>
    using string_map = std::unordered_map<std::string, V>;

    // Schemata for (un-derived) mechanisms.
    string_map<mechanism_info_ptr> info_map_;

    struct derivation {
        std::string parent;
        string_map<value_type> globals;  // global overrides relative to parent
        mechanism_info_ptr derived_info;
    };

    // Parent and global setting values for derived mechanisms.
    string_map<derivation> derived_map_;

    // Prototype register, keyed on mechanism name, then backend type (index).
    string_map<std::unordered_map<std::type_index, mechanism_ptr>> impl_map_;

    // Concrete-type erased helper methods.
    mechanism_ptr instance_impl(std::type_index, const std::string&) const;
    void register_impl(std::type_index, const std::string&, mechanism_ptr);

    // Perform copy and prototype clone from other catalogue (overwrites all entries).
    void copy_impl(const mechanism_catalogue&);
};

// Reference to global default mechanism catalogue.

const mechanism_catalogue& global_default_catalogue();


} // namespace arb

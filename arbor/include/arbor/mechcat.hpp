#pragma once

#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include <arbor/mechinfo.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/optional.hpp>

// Mechanism catalogue maintains:
//
// 1. Collection of mechanism metadata indexed by name.
//
// 2. A further hierarchy of 'derived' mechanisms, that allow specialization of
//    global parameters, ion bindings, and implementations.
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
//
// When a mechanism name of the form "mech/param=value,..." is requested, if the
// mechanism of that name does not already exist in the catalogue, it will be
// implicitly derived from an existing mechanism "mech", with global parameters
// and ion bindings overridden by the supplied assignments that follow the slash.
// If the mechanism in question has a single ion dependence, then that ion name
// can be omitted in the assignments; "mech/oldion=newion" will make the same
// derived mechanism as simply "mech/newion".

namespace arb {

// catalogue_state comprises the private implementation of mechanism_catalogue.
struct catalogue_state;

class mechanism_catalogue {
public:
    using value_type = double;

    mechanism_catalogue();
    mechanism_catalogue(mechanism_catalogue&& other);
    mechanism_catalogue& operator=(mechanism_catalogue&& other);

    mechanism_catalogue(const mechanism_catalogue& other);
    mechanism_catalogue& operator=(const mechanism_catalogue& other);

    void add(const std::string& name, mechanism_info info);

    // Has `name` been added, derived, or can it be implicitly derived?
    bool has(const std::string& name) const;

    // Is `name` a derived mechanism or can it be implicitly derived?
    bool is_derived(const std::string& name) const;

    // Read-only access to mechanism info.
    mechanism_info operator[](const std::string& name) const;

    // Read-only access to mechanism fingerprint.
    const mechanism_fingerprint& fingerprint(const std::string& name) const;

    // Construct a schema for a mechanism derived from an existing entry,
    // with a sequence of overrides for global scalar parameter settings
    // and a set of ion renamings.
    void derive(const std::string& name, const std::string& parent,
                const std::vector<std::pair<std::string, double>>& global_params,
                const std::vector<std::pair<std::string, std::string>>& ion_remap = {});

    // Remove mechanism from catalogue, together with any derivations of it.
    void remove(const std::string& name);

    // Clone the implementation associated with name (search derivation hierarchy starting from
    // most derived) and return together with any global overrides.
    template <typename B>
    struct cat_instance {
        std::unique_ptr<concrete_mechanism<B>> mech;
        mechanism_overrides overrides;
    };

    template <typename B>
    cat_instance<B> instance(const std::string& name) const {
        auto mech = instance_impl(std::type_index(typeid(B)), name);

        return cat_instance<B>{
            std::unique_ptr<concrete_mechanism<B>>(dynamic_cast<concrete_mechanism<B>*>(mech.first.release())),
            std::move(mech.second)
        };
    }

    // Associate a concrete (prototype) mechanism for a given back-end B with a (possibly derived)
    // mechanism name.
    template <typename B>
    void register_implementation(const std::string& name, std::unique_ptr<concrete_mechanism<B>> proto) {
        mechanism_ptr generic_proto = mechanism_ptr(proto.release());
        register_impl(std::type_index(typeid(B)), name, std::move(generic_proto));
    }

    ~mechanism_catalogue();

private:
    std::unique_ptr<catalogue_state> state_;

    std::pair<mechanism_ptr, mechanism_overrides> instance_impl(std::type_index, const std::string&) const;
    void register_impl(std::type_index, const std::string&, mechanism_ptr);
};


// Reference to global default mechanism catalogue.

const mechanism_catalogue& global_default_catalogue();

} // namespace arb

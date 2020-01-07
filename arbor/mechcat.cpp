#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/either.hpp>

#include "util/maputil.hpp"

/* Notes on implementation:
 *
 * The catalogue maintains the following data:
 *
 * 1. impl_map_
 *
 *    This contains the mapping between mechanism names and concrete mechanisms
 *    for a specific backend that have been registered with
 *    register_impl().
 *
 *    It is a two-level map, first indexed by name, and then by the back-end
 *    type (using std::type_index).
 *
 * 2. info_map_
 *
 *    Contains the mechanism_info metadata for a mechanism, as given to the
 *    catalogue via the add() method.
 *
 * 3. derived_map_
 *
 *    A 'derived' mechanism is one that shares the same metadata schema as its
 *    parent, but with possible overrides to its global scalar parameters and
 *    to the bindings of its ion names.
 *
 *    The derived_map_ entry for a given mechanism gives: the parent mechanism
 *    from which it is derived (which might also be a derived mechanism); the
 *    set of changes to global parameters relative to its parent; the set of
 *    ion rebindings relative to its parent; and an updated copy of the
 *    mechanism_info metadata that reflects those changes.
 *
 * The derived_map_ and info_map_ together constitute a forest: info_map_ has
 * an entry for each un-derived mechanism in the catalogue, while for any
 * derived mechanism, the parent field in derived_map_ provides the parent in
 * the derivation tree, or a root mechanism which is catalogued in info_map_.
 *
 * When an instance of the mechanism is requested from the catalogue, the
 * instance_impl_() function walks up the derivation tree to find the first
 * entry which has an associated implementation. It then accumulates the set of
 * global parameter and ion overrides that need to be applied, starting from
 * the top-most (least-derived) ancestor and working down to the requested derived
 * mechanism.
 *
 * The private implementation class catalogue_state does not throw any (catalogue
 * related) exceptions, but instead propagates errors via util::either to the
 * mechanism_catalogue methods for handling.
 */

namespace arb {

using util::value_by_key;
using util::optional;
using util::nullopt;

using std::make_unique;
using std::make_exception_ptr;

using mechanism_info_ptr = std::unique_ptr<mechanism_info>;

template <typename V>
using string_map = std::unordered_map<std::string, V>;

template <typename T>
struct hopefully_typemap {
    using type = util::either<T, std::exception_ptr>;
};

template <>
struct hopefully_typemap<void> {
    struct placeholder_type {};
    using type = util::either<placeholder_type, std::exception_ptr>;
};

template <typename T>
using hopefully = typename hopefully_typemap<T>::type;

// Convert hopefully<T> to T or throw.

template <typename T>
const T& value(const util::either<T, std::exception_ptr>& x) {
    if (!x) {
        std::rethrow_exception(x.second());
    }
    return x.first();
}

template <typename T>
T value(util::either<T, std::exception_ptr>&& x) {
    if (!x) {
        std::rethrow_exception(x.second());
    }
    return std::move(x.first());
}

void value(const hopefully<void>& x) {
    if (!x) {
        std::rethrow_exception(x.second());
    }
}

struct derivation {
    std::string parent;
    string_map<double> globals;        // global overrides relative to parent
    string_map<std::string> ion_remap; // ion name remap overrides relative to parent
    mechanism_info_ptr derived_info;
};


// (Pimpl) catalogue state.

struct catalogue_state {
    catalogue_state() = default;

    catalogue_state(const catalogue_state& other) {
        info_map_.clear();
        for (const auto& kv: other.info_map_) {
            info_map_[kv.first] = make_unique<mechanism_info>(*kv.second);
        }

        derived_map_.clear();
        for (const auto& kv: other.derived_map_) {
            const derivation& v = kv.second;
            derived_map_[kv.first] = {v.parent, v.globals, v.ion_remap, make_unique<mechanism_info>(*v.derived_info)};
        }

        impl_map_.clear();
        for (const auto& name_impls: other.impl_map_) {
            std::unordered_map<std::type_index, std::unique_ptr<mechanism>> impls;
            for (const auto& tidx_mptr: name_impls.second) {
                impls[tidx_mptr.first] = tidx_mptr.second->clone();
            }

            impl_map_[name_impls.first] = std::move(impls);
        }
    }

    // Check for presence of mechanism or derived mechanism.
    bool defined(const std::string& name) const {
        return info_map_.count(name) || derived_map_.count(name);
    }

    // Check if name is derived or implicitly derivable.
    bool is_derived(const std::string& name) const {
        return derived_map_.count(name) || derive(name);
    }

    // Set mechanism info (unchecked).
    void bind(const std::string& name, mechanism_info info) {
        info_map_[name] = mechanism_info_ptr(new mechanism_info(std::move(info)));
    }

    // Add derived mechanism (unchecked).
    void bind(const std::string& name, derivation deriv) {
        derived_map_[name] = std::move(deriv);
    }

    // Register concrete mechanism for a back-end type.
    hopefully<void> register_impl(std::type_index tidx, const std::string& name, std::unique_ptr<mechanism> mech) {
        if (auto fptr = fingerprint_ptr(name)) {
            if (mech->fingerprint()!=*fptr.first()) {
                return make_exception_ptr(fingerprint_mismatch(name));
            }

            impl_map_[name][tidx] = std::move(mech);
        }
        else {
            return fptr.second();
        }

        return {};
    }

    // Remove mechanism and its derivations and implementations.
    void remove(const std::string& name) {
        derived_map_.erase(name);
        info_map_.erase(name);
        impl_map_.erase(name);

        // Erase any dangling derivation map entries.
        std::size_t n_delete;
        do {
            n_delete = 0;
            for (auto it = derived_map_.begin(); it!=derived_map_.end(); ) {
                const auto& parent = it->second.parent;
                if (info_map_.count(parent) || derived_map_.count(parent)) {
                    ++it;
                }
                else {
                    impl_map_.erase(it->first);
                    derived_map_.erase(it++);
                    ++n_delete;
                }
            }
        } while (n_delete>0);
    }

    // Retrieve mechanism info for mechanism, derived mechanism, or implicitly
    // derived mechanism.
    hopefully<mechanism_info> info(const std::string& name) const {
        if (const auto& deriv = value_by_key(derived_map_, name)) {
            return *(deriv->derived_info.get());
        }
        else if (auto p = value_by_key(info_map_, name)) {
            return *(p->get());
        }
        else if (auto deriv = derive(name)) {
            return *(deriv.first().derived_info.get());
        }
        else {
            return deriv.second();
        }
    }

    // Retrieve mechanism fingerprint. The fingerprint of a derived mechanisms
    // is that of its parent.
    hopefully<const mechanism_fingerprint*> fingerprint_ptr(const std::string& name) const {
        hopefully<derivation> implicit_deriv;
        const std::string* base = &name;

        if (!defined(name)) {
            if ((implicit_deriv = derive(name))) {
                base = &implicit_deriv.first().parent;
            }
            else {
                return implicit_deriv.second();
            }
        }

        while (auto maybe_deriv = value_by_key(derived_map_, *base)) {
            base = &maybe_deriv->parent;
        }

        if (const auto& p = value_by_key(info_map_, *base)) {
            return &p.value()->fingerprint;
        }

        throw arbor_internal_error("inconsistent catalogue map state");
    }

    // Construct derived mechanism based on existing parent mechanism and overrides.
    hopefully<derivation> derive(
        const std::string& name, const std::string& parent,
        const std::vector<std::pair<std::string, double>>& global_params,
        const std::vector<std::pair<std::string, std::string>>& ion_remap_vec) const
    {
        if (defined(name)) {
            return make_exception_ptr(duplicate_mechanism(name));
        }
        else if (!defined(parent)) {
            return make_exception_ptr(no_such_mechanism(parent));
        }

        string_map<std::string> ion_remap_map(ion_remap_vec.begin(), ion_remap_vec.end());
        derivation deriv = {parent, {}, ion_remap_map, nullptr};

        mechanism_info_ptr new_info;
        if (auto parent_info = info(parent)) {
            new_info.reset(new mechanism_info(parent_info.first()));
        }
        else {
            return parent_info.second();
        }

        // Update global parameter values in info for derived mechanism.

        for (const auto& kv: global_params) {
            const auto& param = kv.first;
            const auto& value = kv.second;

            if (auto p = value_by_key(new_info->globals, param)) {
                if (!p->valid(value)) {
                    return make_exception_ptr(invalid_parameter_value(name, param, value));
                }
            }
            else {
                return make_exception_ptr(no_such_parameter(name, param));
            }

            deriv.globals[param] = value;
            new_info->globals.at(param).default_value = value;
        }

        for (const auto& kv: ion_remap_vec) {
            if (!new_info->ions.count(kv.first)) {
                return make_exception_ptr(invalid_ion_remap(name, kv.first, kv.second));
            }
        }

        // Update ion dependencies in info to reflect the requested ion remapping.

        string_map<ion_dependency> new_ions;
        for (const auto& kv: new_info->ions) {
            if (auto new_ion = value_by_key(ion_remap_map, kv.first)) {
                if (!new_ions.insert({*new_ion, kv.second}).second) {
                    return make_exception_ptr(invalid_ion_remap(name, kv.first, *new_ion));
                }
            }
            else {
                if (!new_ions.insert(kv).second) {
                    // (find offending remap to report in exception)
                    for (const auto& entry: ion_remap_map) {
                        if (entry.second==kv.first) {
                            return make_exception_ptr(invalid_ion_remap(name, kv.first, entry.second));
                        }
                    }
                    throw arbor_internal_error("inconsistent catalogue ion remap state");
                }
            }
        }
        new_info->ions = std::move(new_ions);

        deriv.derived_info = std::move(new_info);
        return deriv;
    }

    // Implicit derivation.
    hopefully<derivation> derive(const std::string& name) const {
        if (defined(name)) {
            return make_exception_ptr(duplicate_mechanism(name));
        }

        auto i = name.find_last_of('/');
        if (i==std::string::npos) {
            return make_exception_ptr(no_such_mechanism(name));
        }

        std::string base = name.substr(0, i);
        if (!defined(base)) {
            return make_exception_ptr(no_such_mechanism(base));
        }

        std::string suffix = name.substr(i+1);

        const mechanism_info_ptr& info = derived_map_.count(base)? derived_map_.at(base).derived_info: info_map_.at(base);
        bool single_ion = info->ions.size()==1u;
        auto is_ion = [&info](const std::string& name) -> bool { return info->ions.count(name); };

        std::vector<std::pair<std::string, double>> global_params;
        std::vector<std::pair<std::string, std::string>> ion_remap;

        while (!suffix.empty()) {
            std::string assign;

            auto comma = suffix.find(',');
            if (comma==std::string::npos) {
                assign = suffix;
                suffix.clear();
            }
            else {
                assign = suffix.substr(0, comma);
                suffix = suffix.substr(comma+1);
            }

            std::string k, v;
            auto eq = assign.find('=');
            if (eq==std::string::npos) {
                if (!single_ion) {
                    return make_exception_ptr(invalid_ion_remap(assign));
                }

                k = info->ions.begin()->first;
                v = assign;
            }
            else {
                k = assign.substr(0, eq);
                v = assign.substr(eq+1);
            }

            if (is_ion(k)) {
                ion_remap.push_back({k, v});
            }
            else {
                char* end = 0;
                double v_value = std::strtod(v.c_str(), &end);
                if (!end || *end) {
                    return make_exception_ptr(invalid_parameter_value(name, k, v));
                }
                global_params.push_back({k, v_value});
            }
        }

        return derive(name, base, global_params, ion_remap);
    }

    // Retrieve implementation for this mechanism name or closest ancestor.
    hopefully<std::unique_ptr<mechanism>> implementation(std::type_index tidx, const std::string& name) const {
        const std::string* impl_name = &name;
        hopefully<derivation> implicit_deriv;

        if (!defined(name)) {
            implicit_deriv = derive(name);
            if (!implicit_deriv) {
                return implicit_deriv.second();
            }
            impl_name = &implicit_deriv.first().parent;
        }

        for (;;) {
            if (const auto mech_impls = value_by_key(impl_map_, *impl_name)) {
                if (auto p = value_by_key(mech_impls.value(), tidx)) {
                    return p->get()->clone();
                }
            }

            // Try parent instead.
            if (const auto p = value_by_key(derived_map_, *impl_name)) {
                impl_name = &p->parent;
            }
            else {
                return make_exception_ptr(no_such_implementation(name));
            }
        }
    }

    // Accumulate override set from derivation chain.
    hopefully<mechanism_overrides> overrides(const std::string& name) const {
        mechanism_overrides over;

        auto apply_deriv = [](mechanism_overrides& over, const derivation& deriv) {
            for (auto& kv: deriv.globals) {
                over.globals[kv.first] = kv.second;
            }

            if (!deriv.ion_remap.empty()) {
                string_map<std::string> new_rebind = deriv.ion_remap;
                for (auto& kv: over.ion_rebind) {
                    if (auto opt_v = value_by_key(deriv.ion_remap, kv.second)) {
                        new_rebind.erase(kv.second);
                        new_rebind[kv.first] = *opt_v;
                    }
                }
                for (auto& kv: over.ion_rebind) {
                    if (!value_by_key(deriv.ion_remap, kv.second)) {
                        new_rebind[kv.first] = kv.second;
                    }
                }
                std::swap(new_rebind, over.ion_rebind);
            }
        };

        // Recurse up the derivation tree to find the most distant ancestor;
        // accumulate global parameter settings and ion remappings down to the
        // requested mechanism.

        auto apply_globals = [this, &apply_deriv](auto& self, const std::string& name, mechanism_overrides& over) -> void {
            if (auto p = value_by_key(derived_map_, name)) {
                self(self, p->parent, over);
                apply_deriv(over, *p);
            }
        };

        util::optional<derivation> implicit_deriv;
        if (!defined(name)) {
            if (auto deriv = derive(name)) {
                implicit_deriv = std::move(deriv.first());
            }
            else {
                return deriv.second();
            }
        }

        apply_globals(apply_globals, implicit_deriv? implicit_deriv->parent: name, over);
        if (implicit_deriv) {
            apply_deriv(over, implicit_deriv.value());
        }

        return over;
    }

    // Schemata for (un-derived) mechanisms.
    string_map<mechanism_info_ptr> info_map_;

    // Parent and global setting values for derived mechanisms.
    string_map<derivation> derived_map_;

    // Prototype register, keyed on mechanism name, then backend type (index).
    string_map<std::unordered_map<std::type_index, mechanism_ptr>> impl_map_;
};

// Mechanism catalogue method implementations.

mechanism_catalogue::mechanism_catalogue():
    state_(new catalogue_state)
{}

mechanism_catalogue::mechanism_catalogue(mechanism_catalogue&& other) = default;
mechanism_catalogue& mechanism_catalogue::operator=(mechanism_catalogue&& other) = default;

mechanism_catalogue::mechanism_catalogue(const mechanism_catalogue& other):
    state_(new catalogue_state(*other.state_))
{}

mechanism_catalogue& mechanism_catalogue::operator=(const mechanism_catalogue& other) {
    state_.reset(new catalogue_state(*other.state_));
    return *this;
}

void mechanism_catalogue::add(const std::string& name, mechanism_info info) {
    if (state_->defined(name)) {
        throw duplicate_mechanism(name);
    }
    state_->bind(name, std::move(info));
}

bool mechanism_catalogue::has(const std::string& name) const {
    return state_->defined(name) || state_->derive(name);
}

bool mechanism_catalogue::is_derived(const std::string& name) const {
    return state_->is_derived(name);
}

mechanism_info mechanism_catalogue::operator[](const std::string& name) const {
    return value(state_->info(name));
}

const mechanism_fingerprint& mechanism_catalogue::fingerprint(const std::string& name) const {
    return *value(state_->fingerprint_ptr(name));
}

void mechanism_catalogue::derive(const std::string& name, const std::string& parent,
    const std::vector<std::pair<std::string, double>>& global_params,
    const std::vector<std::pair<std::string, std::string>>& ion_remap_vec)
{
    state_->bind(name, value(state_->derive(name, parent, global_params, ion_remap_vec)));
}

void mechanism_catalogue::remove(const std::string& name) {
    if (!has(name)) {
        throw no_such_mechanism(name);
    }
    state_->remove(name);
}

void mechanism_catalogue::register_impl(std::type_index tidx, const std::string& name, std::unique_ptr<mechanism> mech) {
    value(state_->register_impl(tidx, name, std::move(mech)));
}

std::pair<mechanism_ptr, mechanism_overrides> mechanism_catalogue::instance_impl(std::type_index tidx, const std::string& name) const {
    std::pair<mechanism_ptr, mechanism_overrides> result;
    result.first = value(state_->implementation(tidx, name));
    result.second = value(state_->overrides(name));

    return result;
}

mechanism_catalogue::~mechanism_catalogue() = default;

} // namespace arb

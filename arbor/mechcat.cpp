#include <map>
#include <memory>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/mechcat.hpp>

#include "util/maputil.hpp"

/* Notes on implementation:
 *
 * The catalogue maintains the following data:
 *
 * 1. impl_map_
 *
 *    This contains the mapping between mechanism names and concrete mechanisms
 *    for a specific backend that have been registered with
 *    register_implementation().
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
 */

namespace arb {

using util::value_by_key;
using std::make_unique;

void mechanism_catalogue::add(const std::string& name, mechanism_info info) {
    if (has(name)) {
        throw duplicate_mechanism(name);
    }

    info_map_[name] = mechanism_info_ptr(new mechanism_info(std::move(info)));
}

const mechanism_info& mechanism_catalogue::operator[](const std::string& name) const {
    if (const auto& deriv = value_by_key(derived_map_, name)) {
        return *(deriv->derived_info.get());
    }
    else if (auto p = value_by_key(info_map_, name)) {
        return *(p->get());
    }

    throw no_such_mechanism(name);
}

const mechanism_fingerprint& mechanism_catalogue::fingerprint(const std::string& name) const {
    std::string base = name;
    while (auto deriv = value_by_key(derived_map_, base)) {
        base = deriv->parent;
    }

    if (const auto& p = value_by_key(info_map_, base)) {
        return p.value()->fingerprint;
    }

    throw no_such_mechanism(name);
}

void mechanism_catalogue::derive(const std::string& name, const std::string& parent,
    const std::vector<std::pair<std::string, double>>& global_params,
    const std::vector<std::pair<std::string, std::string>>& ion_remap_vec)
{
    if (has(name)) {
        throw duplicate_mechanism(name);
    }

    if (!has(parent)) {
        throw no_such_mechanism(parent);
    }

    string_map<std::string> ion_remap_map(ion_remap_vec.begin(), ion_remap_vec.end());
    derivation deriv = {parent, {}, ion_remap_map, nullptr};
    mechanism_info_ptr info = mechanism_info_ptr(new mechanism_info((*this)[deriv.parent]));

    // Update global parameter values in info for derived mechanism.

    for (const auto& kv: global_params) {
        const auto& param = kv.first;
        const auto& value = kv.second;

        if (auto p = value_by_key(info->globals, param)) {
            if (!p->valid(value)) {
                throw invalid_parameter_value(name, param, value);
            }
        }
        else {
            throw no_such_parameter(name, param);
        }

        deriv.globals[param] = value;
        info->globals.at(param).default_value = value;
    }

    for (const auto& kv: ion_remap_vec) {
        if (!info->ions.count(kv.first)) {
            throw invalid_ion_remap(name, kv.first, kv.second);
        }
    }

    // Update ion dependencies in info to reflect the requested ion remapping.

    string_map<ion_dependency> new_ions;
    for (const auto& kv: info->ions) {
        if (auto new_ion = value_by_key(ion_remap_map, kv.first)) {
            if (!new_ions.insert({*new_ion, kv.second}).second) {
                throw invalid_ion_remap(name, kv.first, *new_ion);
            }
        }
        else {
            if (!new_ions.insert(kv).second) {
                // (find offending remap to report in exception)
                for (const auto& entry: ion_remap_map) {
                    if (entry.second==kv.first) {
                        throw invalid_ion_remap(name, kv.first, entry.second);
                    }
                }
                throw arbor_internal_error("inconsistent catalogue ion remap state");
            }
        }
    }
    info->ions = std::move(new_ions);

    deriv.derived_info = std::move(info);
    derived_map_[name] = std::move(deriv);
}

void mechanism_catalogue::remove(const std::string& name) {
    if (!has(name)) {
        throw no_such_mechanism(name);
    }

    if (is_derived(name)) {
        derived_map_.erase(name);
    }
    else {
        info_map_.erase(name);
        impl_map_.erase(name);
    }

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
                derived_map_.erase(it++);
                ++n_delete;
            }
        }
    } while (n_delete>0);
}

std::pair<std::unique_ptr<mechanism>, mechanism_overrides>
mechanism_catalogue::instance_impl(std::type_index tidx, const std::string& name) const {
    std::pair<std::unique_ptr<mechanism>, mechanism_overrides> mech;

    // Find implementation associated with this name or its closest ancestor.

    auto impl_name = name;
    const mechanism* prototype = nullptr;

    for (;;) {
        if (const auto mech_impls = value_by_key(impl_map_, impl_name)) {
            if (auto p = value_by_key(mech_impls.value(), tidx)) {
                prototype = p->get();
                break;
            }
        }

        // Try parent instead.
        if (const auto p = value_by_key(derived_map_, impl_name)) {
            impl_name = p->parent;
        }
        else {
            throw no_such_implementation(name);
        }
    }

    mech.first = prototype->clone();

    // Recurse up the derivation tree to find the most distant ancestor;
    // accumulate global parameter settings and ion remappings down to the
    // requested mechanism.

    auto apply_globals = [this](auto& self, const std::string& name, mechanism_overrides& over) -> void {
        if (auto p = value_by_key(derived_map_, name)) {
            self(self, p->parent, over);

            for (auto& kv: p->globals) {
                over.globals[kv.first] = kv.second;
            }

            if (!p->ion_remap.empty()) {
                string_map<std::string> new_rebind = p->ion_remap;
                for (auto& kv: over.ion_rebind) {
                    if (auto opt_v = value_by_key(p->ion_remap, kv.second)) {
                        new_rebind.erase(kv.second);
                        new_rebind[kv.first] = *opt_v;
                    }
                }
                for (auto& kv: over.ion_rebind) {
                    if (!value_by_key(p->ion_remap, kv.second)) {
                        new_rebind[kv.first] = kv.second;
                    }
                }
                std::swap(new_rebind, over.ion_rebind);
            }
        }
    };
    apply_globals(apply_globals, name, mech.second);
    return mech;
}

void mechanism_catalogue::register_impl(std::type_index tidx, const std::string& name, std::unique_ptr<mechanism> mech) {
    const mechanism_info& info = (*this)[name];

    if (mech->fingerprint()!=info.fingerprint) {
        throw fingerprint_mismatch(name);
    }

    impl_map_[name][tidx] = std::move(mech);
}

void mechanism_catalogue::copy_impl(const mechanism_catalogue& other) {
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

void parameterize_over_ion(mechanism_catalogue& cat, const std::string& name, const std::string& ion) {
    mechanism_info info = cat[name];
    if (info.ions.size()!=1) {
        throw invalid_ion_remap(name);
    }

    std::string from_ion = info.ions.begin()->first;
    cat.derive(name+"/"+ion, name, {}, {{from_ion, ion}});
}

} // namespace arb

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/make_unique.hpp>

#include "util/maputil.hpp"

namespace arb {

using util::value_by_key;
using util::make_unique;

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

void mechanism_catalogue::derive(const std::string& name, const std::string& parent, const std::vector<std::pair<std::string, double>>& global_params) {
    if (has(name)) {
        throw duplicate_mechanism(name);
    }

    if (!has(parent)) {
        throw no_such_mechanism(parent);
    }

    derivation deriv = {parent, {}, nullptr};
    mechanism_info_ptr info = mechanism_info_ptr(new mechanism_info((*this)[deriv.parent]));

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

std::unique_ptr<mechanism> mechanism_catalogue::instance_impl(std::type_index tidx, const std::string& name) const {
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

    std::unique_ptr<mechanism> mech = prototype->clone();
    // TODO: make recursive lambda without std::function in C++14
    std::function<void (std::string, mechanism*)> apply_globals =
        [this, &apply_globals](const std::string& name, mechanism* mptr)
    {
        if (auto p = value_by_key(derived_map_, name)) {
            apply_globals(p->parent, mptr);

            for (auto& kv: p->globals) {
                mptr->set_global(kv.first, kv.second);
            }
        }
    };

    apply_globals(name, mech.get());
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
        derived_map_[kv.first] = {v.parent, v.globals, make_unique<mechanism_info>(*v.derived_info)};
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

} // namespace arb

#include <string>
#include <utility>

#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/util/expected.hpp>

namespace arb {
// Evaluation of a named region or locset requires the recursive evaluation of
// any component regions or locsets in its definition.
//
// During the initialization phase, 'named' expressions will be looked up in the
// provided label_dict, and the maps updated accordingly. Post-initialization,
// label_dict_ptr will be null, and concrete regions/locsets will only be retrieved
// from the maps established during initialization.
//
// NOTE: This is _recursive_ since the call to _thingify_ will use _regions_ and ilk.
// This is also why we need to tuck away the label_dict inside this class.

template <typename ConcreteMap, typename LabelMap>
static const auto& try_lookup(const mprovider& provider, const std::string& name, ConcreteMap& map, const LabelMap& dict) {
    auto it = map.find(name);
    if (it==map.end()) {
        map.emplace(name, util::unexpect);
        auto it = dict.find(name);
        if (it==dict.end()) throw unbound_name(name);
        return (map[name] = thingify(it->second, provider)).value();
    }
    else if (!it->second) {
        throw circular_definition(name);
    }
    else {
        return it->second.value();
    }
}

template <typename ConcreteMap>
static const auto& try_lookup(const mprovider& provider, const std::string& name, ConcreteMap& map) {
    auto it = map.find(name);
    if (it==map.end()) {
        throw unbound_name(name);
    }
    else if (!it->second) {
        throw circular_definition(name);
    }
    else {
        return it->second.value();
    }
}

mprovider::mprovider(arb::morphology m, const label_dict* ldptr):
    morphology_(m),
    embedding_(m),
    label_dict_ptr(ldptr) {
    // Evaluate each named region or locset in provided dictionary
    // to populate concrete regions_, locsets_ maps.
    if (label_dict_ptr) {
        for (const auto& pair: label_dict_ptr->regions()) {
            (void)(this->region(pair.first));
        }

        for (const auto& pair: label_dict_ptr->locsets()) {
            (void)(this->locset(pair.first));
        }

        for (const auto& pair: label_dict_ptr->iexpressions()) {
            (void)(this->iexpr(pair.first));
        }
        label_dict_ptr = nullptr;
    }
}


const mextent& mprovider::region(const std::string& name) const {
    if (label_dict_ptr) {
        return try_lookup(*this, name, regions_, label_dict_ptr->regions());
    } else {
        return try_lookup(*this, name, regions_);
    }
}

const mlocation_list& mprovider::locset(const std::string& name) const {
    if (label_dict_ptr) {
        return try_lookup(*this, name, locsets_, label_dict_ptr->locsets());
    } else {
        return try_lookup(*this, name, locsets_);
    }
}

const iexpr_ptr& mprovider::iexpr(const std::string& name) const {
    if (label_dict_ptr) {
        return try_lookup(*this, name, iexpressions_, label_dict_ptr->iexpressions());
    } else {
        return try_lookup(*this, name, iexpressions_);
    }
}

} // namespace arb

#include <string>

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
template <typename RegOrLocMap, typename LabelDictMap>
static const auto& try_build(const mprovider& provider,
                             const std::string& name,
                             RegOrLocMap& map,
                             const LabelDictMap& dict) {
    if (auto it = map.find(name); it != map.end()) {
        if (it->second) return it->second.value();
        throw circular_definition(name);
    }
    map[name] = {};
    if (auto nm = dict.find(name); nm != dict.end()) {
        // NOTE this is the point of recursion!
        map[name] = thingify(nm->second, provider);
        return map[name].value();
    }
    throw unbound_name(name);
}

template <typename RegOrLocMap>
static const auto& try_lookup(const mprovider& provider,
                             const std::string& name,
                             RegOrLocMap& map){
    if (auto it = map.find(name); it != map.end()) {
        if (it->second) return it->second.value();
        throw circular_definition(name);
    }
    throw unbound_name(name);
}


const mextent& mprovider::region(const std::string& name) const {
    if (dict_) try_build(*this, name, regions_, dict_->regions());
    return try_lookup(*this, name, regions_);
}
const mlocation_list& mprovider::locset(const std::string& name) const {
    if (dict_) try_build(*this, name, locsets_, dict_->locsets());
    return try_lookup(*this, name, locsets_);
}
const iexpr_ptr& mprovider::iexpr(const std::string& name) const {
    if (dict_) try_build(*this, name, iexpressions_, dict_->iexpressions());
    return try_lookup(*this, name, iexpressions_);
}

} // namespace arb

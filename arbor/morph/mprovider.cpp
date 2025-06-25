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
template <typename RegOrLocMap, typename LabelDictMap>
static const auto& try_build(const mprovider& provider, const std::string& name, RegOrLocMap& map, const LabelDictMap& dict) {
    auto it = map.find(name);
    if (it == map.end()) {
        map.emplace(name, std::nullopt);
        if (auto it = dict.find(name); it != dict.end()) {
            // NOTE this is the point of recursion!
            map[name] = thingify(it->second, provider);
            return map[name].value();
        }
        throw unbound_name(name);
    }
    else if (it->second) {
        return it->second.value();
    }
    throw circular_definition(name);
}

const mextent& mprovider::region(const std::string& name) const { return try_build(*this, name, regions_, dict_.regions()); }
const mlocation_list& mprovider::locset(const std::string& name) const { return try_build(*this, name, locsets_, dict_.locsets()); }
const iexpr_ptr& mprovider::iexpr(const std::string& name) const { return try_build(*this, name, iexpressions_, dict_.iexpressions()); }

} // namespace arb

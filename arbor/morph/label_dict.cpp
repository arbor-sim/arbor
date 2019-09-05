#include <algorithm>
#include <set>
#include <fstream>
#include <unordered_map>

#include <arbor/morph/error.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/region.hpp>

#include "util/filter.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"
#include "io/sepval.hpp"

namespace arb {

size_t label_dict::size() const {
    return locsets_.size() + regions_.size();
}

void label_dict::set(const std::string& name, arb::locset p) {
    if (regions_.count(name)) {
        throw morphology_error(util::pprintf(
                "Attempt to add a locset \"{}\" to a label dictionary that already contains a region with the same name.", name));
    }
    locsets_.insert({name, p});
}

void label_dict::set(const std::string& name, arb::region r) {
    if (locsets_.count(name)) {
        throw morphology_error(util::pprintf(
                "Attempt to add a region \"{}\" to a label dictionary that already contains a locset with the same name.", name));
    }
    regions_.insert({name, r});
}

util::optional<const region&> label_dict::region(const std::string& name) const {
    auto it = regions_.find(name);
    if (it==regions_.end() ) return {};
    return it->second;
}

util::optional<const locset&> label_dict::locset(const std::string& name) const {
    auto it = locsets_.find(name);
    if (it==locsets_.end() ) return {};
    return it->second;
}

const std::unordered_map<std::string, locset>& label_dict::locsets() const {
    return locsets_;
}

const std::unordered_map<std::string, region>& label_dict::regions() const {
    return regions_;
}

} // namespace arb

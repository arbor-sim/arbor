#include <optional>
#include <unordered_map>
#include <utility>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

label_dict& label_dict::add_swc_tags() {
    set("soma", reg::tagged(1));
    set("axon", reg::tagged(2));
    set("dend", reg::tagged(3));
    set("apic", reg::tagged(4));
    return *this;
}

size_t label_dict::size() const { return locsets_.size() + regions_.size() + iexpressions_.size(); }

label_dict& label_dict::set(const std::string& name, arb::locset ls) {
    if (regions_.contains(name) || iexpressions_.contains(name)) throw label_type_mismatch(name);
    locsets_[name] = std::move(ls);
    return *this;
}

label_dict& label_dict::set(const std::string& name, arb::region reg) {
    if (locsets_.contains(name) || iexpressions_.contains(name)) throw label_type_mismatch(name);
    regions_[name] = std::move(reg);
    return *this;
}

label_dict& label_dict::set(const std::string& name, arb::iexpr expr) {
    if (locsets_.contains(name) || regions_.contains(name)) throw label_type_mismatch(name);
    iexpressions_[name] = std::move(expr);
    return *this;
}

std::size_t label_dict::erase(const std::string& name) { return locsets_.erase(name) + regions_.erase(name) + iexpressions_.erase(name); }

label_dict& label_dict::extend(const label_dict& other, const std::string& prefix) {
    for (const auto& [k, v]: other.locsets()) set(prefix + k, v);
    for (const auto& [k, v]: other.regions()) set(prefix + k, v);
    for (const auto& [k, v]: other.iexpressions()) set(prefix + k, v);
    return *this;
}

std::optional<region> label_dict::region(const std::string& name) const {
    auto it = regions_.find(name);
    if (it==regions_.end()) return {};
    return it->second;
}

std::optional<locset> label_dict::locset(const std::string& name) const {
    auto it = locsets_.find(name);
    if (it==locsets_.end()) return {};
    return it->second;
}

std::optional<iexpr> label_dict::iexpr(const std::string& name) const {
    auto it = iexpressions_.find(name);
    if (it==iexpressions_.end()) return std::nullopt;
    return it->second;
}

const std::unordered_map<std::string, locset>& label_dict::locsets() const { return locsets_; }
const std::unordered_map<std::string, region>& label_dict::regions() const { return regions_; }
const std::unordered_map<std::string, iexpr>& label_dict::iexpressions() const { return iexpressions_; }

} // namespace arb

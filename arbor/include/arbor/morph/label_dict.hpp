#pragma once

#include <memory>
#include <unordered_map>

#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/util/optional.hpp>

namespace arb {

class label_dict {
    using ps_map = std::unordered_map<std::string, arb::locset>;
    using reg_map = std::unordered_map<std::string, arb::region>;
    ps_map locsets_;
    reg_map regions_;

public:

    void set(const std::string& name, locset ls);
    void set(const std::string& name, region reg);

    util::optional<const arb::region&> region(const std::string& name) const;
    util::optional<const arb::locset&> locset(const std::string& name) const;

    const ps_map& locsets() const;
    const reg_map& regions() const;

    std::size_t size() const;
};

} //namespace arb

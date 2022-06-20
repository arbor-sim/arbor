#pragma once

#include <memory>
#include <optional>
#include <unordered_map>

#include <arbor/export.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

namespace arb {

class ARB_ARBOR_API label_dict {
    using ps_map = std::unordered_map<std::string, arb::locset>;
    using reg_map = std::unordered_map<std::string, arb::region>;
    ps_map locsets_;
    reg_map regions_;

public:
    // construct a label dict with SWC tags predefined
    static label_dict with_swc_tags() {
        auto res = label_dict{};
        res.set("soma", reg::tagged(1));
        res.set("axon", reg::tagged(2));
        res.set("dend", reg::tagged(3));
        res.set("apic", reg::tagged(4));
        return res;
    }

    void import(const label_dict& other, const std::string& prefix = "");

    void set(const std::string& name, locset ls);
    void set(const std::string& name, region reg);

    std::optional<arb::region> region(const std::string& name) const;
    std::optional<arb::locset> locset(const std::string& name) const;

    const ps_map& locsets() const;
    const reg_map& regions() const;

    std::size_t size() const;
};

} //namespace arb

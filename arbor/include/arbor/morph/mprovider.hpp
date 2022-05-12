#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <arbor/export.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/iexpr.hpp>

namespace arb {

using concrete_embedding = embed_pwlin;

struct ARB_ARBOR_API mprovider {
    mprovider(arb::morphology m, const label_dict& dict): mprovider(m, &dict) {}
    explicit mprovider(arb::morphology m): mprovider(m, nullptr) {}

    // Throw exception on missing or recursive definition.
    const mextent& region(const std::string& name) const;
    const mlocation_list& locset(const std::string& name) const;
    const iexpr_ptr& iexpr(const std::string& name) const;

    // Read-only access to morphology and constructed embedding.
    const auto& morphology() const { return morphology_; }
    const auto& embedding() const { return embedding_; }

private:
    mprovider(arb::morphology m, const label_dict* ldptr):
        morphology_(m), embedding_(m), label_dict_ptr(ldptr) { init(); }

    arb::morphology morphology_;
    concrete_embedding embedding_;

    struct circular_def {};

    // Maps are mutated only during initialization phase of mprovider.
    mutable std::unordered_map<std::string, util::expected<mextent, circular_def>> regions_;
    mutable std::unordered_map<std::string, util::expected<mlocation_list, circular_def>> locsets_;
    mutable std::unordered_map<std::string, util::expected<iexpr_ptr, circular_def>> iexpressions_;

    // Non-null only during initialization phase.
    mutable const label_dict* label_dict_ptr;

    // Perform greedy initialization of concrete region, locset maps.
    void init();
};

} // namespace arb

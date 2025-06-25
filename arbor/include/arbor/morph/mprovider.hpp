#pragma once

#include <string>
#include <unordered_map>

#include <arbor/export.hpp>
#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/iexpr.hpp>

namespace arb {

using concrete_embedding = embed_pwlin;

struct ARB_ARBOR_API mprovider {
    mprovider(const arb::morphology& m): morphology_(m), embedding_(m) {}
    mprovider(const arb::morphology& m, const label_dict& dict): morphology_(m), embedding_(m), dict_(dict) {}

    // Throw exception on missing or recursive definition.
    const mextent& region(const std::string& name) const;
    const mlocation_list& locset(const std::string& name) const;
    const iexpr_ptr& iexpr(const std::string& name) const;

    // Read-only access to morphology and constructed embedding.
    const auto& morphology() const { return morphology_; }
    const auto& embedding() const { return embedding_; }

private:
    arb::morphology morphology_;
    concrete_embedding embedding_;
    const label_dict& dict_ = label_dict();

    template<typename M>
    using map = std::unordered_map<std::string, std::optional<M>>;
    // Maps to cache results.
    mutable map<mextent> regions_;
    mutable map<mlocation_list> locsets_;
    mutable map<iexpr_ptr> iexpressions_;
};

} // namespace arb

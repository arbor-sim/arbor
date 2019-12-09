#pragma once

#include <string>
#include <unordered_map>

#include <arbor/morph/embed_pwlin1d.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/util/either.hpp>

namespace arb {

using concrete_embedding = embed_pwlin1d;

struct mprovider {
    mprovider(arb::morphology m, const label_dict& dict):
        morphology_(m), embedding_(m), label_dict_ptr(&dict) { init(); }

    explicit mprovider(arb::morphology m):
        morphology_(m), embedding_(m), label_dict_ptr(nullptr) { init(); }

    // Throw exception on missing or recursive definition.
    const mcable_list& region(const std::string& name) const;
    const mlocation_list& locset(const std::string& name) const;

    // Read-only access to morphology and constructed embedding.
    const auto& morphology() const { return morphology_; }
    const auto& embedding() const { return embedding_; }

private:
    arb::morphology morphology_;
    concrete_embedding embedding_;

    struct circular_def {};

    // Maps are mutated only during initialization phase of mprovider.
    mutable std::unordered_map<std::string, util::either<mcable_list, circular_def>> regions_;
    mutable std::unordered_map<std::string, util::either<mlocation_list, circular_def>> locsets_;

    // Non-null only during initialization phase.
    mutable const label_dict* label_dict_ptr;

    // Perform greedy initialization of concrete region, locset maps.
    void init();
};

} // namespace arb

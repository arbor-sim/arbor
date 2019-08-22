#pragma once

#include <mutex>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

class em_morphology {
    morphology morpho_;
    std::mutex mutex_;

    mutable std::vector<mlocation> terminals_;
    mutable std::vector<mlocation> forks_;

public:
    em_morphology(const morphology& m):
        morpho_(m) {}

    const morphology& morph() const;

    mlocation_list terminals() const;
    mlocation_list forks() const;
    mlocation_list root() const;

    mlocation sample2loc(msize_t sid) const;
};

} // namespace arb

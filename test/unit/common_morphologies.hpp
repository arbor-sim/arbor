#pragma once

// A set of morphologies for testing discretization.

#include <utility>
#include <vector>
#include <arbor/morph/morphology.hpp>

namespace common_morphology {

inline std::vector<arb::msample> make_morph_samples(unsigned n) {
    std::vector<arb::msample> ms;
    for (unsigned i = 0; i<n; ++i) ms.push_back({{0., 0., (double)i, 0.5}, 5});
    return ms;
}

// Test morphologies for CV determination:
// Sample points have radius 0.5, giving an initial branch length of 1.0
// for morphologies with spherical roots.

static const arb::morphology m_empty;

// spherical root, one branch
static const arb::morphology m_sph_b1{arb::sample_tree(make_morph_samples(1), {arb::mnpos}), true};

// regular root, one branch
static const arb::morphology m_reg_b1{arb::sample_tree(make_morph_samples(2), {arb::mnpos, 0u}), false};

// spherical root, six branches
static const arb::morphology m_sph_b6{arb::sample_tree(make_morph_samples(8), {arb::mnpos, 0u, 1u, 0u, 3u, 4u, 4u, 4u}), true};

// regular root, six branches
static const arb::morphology m_reg_b6{arb::sample_tree(make_morph_samples(7), {arb::mnpos, 0u, 1u, 1u, 2u, 2u, 2u}), false};

// regular root, six branches, mutiple top level branches.
static const arb::morphology m_mlt_b6{arb::sample_tree(make_morph_samples(7), {arb::mnpos, 0u, 1u, 1u, 0u, 4u, 4u}), false};

static std::pair<const char*, arb::morphology> test_morphologies[] = {
    {"m_empty",  m_empty},
    {"m_sph_b1", m_sph_b1},
    {"m_reg_b1", m_reg_b1},
    {"m_sph_b6", m_sph_b6},
    {"m_reg_b6", m_reg_b6},
    {"m_mlt_b6", m_mlt_b6}
};

} // namespace common_morphology

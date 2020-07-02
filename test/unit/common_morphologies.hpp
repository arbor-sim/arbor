#pragma once

// A set of morphologies for testing discretization.

#include "util/span.hpp"

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <cstring>
#include <utility>
#include <vector>

#include "io/sepval.hpp"

namespace common_morphology {

inline arb::morphology make_morph(std::vector<arb::msize_t> parents, const char* name="") {
    auto nseg = parents.size();;
    auto point = [](int i) {return arb::mpoint{0, 0, (double)i, 0.5};};
    int tag = 1;

    if (!parents.size()) return {};

    arb::segment_tree tree;
    tree.append(arb::mnpos, point(-1), point(0), tag);
    for (arb::msize_t i=1; i<nseg; ++i) {
        int p = parents[i]==arb::mnpos? -1: parents[i];
        tree.append(parents[i], point(p), point(i), tag);
    }

    if (strlen(name)) {
        std::cout << name << " (" << arb::io::sepval(parents, ' ') << ")\n" << tree << "\n";
        std::cout << arb::morphology(tree) << "\n----------------------------------------------------\n";
    }

    return arb::morphology(tree);
}

// Test morphologies for CV determination:
// Sample points have radius 0.5, giving an initial branch length of 1.0
// for morphologies with spherical roots.

static const arb::morphology m_empty = make_morph({});

// regular root, one branch
//static const arb::morphology m_reg_b1{arb::segment_tree(make_morph_samples(2), {arb::mnpos, 0u})};
static const arb::morphology m_reg_b1 = make_morph({arb::mnpos});

// regular root, six branches
// branch 0 has child branches 1 and 2; branch 2 has child branches 3, 4 and 5.
static const arb::morphology m_reg_b6 = make_morph({arb::mnpos, 0u, 0u, 1u, 1u, 1u});
//static const arb::morphology m_reg_b6 = make_morph({arb::mnpos, 0u, 0u, 2u, 2u, 2u}, "m_reg_b6");
//static const arb::morphology m_reg_b6 = make_morph({arb::mnpos, 0u, 1u, 1u, 2u, 2u, 2u}, true);

// regular root, six branches, mutiple top level branches.
// branch 0 has child branches 1 and 2; branch 3 has child branches 4 and 5.
//static const arb::morphology m_mlt_b6 = make_morph({arb::mnpos, 0u, 1u, 1u, 0u, 4u, 4u});
static const arb::morphology m_mlt_b6 = make_morph({arb::mnpos, 0u, 0u, arb::mnpos, 3u, 3u});

static std::pair<const char*, arb::morphology> test_morphologies[] = {
    {"m_empty",  m_empty},
    {"m_reg_b1", m_reg_b1},
    {"m_reg_b6", m_reg_b6},
    {"m_mlt_b6", m_mlt_b6}
};

} // namespace common_morphology

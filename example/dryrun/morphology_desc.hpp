#pragma once

// Maintain a pool of morphologies for use with miniapp recipes.
// The default pool comprises a single ball-and-stick morphology;
// sets of morphologies can be loaded from SWC files.

#include <memory>
#include <string>
#include <vector>

#include <arbor/morphology.hpp>

#include <aux/glob.hpp>
#include <aux/path.hpp>

namespace arb {

    static morphology make_basic_y_morphology() {
        morphology morph;

        // soma of diameter 12.6157 microns.
        // proximal section: 200 microns, radius 0.5 microns.
        // two terminal branches, each: 100 microns, terminal radius 0.25 microns.
        morph.soma.r = 12.6157/2;
        double x = morph.soma.r;
        morph.add_section({{x, 0, 0, 0.5}, {x+200, 0, 0, 0.5}});
        x += 200;
        morph.add_section({{x, 0, 0, 0.5}, {x+100, 0, 0, 0.25}}, 1u);
        morph.add_section({{x, 0, 0, 0.5}, {x+100, 0, 0, 0.25}}, 1u);

        morph.assert_valid();
        return morph;
    }
} // namespace arb

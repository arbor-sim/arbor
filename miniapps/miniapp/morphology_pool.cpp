#include <fstream>
#include <memory>
#include <vector>

#include <morphology.hpp>
#include <swcio.hpp>
#include <util/path.hpp>

#include "morphology_pool.hpp"

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

morphology_pool default_morphology_pool(make_basic_y_morphology());

void load_swc_morphology(morphology_pool& pool, const util::path& swc_path) {
    std::ifstream fi;
    fi.exceptions(std::ifstream::failbit);

    fi.open(swc_path.c_str());
    pool.insert(io::swc_as_morphology(io::parse_swc_file(fi)));
}

void load_swc_morphology_glob(morphology_pool& pool, const std::string& swc_pattern) {
    std::ifstream fi;
    fi.exceptions(std::ifstream::failbit);

    auto swc_paths = util::glob(swc_pattern);
    for (const auto& p: swc_paths) {
        fi.open(p.c_str());
        pool.insert(io::swc_as_morphology(io::parse_swc_file(fi)));
        pool[pool.size()-1].assert_valid();
        fi.close();
    }
}


} // namespace arb

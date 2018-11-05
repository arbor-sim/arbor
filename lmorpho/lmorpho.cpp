#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include <arbor/morphology.hpp>
#include <arbor/util/optional.hpp>
#include <sup/tinyopt.hpp>

#include "morphio.hpp"
#include "lsystem.hpp"
#include "lsys_models.hpp"

using arb::util::optional;
using arb::util::nullopt;
using arb::util::just;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -n, --count=N      Number of morphologies to generate.\n"
"  -m, --model=MODEL  Use L-system MODEL for generation (see below).\n"
"  -g, --segment=DX   Segment model into compartments of max size DX µm.\n"
"  --swc=FILE         Output morphologies as SWC to FILE (see below).\n"
"  --pvec=FILE        Output 'parent vector' structural representation\n"
"                     to FILE.\n"
"  -h, --help         Emit this message and exit.\n"
"\n"
"Generate artificial neuron morphologies based on L-system descriptions.\n"
"\n"
"If a FILE argument contains a '%', then one file will be written for\n"
"each generated morphology, with the '%' replaced by the index of the\n"
"morphology, starting from zero. Output for each morphology will otherwise\n"
"be concatenated: SWC files will be headed by a comment line with the\n"
"index of the morphology; parent vectors will be merged into one long\n"
"vector.  A FILE argument of '-' corresponds to standard output.\n"
"\n"
"Currently supported MODELs:\n"
"    motoneuron    Adult cat spinal alpha-motoneurons, based on models\n"
"                  and data in Burke 1992 and Ascoli 2001.\n"
"    purkinje      Guinea pig Purkinje cells, basd on models and data\n"
"                  from Rapp 1994 and Ascoli 2001.\n";

int main(int argc, char** argv) {
    // options
    int n_morph = 1;
    optional<unsigned> rng_seed;
    optional<std::string> swc_file;
    optional<std::string> pvector_file;
    double segment_dx = 0;

    std::pair<const char*, const lsys_param*> models[] = {
        {"motoneuron", &alpha_motoneuron_lsys},
        {"purkinje", &purkinje_lsys}
    };
    lsys_param P;

    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto o = to::parse_opt<int>(arg, 'n', "count")) {
                n_morph = *o;
            }
            if (auto o = to::parse_opt<int>(arg, 's', "seed")) {
                rng_seed = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "swc")) {
                swc_file = *o;
            }
            else if (auto o = to::parse_opt<double>(arg, 'g', "segment")) {
                segment_dx = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 'p', "pvec")) {
                pvector_file = *o;
            }
            else if (auto o = to::parse_opt<const lsys_param*>(arg, 'm', "model", to::keywords(models))) {
                P = **o;
            }
            else if (to::parse_opt(arg, 'h', "help")) {
                std::cout << "Usage: " << argv[0] << " " << usage_str;
                return 0;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        std::minstd_rand g;
        if (rng_seed) g.seed(rng_seed.value());

        auto emit_swc = swc_file? just(swc_emitter(*swc_file, n_morph)): nullopt;
        auto emit_pvec = pvector_file? just(pvector_emitter(*pvector_file, n_morph)): nullopt;

        for (int i=0; i<n_morph; ++i) {
            auto morph = generate_morphology(P, g);
            morph.segment(segment_dx);

            if (emit_swc) (*emit_swc)(i, morph);
            if (emit_pvec) (*emit_pvec)(i, morph);
        }
    }
    catch (to::parse_opt_error& e) {
        std::cerr << argv[0] << ": " << e.what() << "\n";
        std::cerr << "Try '" << argv[0] << " --help' for more information.\n";
        std::exit(2);
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }
}


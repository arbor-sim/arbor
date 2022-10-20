#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include <tinyopt/tinyopt.h>

#include "lsystem.hpp"
#include "lsys_models.hpp"

#include <arbor/cable_cell.hpp>
#include <arborio/cableio.hpp>
#include <arborio/label_parse.hpp>

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -n, --count=N      Number of morphologies to generate.\n"
"  -m, --model=MODEL  Use L-system MODEL for generation (see below).\n"
"  --acc=FILE         Output morphologies as ACC to FILE.\n"
"  -h, --help         Emit this message and exit.\n"
"\n"
"Generate artificial neuron morphologies based on L-system descriptions.\n"
"\n"
"If a FILE argument contains a '%', then one file will be written for\n"
"each generated morphology, with the '%' replaced by the index of the\n"
"morphology, starting from zero.\n"
"A FILE argument of '-' corresponds to standard output.\n"
"\n"
"Currently supported MODELs:\n"
"    motoneuron    Adult cat spinal alpha-motoneurons, based on models\n"
"                  and data in Burke 1992 and Ascoli 2001.\n"
"    purkinje      Guinea pig Purkinje cells, basd on models and data\n"
"                  from Rapp 1994 and Ascoli 2001.\n";

int main(int argc, char** argv) {
    // options
    int n_morph = 1;
    std::optional<unsigned> rng_seed = 0;
    lsys_param P;
    bool help = false;
    std::optional<std::string> acc_file;

    std::pair<const char*, const lsys_param*> models[] = {
        {"motoneuron", &alpha_motoneuron_lsys},
        {"purkinje", &purkinje_lsys}
    };


    try {
        for (auto arg = argv+1; *arg; ) {
            bool ok = false;
            ok |= (n_morph << to::parse<int>(arg, "--n", "--count")).has_value();
            ok |= (rng_seed << to::parse<int>(arg, "--s", "--seed")).has_value();
            ok |= (acc_file << to::parse<std::string>(arg, "-f", "--acc")).has_value();
            const lsys_param* o;
            if((o << to::parse<const lsys_param*>(arg, to::keywords(models), "-m", "--model")).has_value()) {
                P = *o;
            }
            ok |= (help << to::parse(arg, "-h", "--help")).has_value();
            if (!ok) throw to::option_error("unrecognized argument", *arg);
        }

        if (help) {
            to::usage(argv[0], usage_str);
            return 0;
        }

        std::minstd_rand g;
        if (rng_seed) {
            g.seed(rng_seed.value());
        }

        using namespace arborio::literals;

        auto apical = apical_lsys;
        auto basal = basal_lsys;

        for (int i = 0; i < n_morph; ++i) {
            const arb::segment_tree morph = generate_morphology(P.diam_soma, std::vector<lsys_param>{apical, basal}, g);

            arb::label_dict labels;
            labels.set("soma", "(tag 1)"_reg);
            labels.set("basal", "(tag 3)"_reg);
            labels.set("apical", "(tag 4)"_reg);
            arb::decor decor;

            arb::cable_cell cell(morph, decor, labels);

            if(acc_file) {
                std::string filename = acc_file.value();
                const std::string toReplace("%");
                auto pos = filename.find(toReplace);
                if(pos != std::string::npos) {
                    filename.replace(pos, toReplace.length(), std::to_string(i));
                }

                if(filename == "-") {
                    arborio::write_component(std::cout, cell);
                } else {
                    std::ofstream of(filename);
                    arborio::write_component(of, cell);
                }

            }
        }

    }  catch (to::option_error& e) {
        std::cerr << argv[0] << ": " << e.what() << "\n";
        std::cerr << "Try '" << argv[0] << " --help' for more information.\n";
        std::exit(2);
    }
    catch (std::exception& e) {
        std::cerr << "caught exception: " << e.what() << "\n";
        std::exit(1);
    }

}


#include <algorithm>
#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include <swcio.hpp>
#include <tinyopt.hpp>

#include "morphology.h"
#include "lsystem.h"

using nest::mc::io::swc_record;
namespace to = nest::mc::to;

const char* usage_str =
"[OPTION]...\n"
"\n"
"  -n, --count=N   Number of morphologies to generate\n"
"  --swc=FILE      Output morphologies as SWC to FILE; '%' in the\n"
"                  file name is replaced with the number of the morphology.\n";

template <typename... Args>
std::string strprintf(const char* fmt, Args&&... args) {
    thread_local static std::vector<char> buffer(1024);

    for (;;) {
        int n = std::snprintf(buffer.data(), buffer.size(), fmt, std::forward<Args>(args)...);
        if (n<0) return ""; // error
        if (n<buffer.size()) return std::string(buffer.data());
        buffer.resize(2*n);
    }
}

template <typename... Args>
std::string strprintf(std::string fmt, Args&&... args) {
    return strprintf(fmt.c_str(), std::forward<Args>(args)...);
}

struct multi_file_writer {
    std::ofstream file_;
    bool concat_ = false;
    bool use_stdout_ = false;
    std::string fmt_;       // use if not concat_
    std::string filename_;  // use if concat_
    int current_n_ = 0;

    explicit multi_file_writer(const std::string& pattern, int digits=0) {
        auto npos = std::string::npos;

        file_.exceptions(std::ofstream::failbit);
        concat_ = (pattern.find("%")==npos);
        use_stdout_ = pattern.empty() || pattern=="-";

        if (!concat_) {
            std::string nfmt = digits? "%0"+std::to_string(digits)+"d": "%d";
            std::string::size_type i = 0;
            for (;;) {
                auto p = pattern.find("%", i);

                if (p==npos) {
                    fmt_ += pattern.substr(i);
                    break;
                }
                else {
                    fmt_ += pattern.substr(i, p-i);
                    fmt_ += i==0? nfmt: "%%";
                    i = p+1;
                }
            }
        }
        else {
            filename_ = pattern;
        }
    }

    void open(int n) {
        if (use_stdout_ || (file_.is_open() && (concat_ || n==current_n_))) {
            return;
        }

        if (file_.is_open()) file_.close();

        std::string fname = concat_? filename_: strprintf(fmt_, n);
        file_.open(fname);

        current_n_ = n;
    }

    void close() {
        file_.close();
    }

    std::ostream& stream() { return use_stdout_? std::cout: file_; }
};

std::vector<swc_record> as_swc(const morphology& morph) {
    using kind = swc_record::kind;
    std::map<int, int> parent_end_id;
    std::vector<swc_record> swc;

    // soma
    const auto &p = morph.soma;
    int id = 0;
    parent_end_id[0] = 0;
    swc.emplace_back(kind::soma, id, p.x, p.y, p.z, p.r, -1);

    // dendrites:
    for (auto& seg: morph.segments) {
        int parent = parent_end_id[seg.parent_id];

        const auto& points = seg.points;
        auto n = points.size();
        if (n<2) {
            throw std::runtime_error(strprintf("surprisingly short cable: id=%d, size=%ul", seg.id, n));
        }

        // include first point only for dendrites segments attached to soma.
        if (seg.parent_id==0) {
            const auto& p = points[0];
            swc.emplace_back(kind::dendrite, ++id, p.x, p.y, p.z, p.r, parent);
            parent = id;
        }

        for (unsigned i = 1; i<n-1; ++i) {
            const auto& p = points[i];
            swc.emplace_back(kind::dendrite, ++id, p.x, p.y, p.z, p.r, parent);
            parent = id;
        }
        const auto& p = points.back();
        swc.emplace_back(seg.terminal? kind::end_point: kind::fork_point, ++id, p.x, p.y, p.z, p.r, parent);
        parent_end_id[seg.id] = id;
    }

    return swc;
}

int main(int argc, char** argv) {
    // options
    int n_morph = 1;
    int rng_seed = 0;
    bool set_rng_seed = false;
    bool emit_swc = false;
    std::string swc_file = "";

    try {
        auto arg = argv+1;
        while (*arg) {
            if (auto o = to::parse_opt<int>(arg, 'n', "count")) {
                n_morph = *o;
            }
            else if (auto o = to::parse_opt<std::string>(arg, 0, "swc")) {
                emit_swc = true;
                swc_file = *o;
            }
            else if (to::parse_opt<void>(arg, 'h', "help")) {
                std::cout << "Usage: " << argv[0] << " " << usage_str;
                return 0;
            }
            else {
                throw to::parse_opt_error(*arg, "unrecognized option");
            }
        }

        lsys_param P;
        std::minstd_rand g;
        if (set_rng_seed) {
            g.seed(rng_seed);
        }

        multi_file_writer swc_writer(swc_file,4);
        for (int i=0; i<n_morph; ++i) {
            auto morph = generate_morphology(P, g);
            if (emit_swc) {
                swc_writer.open(i);
                auto& stream = swc_writer.stream();

                std::vector<swc_record> swc = as_swc(morph);
                stream << "# lmorpho generated morphology\n# index: " << i << "\n";
                std::copy(swc.begin(), swc.end(), std::ostream_iterator<swc_record>(stream, "\n"));
            }

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


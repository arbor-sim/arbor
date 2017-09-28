#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <morphology.hpp>
#include <swcio.hpp>
#include <util/strprintf.hpp>

#include "morphio.hpp"

using arb::io::swc_record;
using arb::util::strprintf;

std::vector<swc_record> as_swc(const arb::morphology& morph);

// Multi-file manager implementation.
multi_file::multi_file(const std::string& pattern, int digits) {
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

void multi_file::open(unsigned n) {
    if (use_stdout_ || (file_.is_open() && (concat_ || n==current_n_))) {
        return;
    }

    if (file_.is_open()) file_.close();

    std::string fname = concat_? filename_: strprintf(fmt_, n);
    file_.open(fname);

    current_n_ = n;
}

// SWC transform

using arb::io::swc_record;

// TODO: Move this functionality to arbor library.
std::vector<swc_record> as_swc(const arb::morphology& morph) {
    using kind = swc_record::kind;
    std::map<int, int> parent_end_id;
    std::vector<swc_record> swc;

    // soma
    const auto &p = morph.soma;
    int id = 0;
    parent_end_id[0] = 0;
    swc.emplace_back(kind::soma, id, p.x, p.y, p.z, p.r, -1);

    // dendrites:
    for (auto& sec: morph.sections) {
        int parent = parent_end_id[sec.parent_id];

        const auto& points = sec.points;
        auto n = points.size();
        if (n<2) {
            throw std::runtime_error(strprintf("surprisingly short cable: id=%d, size=%ul", sec.id, n));
        }

        // Include first point only for dendrites segments attached to soma.
        if (sec.parent_id==0) {
            const auto& p = points[0];
            ++id;
            swc.emplace_back(kind::fork_point, id, p.x, p.y, p.z, p.r, parent);
            parent = id;
        }

        // Interior points.
        for (unsigned i = 1; i<n-1; ++i) {
            const auto& p = points[i];
            ++id;
            swc.emplace_back(kind::dendrite, id, p.x, p.y, p.z, p.r, parent);
            parent = id;
        }

        // Final point (fork or terminal).
        const auto& p = points.back();
        ++id;
        swc.emplace_back(sec.terminal? kind::end_point: kind::fork_point, id, p.x, p.y, p.z, p.r, parent);
        parent_end_id[sec.id] = id;
    }

    return swc;
}

// SWC emitter implementation.

void swc_emitter::operator()(unsigned index, const arb::morphology& m) {
    file_.open(index);
    auto& stream = file_.stream();

    auto swc = as_swc(m);
    stream << "# lmorpho generated morphology\n# index: " << index << "\n";
    std::copy(swc.begin(), swc.end(), std::ostream_iterator<swc_record>(stream, "\n"));
}

// pvector emitter implementation.

std::vector<int> as_pvector(const arb::morphology& morph, unsigned offset) {
    std::map<int, unsigned> parent_index; // section id to segment index
    std::vector<int> pvec;
    unsigned index = offset; // starting segment index

    // soma
    parent_index[0] = index;
    pvec.push_back(-1);
    ++index;

    // dendrites:
    for (auto& sec: morph.sections) {
        int parent = parent_index[sec.parent_id];

        auto n = sec.points.size();
        if (n<2) {
            throw std::runtime_error(strprintf("surprisingly short cable: id=%d, size=%ul", sec.id, n));
        }

        for (unsigned i = 1; i<n; ++i) {
            pvec.push_back(parent);
            parent = index++;
        }

        parent_index[sec.id] = parent;
    }

    return pvec;
}

void pvector_emitter::operator()(unsigned index, const arb::morphology& m) {
    auto pvec = as_pvector(m, offset_);
    if (coalesce_) offset_ += pvec.size();

    file_.open(index);
    auto& stream = file_.stream();
    std::copy(pvec.begin(), pvec.end(), std::ostream_iterator<int>(stream, "\n"));
}


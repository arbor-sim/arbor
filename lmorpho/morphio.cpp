#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <arbor/morphology.hpp>
#include <arbor/swcio.hpp>

#include "morphio.hpp"

using arb::swc_record;

std::vector<swc_record> as_swc(const arb::morphology& morph);

// Multi-file manager implementation.
multi_file::multi_file(const std::string& pattern, int digits) {
    auto npos = std::string::npos;

    file_.exceptions(std::ofstream::failbit);
    concat_ = (pattern.find("%")==npos);
    use_stdout_ = pattern.empty() || pattern=="-";

    if (!concat_) {
        auto p = pattern.find("%");
        fmt_prefix_ = pattern.substr(0, p);
        fmt_suffix_ = pattern.substr(p+1);
        fmt_digits_ = digits;
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

    std::string fname;
    if (concat_) {
        fname = filename_;
    }
    else {
        std::stringstream ss;
        ss << fmt_prefix_ << std::setfill('0') << std::setw(fmt_digits_) << n << fmt_suffix_;
        fname = ss.str();
    }

    file_.open(fname);

    current_n_ = n;
}

static std::string short_cable_message(int id, unsigned sz) {
    std::stringstream ss;
    ss << "surprisingly short cable: id=" << id << ", size=" << sz;
    return ss.str();
}

// SWC transform

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
            throw std::runtime_error(short_cable_message(sec.id, n));
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
            throw std::runtime_error(short_cable_message(sec.id, n));
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


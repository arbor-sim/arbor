#include <cmath>
#include <ios>
#include <limits>
#include <numeric>
#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/morph/segment_tree.hpp>

#include "arbor/morph/primitives.hpp"

#include <arborio/swcio.hpp>

namespace arborio {

// SWC exceptions:

swc_error::swc_error(const std::string& msg, int record_id):
    arbor_exception(msg+": sample id "+std::to_string(record_id)),
    record_id(record_id)
{}

swc_no_such_parent::swc_no_such_parent(int record_id):
    swc_error("Missing SWC parent record", record_id)
{}

swc_record_precedes_parent::swc_record_precedes_parent(int record_id):
    swc_error("SWC parent id is not less than sample id", record_id)
{}

swc_duplicate_record_id::swc_duplicate_record_id(int record_id):
    swc_error("duplicate SWC sample id", record_id)
{}

swc_spherical_soma::swc_spherical_soma(int record_id):
    swc_error("SWC with spherical somata are not supported", record_id)
{}

swc_mismatched_tags::swc_mismatched_tags(int record_id):
    swc_error("Every record not attached to a soma sample must have the same tag as its parent", record_id)
{}

swc_unsupported_tag::swc_unsupported_tag(int record_id):
    swc_error("Unsupported SWC record identifier.", record_id)
{}

// Record I/O:

ARB_ARBORIO_API std::ostream& operator<<(std::ostream& out, const swc_record& record) {
    std::ios_base::fmtflags flags(out.flags());

    out.precision(std::numeric_limits<double>::digits10+2);
    out << record.id << ' ' << record.tag << ' '
        << record.x  << ' ' << record.y   << ' ' << record.z << ' ' << record.r << ' '
        << record.parent_id << '\n';

    out.flags(flags);

    return out;
}

ARB_ARBORIO_API std::istream& operator>>(std::istream& in, swc_record& record) {
    std::string line;
    if (!getline(in, line, '\n')) return in;

    swc_record r;
    std::istringstream s(line);
    s >> r.id >> r.tag >> r.x >> r.y >> r.z >> r.r >> r.parent_id;
    if (s) {
        record = r;
    }
    else {
        in.setstate(std::ios_base::failbit);
    }

    return in;
}

// Parse SWC format data (comments and sequence of SWC records).

static std::vector<swc_record> sort_and_validate_swc(std::vector<swc_record> records) {
    if (records.empty()) return {};

    std::unordered_set<int> seen;
    std::size_t n_rec = records.size();

    for (std::size_t i = 0; i<n_rec; ++i) {
        swc_record& r = records[i];

        if (r.parent_id>=r.id) {
            throw swc_record_precedes_parent(r.id);
        }

        if (!seen.insert(r.id).second) {
            throw swc_duplicate_record_id(r.id);
        }
    }

    std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; });

    for (std::size_t i = 0; i<n_rec; ++i) {
        const swc_record& r = records[i];
        if ((i==0 && r.parent_id!=-1) || (i>0 && !seen.count(r.parent_id))) {
            throw swc_no_such_parent(r.id);
        }
    }

    return records;
}

// swc_data
swc_data::swc_data(std::vector<arborio::swc_record> recs) :
    metadata_(),
    records_(sort_and_validate_swc(std::move(recs))) {};

swc_data::swc_data(std::string meta, std::vector<arborio::swc_record> recs) :
    metadata_(meta),
    records_(sort_and_validate_swc(std::move(recs))) {};

// Parse and validate swc data

ARB_ARBORIO_API swc_data parse_swc(std::istream& in) {
    // Collect any initial comments (lines beginning with '#').

    std::string metadata;
    std::vector<swc_record> records;
    std::string line;

    while (in) {
        auto c = in.get();
        if (c=='#') {
            getline(in, line, '\n');
            auto from = line.find_first_not_of(" \t");
            if (from != std::string::npos) {
                metadata.append(line, from);
            }
            metadata += '\n';
        }
        else {
            in.unget();
            break;
        }
    }

    swc_record r;
    while (in && (in.peek() != '\n') && in >> r) {
        records.push_back(r);
    }

    return swc_data(metadata, std::move(records));
}

ARB_ARBORIO_API swc_data parse_swc(const std::string& text) {
    std::istringstream is(text);
    return parse_swc(is);
}

ARB_ARBORIO_API arb::segment_tree load_swc_arbor_raw(const swc_data& data) {
    const auto& records = data.records();

    if (records.empty())  return {};
    if (records.size()<2) throw swc_spherical_soma(records[0].tag);

    arb::segment_tree tree;
    std::size_t n_seg = records.size()-1;
    tree.reserve(n_seg);

    std::unordered_map<int, std::size_t> id_to_index;
    id_to_index[records[0].id] = 0;

    // Check whether the first sample has at least one child with the same tag
    bool first_tag_match = false;
    int first_id = records[0].id;
    int first_tag = records[0].tag;

    // ith segment is built from i+1th SWC record and its parent.
    for (std::size_t i = 1; i<n_seg+1; ++i) {
        const auto& dist = records[i];
        first_tag_match |= dist.parent_id==first_id && dist.tag==first_tag;

        auto iter = id_to_index.find(dist.parent_id);
        if (iter==id_to_index.end()) throw swc_no_such_parent{dist.id};
        auto parent_idx = iter->second;

        const auto& prox = records[parent_idx];
        arb::msize_t seg_parent = parent_idx? parent_idx-1: arb::mnpos;

        tree.append(seg_parent,
            arb::mpoint{prox.x, prox.y, prox.z, prox.r},
            arb::mpoint{dist.x, dist.y, dist.z, dist.r},
            dist.tag);

        id_to_index[dist.id] = i;
    }

    if (!first_tag_match) {
        throw swc_spherical_soma(first_id);
    }

    return tree;
}

ARB_ARBORIO_API arb::segment_tree load_swc_neuron_raw(const swc_data& data) {
    constexpr int soma_tag = 1;

    const auto n_samples = data.records().size();

    if (n_samples==0) {
        return {};
    }

    // The NEURON interpretation is only applied when the cell has a soma.
    if (data.records()[0].tag != soma_tag) {
        const auto& R = data.records();
        // Search for other soma samples
        if (auto it=std::find_if(R.begin(), R.end(), [](auto& r) {return r.tag==soma_tag;}); it!=R.end()) {
            // The presence of a soma tag when there is a non-soma tag at the root
            // violates the requirement that the parent of a soma sample is also a
            // soma sample.
            throw swc_mismatched_tags(it->id);
        }

        return load_swc_arbor_raw(data);
    }

    // Make a copy of the records and canonicalise them.
    auto records = data.records();
    std::unordered_map<int, int> record_index = {{-1, -1}};
    std::vector<int> old_record_index(n_samples);

    for (std::size_t i=0; i<n_samples; ++i) {
        auto& r = records[i];
        record_index[r.id] = i;
        old_record_index[i] = r.id;
        r.id = i;
        if (!record_index.count(r.parent_id)) {
            throw swc_no_such_parent(r.parent_id);
        }
        r.parent_id = record_index[r.parent_id];
    }

    // Calculate meta-data
    std::vector<int> child_count(n_samples);
    std::size_t n_soma_samples = 0;

    for (std::size_t i=0; i<n_samples; ++i) {
        auto& r = records[i];
        // Only accept soma, axon, dend and apic samples.
        if (!(r.tag>=0 && r.tag<=4)) {
            throw swc_unsupported_tag(old_record_index[i]);
        }
        if (r.tag==soma_tag) {
            ++n_soma_samples;
        }
        int pid = r.parent_id;
        if (pid!=-1) {
            ++child_count[pid];
            const int ptag = records[pid].tag;
            // Assert that sample has the same tag as its parent, or the parent is tagged soma.
            if (r.tag!=ptag && ptag!=soma_tag) {
                throw swc_mismatched_tags(old_record_index[i]);
            }
        }
    }

    const bool spherical_soma = n_soma_samples==1;

    // Construct the segment tree.
    arb::segment_tree tree;

    // It isn't possible to a-priori calculate the exact number of segments
    // without more meta-data, but this will be accurate in the absence of
    // single-sample sub-trees, which should be rare.
    tree.reserve(n_samples+spherical_soma);

    std::unordered_map<int, arb::msize_t> segmap;

    // Construct a soma composed of two cylinders if is represented by a single sample.
    if (spherical_soma) {
        auto& sr = records[0];
        arb::msize_t pid = arb::mnpos;
        pid = tree.append(pid, {sr.x-sr.r, sr.y, sr.z, sr.r},
                               {sr.x,      sr.y, sr.z, sr.r}, soma_tag);
        // Children of the soma sample attach to the distal end of the first segment in the soma.
        segmap[0] = pid;
        pid = tree.append(pid, {sr.x,      sr.y, sr.z, sr.r},
                               {sr.x+sr.r, sr.y, sr.z, sr.r}, soma_tag);
    }
    else {
        segmap[0] = arb::mnpos;
    }

    // Build the tree.
    for (std::size_t i=1; i<n_samples; ++i) {
        auto& r = records[i];
        const auto pid = r.parent_id;
        auto& p = records[pid];
        const int tag = r.tag;

        // Constructing a segment inside the soma or a sub-tree.
        if (tag==p.tag) {
            segmap[i] = tree.append(segmap.at(pid), {p.x, p.y, p.z, p.r}, {r.x, r.y, r.z, r.r}, tag);
        }
        // The start of a sub-tree.
        else if (child_count[i]) {
            // Do not create a segment, instead set up the segmap so that the
            // first segment in the sub-tree will be connected to the soma with
            // a "zero resistance cable".
            segmap[i] = segmap.at(pid);
        }
        // Sub-tree defined with a single sample.
        else {
            // The sub-tree is composed of a single segment connecting the soma
            // to the sample, with constant radius defined by the sample.
            segmap[i] = tree.append(segmap.at(pid), {p.x, p.y, p.z, r.r}, {r.x, r.y, r.z, r.r}, tag);
        }
    }

    return tree;
}

ARB_ARBORIO_API arb::morphology load_swc_neuron(const swc_data& data) { return {load_swc_neuron_raw(data)}; }
ARB_ARBORIO_API arb::morphology load_swc_arbor(const swc_data& data) { return {load_swc_arbor_raw(data)}; }

} // namespace arborio


#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/morph/segment_tree.hpp>
#include <arbor/swcio.hpp>

#include "io/save_ios.hpp"
#include "util/rangeutil.hpp"

namespace arb {

// SWC exceptions:

swc_error::swc_error(const std::string& msg, int record_id):
    arbor_exception(msg+": sample id "+std::to_string(record_id)),
    record_id(record_id)
{}

swc_no_such_parent::swc_no_such_parent(int record_id):
    swc_error("missing SWC parent record", record_id)
{}

swc_record_precedes_parent::swc_record_precedes_parent(int record_id):
    swc_error("SWC parent id is not less than sample id", record_id)
{}

swc_duplicate_record_id::swc_duplicate_record_id(int record_id):
    swc_error("duplicate SWC sample id", record_id)
{}

swc_irregular_id::swc_irregular_id(int record_id):
    swc_error("SWC record id not numbered consecutively from 1", record_id)
{}

swc_spherical_soma::swc_spherical_soma(int record_id):
    swc_error("SWC with spherical somata are not supported", record_id)
{}

bad_swc_data::bad_swc_data(int record_id):
    swc_error("Cannot interpret bad SWC data", record_id)
{}

swc_no_soma::swc_no_soma(int record_id):
    swc_error("No soma found at the root", record_id)
{}

swc_non_consecutive_soma::swc_non_consecutive_soma (int record_id):
    swc_error("Soma samples (tag 1) are not all listed consecutively", record_id)
{}

swc_non_serial_soma::swc_non_serial_soma (int record_id):
    swc_error("Soma samples (tag 1) are not listed serially", record_id)
{}

swc_branchy_soma::swc_branchy_soma (int record_id):
    swc_error("Non-soma sample (tag >= 1) connected to a non-distal sample of the soma", record_id)
{}


// Record I/O

std::ostream& operator<<(std::ostream& out, const swc_record& record) {
    io::save_ios_flags save(out);

    out.precision(std::numeric_limits<double>::digits10+2);
    return out << record.id << ' ' << record.tag << ' '
               << record.x  << ' ' << record.y   << ' ' << record.z << ' ' << record.r << ' '
               << record.parent_id << '\n';
}

std::istream& operator>>(std::istream& in, swc_record& record) {
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

static std::vector<swc_record> sort_and_validate_swc(std::vector<swc_record> records, swc_mode mode) {
    if (records.empty()) return {};

    std::unordered_set<int> seen;
    std::size_t n_rec = records.size();
    int first_id = records[0].id;
    int first_tag = records[0].tag;

    if (records.size()<2) {
        throw swc_spherical_soma(first_id);
    }

    for (std::size_t i = 0; i<n_rec; ++i) {
        swc_record& r = records[i];

        if (r.parent_id>=r.id) {
            throw swc_record_precedes_parent(r.id);
        }

        if (mode==swc_mode::strict && r.id != (int)i+1) {
            throw swc_irregular_id(r.id);
        }

        if (!seen.insert(r.id).second) {
            throw swc_duplicate_record_id(r.id);
        }
    }

    util::sort_by(records, [](auto& r) { return r.id; });
    bool first_tag_match = false;

    for (std::size_t i = 0; i<n_rec; ++i) {
        const swc_record& r = records[i];
        first_tag_match |= r.parent_id==first_id && r.tag==first_tag;

        if ((i==0 && r.parent_id!=-1) || (i>0 && !seen.count(r.parent_id))) {
            throw swc_no_such_parent(r.id);
        }
    }

    if (mode==swc_mode::strict && !first_tag_match) {
        throw swc_spherical_soma(first_id);
    }

    return records;
}

swc_data parse_swc(std::istream& in, swc_mode mode) {
    // Collect any initial comments (lines beginning with '#').

    swc_data data;
    std::string line;

    while (in) {
        auto c = in.get();
        if (c=='#') {
            getline(in, line, '\n');
            auto from = line.find_first_not_of(" \t");
            if (from != std::string::npos) {
                data.metadata.append(line, from);
            }
            data.metadata += '\n';
        }
        else {
            in.unget();
            break;
        }
    }

    swc_record r;
    while (in && in >> r) {
        data.records.push_back(r);
    }

    data.records = sort_and_validate_swc(std::move(data.records), mode);
    return data;
}

swc_data parse_swc(const std::string& text, swc_mode mode) {
    std::istringstream is(text);
    return parse_swc(is, mode);
}

swc_data parse_swc(std::vector<swc_record> records, swc_mode mode) {
    swc_data data;
    data.records = sort_and_validate_swc(std::move(records), mode);
    return data;
}

segment_tree as_segment_tree(const std::vector<swc_record>& records) {
    if (records.empty()) return {};
    if (records.size()<2) throw bad_swc_data{records.front().id};

    segment_tree tree;
    std::size_t n_seg = records.size()-1;
    tree.reserve(n_seg);

    std::unordered_map<int, std::size_t> id_to_index;
    id_to_index[records[0].id] = 0;

    // ith segment is built from i+1th SWC record and its parent.
    for (std::size_t i = 1; i<n_seg+1; ++i) {
        const auto& dist = records[i];

        auto iter = id_to_index.find(dist.parent_id);
        if (iter==id_to_index.end()) throw bad_swc_data{dist.id};
        auto parent_idx = iter->second;

        const auto& prox = records[parent_idx];
        msize_t seg_parent = parent_idx? parent_idx-1: mnpos;

        tree.append(seg_parent,
            mpoint{prox.x, prox.y, prox.z, prox.r},
            mpoint{dist.x, dist.y, dist.z, dist.r},
            dist.tag);

        id_to_index[dist.id] = i;
    }

    return tree;
}

arb::segment_tree load_swc_neuron(const std::vector<swc_record>& records) {
    if (records.empty()) return {};

    auto soma_prox = records.front();

    // Assert that root sample has tag 1.
    if (soma_prox.tag != 1) {
        throw swc_no_soma{soma_prox.id};
    }

    // Map of SWC record id to index in `records`.
    std::unordered_map<int, std::size_t> record_index;
    record_index[soma_prox.id] = 0;

    // Vector of records that make up the soma
    std::vector<swc_record> soma_records = {soma_prox};
    int prev_tag = soma_prox.tag;
    int prev_id  = soma_prox.id;

    for (std::size_t i = 1; i<records.size(); ++i) {
        const auto& r = records[i];
        record_index[r.id] = i;

        if (r.tag == soma_prox.tag && prev_tag != soma_prox.tag) {
            throw swc_non_consecutive_soma{r.id};
        }

        if (r.tag == soma_prox.tag) {
            if (r.parent_id != prev_id) {
                throw swc_non_serial_soma{r.id};
            }
            soma_records.push_back(r);
        }

        if (auto p = r.parent_id; r.tag != 1 && records[record_index[p]].tag == 1 && p != soma_records.back().id) {
            throw swc_branchy_soma{r.id};
        }

        prev_tag = r.tag;
        prev_id = r.id;
    }

    segment_tree tree;

    std::unordered_map<int, msize_t> smap; // SWC record id -> segment id
    // First, construct the soma
    if (soma_records.size() == 1) {
        // Model the soma as a 2 cylinders with total length=2*radius, extended along the y axis
        auto p = tree.append(mnpos, {soma_prox.x, soma_prox.y - soma_prox.r, soma_prox.z, soma_prox.r},
                             {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r}, 1);
        tree.append(p, {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r},
                    {soma_prox.x, soma_prox.y + soma_prox.r, soma_prox.z, soma_prox.r}, 1);
        smap[soma_prox.id] = p;

    } else {
        // Calculate segment lengths
        std::vector<double> soma_segment_lengths;
        for (std::size_t i = 1; i < soma_records.size(); ++i) {
            const auto& p0 = soma_records[i-1];
            const auto& p1 = soma_records[i];
            soma_segment_lengths.push_back(distance(mpoint(p0.x, p0.y, p0.z, p0.r), mpoint(p1.x, p1.y, p1.z, p1.r)));
        }
        double midlength = std::accumulate(soma_segment_lengths.begin(), soma_segment_lengths.end(), 0)/2;

        std::size_t idx = 0;
        for (; idx < soma_segment_lengths.size(); ++idx) {
            auto l = soma_segment_lengths[idx];
            if (midlength > l) {
                midlength -= l;
                continue;
            }
            break;
        }

        // Interpolate along the segment that contains the midpoint of the soma
        double pos_on_segment = midlength/soma_segment_lengths[idx];

        auto& r0 = soma_records[idx];
        auto& r1 = soma_records[idx+1];

        auto x = r0.x + pos_on_segment*(r1.x-r0.x);
        auto y = r0.y + pos_on_segment*(r1.y-r0.y);
        auto z = r0.z + pos_on_segment*(r1.z-r0.z);
        auto r = r0.r + pos_on_segment*(r1.r-r0.r);

        mpoint mid_soma = {x, y, z, r};

        // Construct the soma
        msize_t parent = mnpos;
        for (std::size_t i = 0; i < idx; ++i) {
            const auto& p0 = soma_records[i];
            const auto& p1 = soma_records[i+1];
            parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
        }
        auto soma_seg = tree.append(parent, {r0.x, r0.y, r0.z, r0.r}, mid_soma, 1);

        parent = tree.append(soma_seg, mid_soma, {r1.x, r1.y, r1.z, r1.r}, 1);

        for (std::size_t i = idx + 1; i < soma_records.size(); ++i ) {
            const auto& p0 = soma_records[i];
            const auto& p1 = soma_records[i+1];
            parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
        }

        smap[soma_records.back().id] = soma_seg;
    }

    // Build branches off soma.
    for (const auto& r: records) {
        // Skip the soma samples
        if (r.tag==1) continue;

        // If the sample has a soma sample as its parent don't create a segment.
        if (records[r.parent_id].tag == 1) {
            // Map the sample id to the segment id of the soma
            smap[r.id] = smap[r.parent_id];
            continue;
        }

        const auto p = r.parent_id;
        const auto& prox = records[record_index[p]];
        smap[r.id] = tree.append(smap.at(p), {prox.x, prox.y, prox.z, prox.r}, {r.x, r.y, r.z, r.r}, r.tag);
    }

    return tree;
}

} // namespace arb


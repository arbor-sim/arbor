#include <cmath>
#include <ios>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <arbor/morph/segment_tree.hpp>

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

swc_non_spherical_soma::swc_non_spherical_soma(int record_id):
    swc_error("SWC with multi-sample somata are not supported", record_id)
{}

swc_no_soma::swc_no_soma(int record_id):
    swc_error("No soma (tag 1) found at the root", record_id)
{}

swc_non_consecutive_soma::swc_non_consecutive_soma (int record_id):
    swc_error("Soma samples (tag 1) are not listed consecutively", record_id)
{}

swc_non_serial_soma::swc_non_serial_soma (int record_id):
    swc_error("Soma samples (tag 1) are not listed serially", record_id)
{}

swc_branchy_soma::swc_branchy_soma (int record_id):
    swc_error("Non-soma sample (tag >= 1) connected to a non-distal sample of the soma", record_id)
{}

swc_collocated_soma::swc_collocated_soma(int record_id):
    swc_error("The samples that make the soma (tag 1) are not allowed to be collocated", record_id)
{}

swc_single_sample_segment::swc_single_sample_segment(int record_id):
    swc_error("Segments connected to the soma (tag 1) must have 2 samples with the same tag", record_id)
{}

swc_mismatched_tags::swc_mismatched_tags(int record_id):
    swc_error("Every record not attached to the soma (tag 1) must have the same tag as its parent", record_id)
{}

swc_unsupported_tag::swc_unsupported_tag(int record_id):
    swc_error("Every record must have a tag of 2, 3 or 4, except for the first which must have tag 1", record_id)
{}

swc_unsupported_gaps::swc_unsupported_gaps(int record_id):
    swc_error("No gaps are allowed between the soma and any axons, dendrites or apical dendrites", record_id)
{}

swc_bad_description::swc_bad_description(int record_id):
    swc_error("Need at least 2 samples to form a segment", record_id)
{}


// Record I/O:

std::ostream& operator<<(std::ostream& out, const swc_record& record) {
    std::ios_base::fmtflags flags(out.flags());

    out.precision(std::numeric_limits<double>::digits10+2);
    out << record.id << ' ' << record.tag << ' '
        << record.x  << ' ' << record.y   << ' ' << record.z << ' ' << record.r << ' '
        << record.parent_id << '\n';

    out.flags(flags);

    return out;
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

swc_data parse_swc(std::istream& in) {
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

swc_data parse_swc(const std::string& text) {
    std::istringstream is(text);
    return parse_swc(is);
}

arb::morphology load_swc_arbor(const swc_data& data) {
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

    return arb::morphology(tree);
}

arb::morphology load_swc_neuron(const swc_data& data) {
    const auto& records = data.records();

    // Assert that the file contains at least one sample.
    if (records.empty()) return {};

    const int soma_tag = 1;
    auto soma_prox = records.front();

    // Assert that root sample has tag 1.
    if (soma_prox.tag != soma_tag) throw swc_no_soma{soma_prox.id};

    // check for single soma cell
    bool has_children = false;

    // Map of SWC record id to index in `records`.
    std::unordered_map<int, std::size_t> record_index;
    record_index[soma_prox.id] = 0;

    // Vector of records that make up the soma
    std::vector<swc_record> soma_records = {soma_prox};
    int prev_tag = soma_prox.tag;
    int prev_id = soma_prox.id;

    // Preliminary error checking and building the record_index
    for (std::size_t i = 1; i < records.size(); ++i) {
        const auto& r = records[i];
        record_index[r.id] = i;

        if (r.tag == soma_tag) {
            if (prev_tag != soma_tag)   throw swc_non_consecutive_soma{r.id};
            if (prev_id != r.parent_id) throw swc_non_serial_soma{r.id};
            soma_records.push_back(r);
        }

        // Find record index of the parent
        auto parent_iter = record_index.find(r.parent_id);

        if (parent_iter == record_index.end() || records[parent_iter->second].id == r.id) throw swc_no_such_parent{r.id};

        if (r.tag != soma_tag && records[parent_iter->second].tag == soma_tag) {
            if (r.parent_id != soma_records.back().id) throw swc_branchy_soma{r.id};
            has_children = true;
        }

        prev_tag = r.tag;
        prev_id = r.id;
    }

    arb::segment_tree tree;
    tree.reserve(records.size());

    // Map of SWC record id to index in `tree`.
    std::unordered_map<int, arb::msize_t> tree_index;

    // First, construct the soma
    if (soma_records.size() == 1) {
        if (!has_children) {
            // Model the soma as a 1 cylinder with total length=2*radius, extended along the y axis
            tree.append(arb::mnpos, {soma_prox.x, soma_prox.y - soma_prox.r, soma_prox.z, soma_prox.r},
                        {soma_prox.x, soma_prox.y + soma_prox.r, soma_prox.z, soma_prox.r}, soma_tag);
            return tree;
        }
        // Model the soma as a 2 cylinders with total length=2*radius, extended along the y axis, centered at the sample
        auto p = tree.append(arb::mnpos, {soma_prox.x, soma_prox.y - soma_prox.r, soma_prox.z, soma_prox.r},
                             {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r}, soma_tag);
        tree.append(p, {soma_prox.x, soma_prox.y, soma_prox.z, soma_prox.r},
                    {soma_prox.x, soma_prox.y + soma_prox.r, soma_prox.z, soma_prox.r}, soma_tag);
        tree_index[soma_prox.id] = p;
    }
    else {
        if (!has_children) {
            // Don't split soma at the midpoint
            arb::msize_t parent = arb::mnpos;
            bool collocated_samples = true;
            for (std::size_t i = 0; i < soma_records.size() - 1; ++i) {
                const auto& p0 = soma_records[i];
                const auto& p1 = soma_records[i + 1];
                parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
                collocated_samples &= ((p0.x == p1.x) && (p0.y == p1.y) && (p0.z == p1.z));
            }
            if (collocated_samples) {
                throw swc_collocated_soma{records[0].id};
            }
            return tree;
        }
        // Calculate segment lengths
        bool collocated_samples = true;
        std::vector<double> soma_segment_lengths;
        for (std::size_t i = 0; i < soma_records.size() - 1; ++i) {
            const auto& p0 = soma_records[i];
            const auto& p1 = soma_records[i + 1];
            soma_segment_lengths.push_back(distance(arb::mpoint{p0.x, p0.y, p0.z, p0.r}, arb::mpoint{p1.x, p1.y, p1.z, p1.r}));
            collocated_samples &= ((p0.x == p1.x) && (p0.y == p1.y) && (p0.z == p1.z));
        }
        if (collocated_samples) {
            throw swc_collocated_soma{records[0].id};
        }
        double midlength = std::accumulate(soma_segment_lengths.begin(), soma_segment_lengths.end(), 0.) / 2;

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
        double pos_on_segment = midlength / soma_segment_lengths[idx];

        auto& r0 = soma_records[idx];
        auto& r1 = soma_records[idx + 1];

        auto x = r0.x + pos_on_segment * (r1.x - r0.x);
        auto y = r0.y + pos_on_segment * (r1.y - r0.y);
        auto z = r0.z + pos_on_segment * (r1.z - r0.z);
        auto r = r0.r + pos_on_segment * (r1.r - r0.r);

        arb::mpoint mid_soma = {x, y, z, r};

        // Construct the soma
        arb::msize_t parent = arb::mnpos;
        for (std::size_t i = 0; i < idx; ++i) {
            const auto& p0 = soma_records[i];
            const auto& p1 = soma_records[i + 1];
            parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
        }
        auto soma_seg = tree.append(parent, {r0.x, r0.y, r0.z, r0.r}, mid_soma, 1);

        arb::mpoint r1_p{r1.x, r1.y, r1.z, r1.r};
        parent = mid_soma != r1_p ? tree.append(soma_seg, mid_soma, r1_p, 1) : soma_seg;

        for (std::size_t i = idx + 1; i < soma_records.size() - 1; ++i) {
            const auto& p0 = soma_records[i];
            const auto& p1 = soma_records[i + 1];
            parent = tree.append(parent, {p0.x, p0.y, p0.z, p0.r}, {p1.x, p1.y, p1.z, p1.r}, 1);
        }

        tree_index[soma_records.back().id] = soma_seg;
    }

    // Build branches off soma.
    std::set<int> unused_samples; // Samples that are not part of a segment
    for (const auto& r: records) {
        // Skip the soma samples
        if (r.tag == soma_tag) continue;

        const auto p = r.parent_id;

        // Find parent segment of the record
        auto pseg_iter = tree_index.find(p);
        if (pseg_iter == tree_index.end()) throw swc_no_such_parent{r.id};

        // Find parent record of the record
        auto prec_iter = record_index.find(p);
        if (prec_iter == record_index.end() || records[prec_iter->second].id == r.id) throw swc_no_such_parent{r.id};

        // If the sample has a soma sample as its parent don't create a segment.
        if (records[prec_iter->second].tag == soma_tag) {
            // Map the sample id to the segment id of the soma (parent)
            tree_index[r.id] = pseg_iter->second;
            unused_samples.insert(r.id);
            continue;
        }

        const auto& prox = records[prec_iter->second];
        tree_index[r.id] = tree.append(pseg_iter->second, {prox.x, prox.y, prox.z, prox.r}, {r.x, r.y, r.z, r.r}, r.tag);
        unused_samples.erase(prox.id);
    }

    if (!unused_samples.empty()) {
        throw swc_single_sample_segment(*unused_samples.begin());
    }
    return arb::morphology(tree);
}

arb::morphology load_swc_allen(const swc_data& data, bool no_gaps) {
    auto records = data.records();

    // Assert that the file contains at least one sample.
    if (records.empty()) return {};

    // Map of SWC record id to index in `records`.
    std::unordered_map<int, std::size_t> record_index;

    int soma_id = records[0].id;
    record_index[soma_id] = 0;

    // Assert that root sample has tag 1.
    if (records[0].tag != 1) {
        throw swc_no_soma{records[0].id};
    }

    for (std::size_t i = 1; i < records.size(); ++i) {
        const auto& r = records[i];
        record_index[r.id] = i;

        // Find record index of the parent
        auto p = r.parent_id;
        auto parent_iter = record_index.find(p);

        if (parent_iter == record_index.end() || records[parent_iter->second].id == r.id)
        {
            throw swc_no_such_parent{r.id};
        }

        // Assert that all samples have the same tag as their parent, except those attached to the soma.
        if (p != soma_id && r.tag != records[parent_iter->second].tag) {
            throw swc_mismatched_tags{r.id};
        }

        // Assert that all non-root samples have a tag of 2, 3, or 4.
        if (r.tag < 2) throw swc_non_spherical_soma{r.id};
        if (r.tag > 4) throw swc_unsupported_tag{r.id};
    }

    // Translate the morphology so that the soma is centered at the origin (0,0,0)
    arb::mpoint sloc{records[0].x, records[0].y, records[0].z, records[0].r};
    for (auto& r: records) {
        r.x -= sloc.x;
        r.y -= sloc.y;
        r.z -= sloc.z;
    }

    arb::segment_tree tree;

    // Model the spherical soma as a cylinder with length=2*radius.
    // The cylinder is centred on the origin, and extended along the y axis.
    double soma_rad = sloc.radius;
    tree.append(arb::mnpos, {0, -soma_rad, 0, soma_rad}, {0, soma_rad, 0, soma_rad}, 1);

    // Build branches off soma.
    std::unordered_map<int, arb::msize_t> smap; // SWC record id -> segment id
    std::set<int> unused_samples;               // Samples that are not part of a segment
    for (const auto& r: records) {
        int id = r.id;
        if (id == soma_id) continue;

        // If sample i has the root as its parent don't create a segment.
        if (r.parent_id == soma_id) {
            if (no_gaps) {
                // Assert that this branch starts on the "surface" of the spherical soma.
                auto d = std::fabs(soma_rad - std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z));
                if (d > 1e-3) { // 1 nm tolerance
                    throw swc_unsupported_gaps{r.id};
                }
            }
            // This maps axons and apical dendrites to soma.prox, and dendrites to soma.dist.
            smap[id] = r.tag == 3 ? 0 : arb::mnpos;
            unused_samples.insert(id);
            continue;
        }

        const auto p = r.parent_id;
        const auto& prox = records[record_index[p]];
        smap[id] = tree.append(smap.at(p), {prox.x, prox.y, prox.z, prox.r}, {r.x, r.y, r.z, r.r}, r.tag);
        unused_samples.erase(p);
    }

    if (!unused_samples.empty()) {
        throw swc_single_sample_segment{*unused_samples.begin()};
    }

    return arb::morphology(tree);
}
} // namespace arborio


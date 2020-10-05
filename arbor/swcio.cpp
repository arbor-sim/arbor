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

swc_spherical_soma::swc_spherical_soma(int record_id):
    swc_error("SWC with spherical somata are not supported", record_id)
{}

bad_swc_data::bad_swc_data(int record_id):
    swc_error("Cannot interpret bad SWC data", record_id)
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

} // namespace arb


#pragma once

#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <algorithms.hpp>
#include <morphology.hpp>
#include <point.hpp>
#include <util/debug.hpp>

namespace arb {
namespace io {

class swc_record {
public:
    using id_type = int;
    using coord_type = double;

    // More on SWC files: http://research.mssm.edu/cnic/swc.html
    enum class kind {
        undefined = 0,
        soma,
        axon,
        dendrite,
        apical_dendrite,
        fork_point,
        end_point,
        custom
    };

    kind type = kind::undefined; // record type
    id_type id = 0;              // record id
    coord_type x = 0;            // record coordinates
    coord_type y = 0;
    coord_type z = 0;
    coord_type r = 0;            // record radius
    id_type parent_id= -1;      // record parent's id

    // swc records assume zero-based indexing; root's parent remains -1
    swc_record(swc_record::kind type, int id,
               coord_type x, coord_type y, coord_type z, coord_type r,
               int parent_id):
        type(type), id(id), x(x), y(y), z(z), r(r), parent_id(parent_id)
    {}

    swc_record() = default;
    swc_record(const swc_record& other) = default;
    swc_record& operator=(const swc_record& other) = default;

    bool operator==(const swc_record& other) const {
        return id == other.id &&
            x == other.x &&
            y == other.y &&
            z == other.z &&
            r == other.r &&
            parent_id == other.parent_id;
    }

    friend bool operator!=(const swc_record& lhs, const swc_record& rhs) {
        return !(lhs == rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const swc_record& record);

    coord_type diameter() const {
        return 2*r;
    }

    arb::point<coord_type> coord() const {
        return arb::point<coord_type>(x, y, z);
    }

    arb::section_point as_section_point() const {
        return arb::section_point{x, y, z, r};
    }

    // validity checks
    bool is_consistent() const;
    void assert_consistent() const; // throw swc_error if inconsistent.
};


class swc_error: public std::runtime_error {
public:
    explicit swc_error(const char* msg, std::size_t lineno = 0):
        std::runtime_error(msg), line_number(lineno)
    {}

    explicit swc_error(const std::string& msg, std::size_t lineno = 0):
        std::runtime_error(msg), line_number(lineno)
    {}

    std::size_t line_number;
};

// Parse one record, skipping comments and blank lines.
std::istream& operator>>(std::istream& is, swc_record& record);

// Parse and canonicalize an EOF-terminated sequence of records.
// Throw on parsing failure.
std::vector<swc_record> parse_swc_file(std::istream& is);

// Convert a canonical (see below) sequence of SWC records to a morphology object.
template <typename RandomAccessSequence>
morphology swc_as_morphology(const RandomAccessSequence& swc_records) {
    morphology morph;

    std::vector<swc_record::id_type> swc_parent_index;
    for (const auto& r: swc_records) {
        swc_parent_index.push_back(r.parent_id);
    }

    if (swc_parent_index.empty()) {
        return morph;
    }

    // The parent of soma must be 0, while in SWC files is -1
    swc_parent_index[0] = 0;
    auto branch_index = algorithms::branches(swc_parent_index); // partitions [0, #records] by branch.
    auto parent_branch_index = algorithms::tree_reduce(swc_parent_index, branch_index);

    // sanity check
    EXPECTS(parent_branch_index.size() == branch_index.size() - 1);

    // Add the soma first; then the segments
    const auto& soma = swc_records[0];
    morph.soma = { soma.x, soma.y, soma.z, soma.r };

    auto n_branches = parent_branch_index.size();
    for (std::size_t i = 1; i < n_branches; ++i) {
        auto b_start = std::next(swc_records.begin(), branch_index[i]);
        auto b_end   = std::next(swc_records.begin(), branch_index[i+1]);

        unsigned parent_id = parent_branch_index[i];
        std::vector<section_point> points;
        section_kind kind = section_kind::none;

        if (parent_id != 0) {
            // include the parent of current record if not branching from soma
            auto parent_record = swc_records[swc_parent_index[branch_index[i]]];

            points.push_back(section_point{parent_record.x, parent_record.y, parent_record.z, parent_record.r});
        }

        for (auto b = b_start; b!=b_end; ++b) {
            points.push_back(section_point{b->x, b->y, b->z, b->r});

            switch (b->type) {
            case swc_record::kind::axon:
                kind = section_kind::axon;
                break;
            case swc_record::kind::dendrite:
            case swc_record::kind::apical_dendrite:
                kind = section_kind::dendrite;
                break;
            case swc_record::kind::soma:
                kind = section_kind::soma;
                break;
            default: ; // stick with what we have
            }
        }

        morph.add_section(std::move(points), parent_id, kind);
    }

    morph.assert_valid();
    return morph;
}

// Given a random-access mutable sequence of `swc_record` describing
// a single morphology, check for consistency and renumber records
// so that ids are contiguous within branches, have no gaps, and
// are ordered with repect to parent indices.
template <typename RandomAccessSequence>
void swc_canonicalize_sequence(RandomAccessSequence& swc_records) {
    std::unordered_set<swc_record::id_type> ids;

    std::size_t         num_trees = 0;
    swc_record::id_type last_id   = -1;
    bool                needsort  = false;

    for (const auto& r: swc_records) {
        r.assert_consistent();

        if (r.parent_id == -1 && ++num_trees > 1) {
            // only a single tree is allowed
            throw swc_error("multiple trees found in SWC record sequence");
        }
        if (ids.count(r.id)) {
            throw swc_error("records with duplicated ids in SWC record sequence");
        }

        if (!needsort && r.id < last_id) {
            needsort = true;
        }

        last_id = r.id;
        ids.insert(r.id);
    }

    if (needsort) {
        std::sort(std::begin(swc_records), std::end(swc_records),
            [](const swc_record& a, const swc_record& b) { return a.id<b.id; });
    }

    // Renumber records if necessary
    std::map<swc_record::id_type, swc_record::id_type> idmap;
    swc_record::id_type next_id = 0;
    for (auto& r: swc_records) {
        if (r.id != next_id) {
            auto old_id = r.id;
            r.id = next_id;

            auto new_parent_id = idmap.find(r.parent_id);
            if (new_parent_id != idmap.end()) {
                r.parent_id = new_parent_id->second;
            }

            r.assert_consistent();
            idmap.insert(std::make_pair(old_id, next_id));
        }
        ++next_id;
    }

    // Reject if branches are not contiguously numbered
    std::vector<swc_record::id_type> parent_list = { 0 };
    for (std::size_t i = 1; i < swc_records.size(); ++i) {
        parent_list.push_back(swc_records[i].parent_id);
    }

    if (!arb::algorithms::has_contiguous_compartments(parent_list)) {
        throw swc_error("branches are not contiguously numbered", 0);
    }
}

} // namespace io
} // namespace arb

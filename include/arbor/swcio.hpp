#pragma once

#include <exception>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/morphology.hpp>
#include <arbor/point.hpp>

namespace arb {

struct swc_error: public arbor_exception {
    explicit swc_error(const std::string& msg, unsigned line_number = 0):
        arbor_exception(msg), line_number(line_number)
    {}
    unsigned line_number;
};

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


// Parse one record, skipping comments and blank lines.
std::istream& operator>>(std::istream& is, swc_record& record);

// Parse and canonicalize an EOF-terminated sequence of records.
// Throw on parsing failure.
std::vector<swc_record> parse_swc_file(std::istream& is);

// Convert a canonical (see below) vector of SWC records to a morphology object.
morphology swc_as_morphology(const std::vector<swc_record>& swc_records);

// Given a vector of random-access mutable sequence of `swc_record` describing
// a single morphology, check for consistency and renumber records
// so that ids are contiguous within branches, have no gaps, and
// are ordered with repect to parent indices.
void swc_canonicalize(std::vector<swc_record>& swc_records);

} // namespace arb

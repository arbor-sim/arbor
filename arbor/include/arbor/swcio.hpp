#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/segment_tree.hpp>

namespace arb {

// SWC exceptions are thrown by `parse_swc`, and correspond
// to inconsistent, or in `strict` mode, dubious SWC data.

struct swc_error: public arbor_exception {
    swc_error(const std::string& msg, int record_id);
    int record_id;
};

// Parent id in record has no corresponding SWC record,
// nor is the record the root record with parent id -1.
struct swc_no_such_parent: swc_error {
    explicit swc_no_such_parent(int record_id);
};

// Parent id is greater than or equal to record id.
struct swc_record_precedes_parent: swc_error {
    explicit swc_record_precedes_parent(int record_id);
};

// Multiple records cannot have the same id.
struct swc_duplicate_record_id: swc_error {
    explicit swc_duplicate_record_id(int record_id);
};

// Irregular record ordering.
struct swc_irregular_id: swc_error {
    explicit swc_irregular_id(int record_id);
};

// Smells like a spherical soma.
struct swc_spherical_soma: swc_error {
    explicit swc_spherical_soma(int record_id);
};

// Bad or inconsistent SWC data was fed to an `swc_data` consumer.
struct bad_swc_data: swc_error {
    explicit bad_swc_data(int record_id);
};

struct swc_record {
    int id = 0;          // sample number
    int tag = 0;         // structure identifier (tag)
    double x = 0;        // sample coordinates
    double y = 0;
    double z = 0;
    double r = 0;        // sample adius
    int parent_id= -1;   // record parent's sample number

    swc_record() = default;
    swc_record(int id, int tag, double x, double y, double z, double r, int parent_id):
        id(id), tag(tag), x(x), y(y), z(z), r(r), parent_id(parent_id)
    {}

    bool operator==(const swc_record& other) const {
        return id == other.id &&
            x == other.x &&
            y == other.y &&
            z == other.z &&
            r == other.r &&
            parent_id == other.parent_id;
    }

    bool operator!=(const swc_record& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream&, const swc_record&);
    friend std::istream& operator>>(std::istream&, swc_record&);
};

struct swc_data {
    std::string metadata;
    std::vector<swc_record> records;
};

// Read SWC records from stream, collecting any initial metadata represented
// in comments (stripping initial '#' and subsequent whitespace).
// Stops at EOF or after reading the first line that does not parse as SWC.
//
// Note that 'one-point soma' SWC files are explicitly not supported.
//
// In `relaxed` mode, it will check that:
//     * There are no duplicate record ids.
//     * All record ids are positive.
//     * There are no records whose parent id is not less than the record id.
//     * Only one record has parent id -1; all other parent ids correspond to records.
//     * There are at least two records.
//
// In `strict` mode, it will additionally check:
//     * Record ids are numbered contiguously from 1.
//     * The data cannot be interpreted as a 'spherical soma' SWC file.
//       Specifically, the root record shares its tag with at least one other
//       record with has the root as parent.
//
// Throws a corresponding exception of type derived from `swc_error` if any of the
// conditions above are encountered.
//
// SWC records are stored in id order.

enum class swc_mode { relaxed, strict };

swc_data parse_swc(std::istream&, swc_mode = swc_mode::strict);
swc_data parse_swc(const std::string& text, swc_mode mode = swc_mode::strict);

// Parse a series of existing SWC records.

swc_data parse_swc(std::vector<swc_record>, swc_mode = swc_mode::strict);

// Convert a valid, ordered sequence of SWC records to a morphological segment tree.

segment_tree as_segment_tree(const std::vector<swc_record>&);

inline segment_tree as_segment_tree(const swc_data& data) {
    return as_segment_tree(data.records);
}


} // namespace arb

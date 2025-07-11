#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

#include <arbor/arbexcept.hpp>

#include <arborio/loaded_morphology.hpp>
#include <arborio/export.hpp>

namespace arborio {

// SWC exceptions are thrown by `parse_swc`, and correspond
// to inconsistent, or in `strict` mode, dubious SWC data.

struct ARB_SYMBOL_VISIBLE swc_error: public arb::arbor_exception {
    swc_error(const std::string& msg, int record_id);
    int record_id;
};

// Parent id in record has no corresponding SWC record,
// nor is the record the root record with parent id -1.
struct ARB_SYMBOL_VISIBLE swc_no_such_parent: swc_error {
    explicit swc_no_such_parent(int record_id, int parent);
};

// Parent id is greater than or equal to record id.
struct ARB_SYMBOL_VISIBLE swc_record_precedes_parent: swc_error {
    explicit swc_record_precedes_parent(int record_id);
};

// Multiple records cannot have the same id.
struct ARB_SYMBOL_VISIBLE swc_duplicate_record_id: swc_error {
    explicit swc_duplicate_record_id(int record_id);
};

// Smells like a spherical soma.
struct ARB_SYMBOL_VISIBLE swc_spherical_soma: swc_error {
    explicit swc_spherical_soma(int record_id);
};

// Segment cannot have samples with different tags
struct ARB_SYMBOL_VISIBLE swc_mismatched_tags: swc_error {
    explicit swc_mismatched_tags(int record_id);
};

// Only tags 1, 2, 3, 4 supported
struct ARB_SYMBOL_VISIBLE swc_unsupported_tag: swc_error {
    explicit swc_unsupported_tag(int record_id, int tag);
};

struct ARB_ARBORIO_API swc_record {
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

struct ARB_ARBORIO_API swc_data {
private:
    std::string metadata_;
    std::vector<swc_record> records_;

public:
    swc_data() = delete;
    swc_data(std::vector<arborio::swc_record>, bool lax=false);
    swc_data(std::string, std::vector<arborio::swc_record>, bool lax=false);

    const std::vector<swc_record>& records() const {return records_;};
    std::string metadata() const {return metadata_;};
};

// Read SWC records from stream, collecting any initial metadata represented
// in comments (stripping initial '#' and subsequent whitespace).
// Stops at EOF or after reading the first line that does not parse as SWC.
//
// It will check that:
//     * There are no duplicate record ids.
//     * All record ids are positive.
//     * Only one record has parent id -1; all other parent ids correspond to records.
//     * The root record must share its tag with at least one other record
//       which has the root as parent. This implies that there must be at least
//       two SWC records.
//
// If lax_ordering is false it will also
//     * There are no records whose parent id is not less than the record id.
//
// Throws a corresponding exception of type derived from `swc_error` if any of the
// conditions above are encountered.
//
// SWC records are returned in id order.
ARB_ARBORIO_API swc_data parse_swc(std::istream&, bool lax_ordering=false);
ARB_ARBORIO_API swc_data parse_swc(const std::string&, bool lax_ordering=false);

// Convert a valid, ordered sequence of SWC records into a morphology.
//
// Note that 'one-point soma' SWC files are explicitly not supported.
//
// The segments of the generated morphology  will be contiguous. There will be
// one segment for each SWC record after the first: this record defines the tag
// and distal point of the segment, while the proximal point is taken from the
// parent record.
struct ARB_ARBORIO_API swc_loader_options {
   // allow skips in parent/child relations
   bool allow_non_monotonic_ids = false;
   // disable checking for tag congruence between parent and child
   bool allow_mismatched_tags = false;
   // supported tags
   std::map<int, std::string> tags = {{1, "soma"}, {2, "axon"}, {3, "dend"}, {4, "apic"}};
};

ARB_ARBORIO_API loaded_morphology load_swc_arbor(const swc_data& data);
ARB_ARBORIO_API loaded_morphology load_swc_arbor(const std::filesystem::path& fn);

// As above, will convert a valid, ordered sequence of SWC records into a morphology
//
// Note that 'one-point soma' SWC files are supported here
//
// Complies inferred SWC rules from NEURON, explicitly listed in the docs.
ARB_ARBORIO_API loaded_morphology load_swc_neuron(const swc_data& data, const swc_loader_options& opts={});
ARB_ARBORIO_API loaded_morphology load_swc_neuron(const std::filesystem::path& fn, const swc_loader_options& opts={});

} // namespace arborio

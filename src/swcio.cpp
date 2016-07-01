#include <algorithm>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <unordered_set>

#include "algorithms.hpp"
#include "point.hpp"
#include "swcio.hpp"
#include "util.hpp"

namespace nest {
namespace mc {
namespace io {

//
// swc_record implementation
//
void swc_record::renumber(id_type new_id, std::map<id_type, id_type> &idmap)
{
    auto old_id = id_;
    id_ = new_id;

    // Obtain parent_id from the map
    auto new_parent_id = idmap.find(parent_id_);
    if (new_parent_id != idmap.end()) {
        parent_id_ = new_parent_id->second;
    }

    check_consistency();
    idmap.insert(std::make_pair(old_id, new_id));
}

void swc_record::check_consistency() const
{
    // Check record type as well; enum's do not offer complete type safety,
    // since you can cast anything that fits to its underlying type
    if (static_cast<int>(type_) < 0 ||
        static_cast<int>(type_) > static_cast<int>(kind::custom)) {
        throw std::invalid_argument("unknown record type");
    }

    if (id_ < 0) {
        throw std::invalid_argument("negative ids not allowed");
    }

    if (parent_id_ < -1) {
        throw std::invalid_argument("parent_id < -1 not allowed");
    }

    if (parent_id_ >= id_) {
        throw std::invalid_argument("parent_id >= id is not allowed");
    }

    if (r_ < 0) {
        throw std::invalid_argument("negative radii are not allowed");
    }
}

std::istream &operator>>(std::istream &is, swc_record &record)
{
    swc_parser parser;
    parser.parse_record(is, record);
    return is;
}


std::ostream &operator<<(std::ostream &os, const swc_record &record)
{
    // output in one-based indexing
    os << record.id_+1 << " "
       << static_cast<int>(record.type_) << " "
       << std::setprecision(7) << record.x_ << " "
       << std::setprecision(7) << record.y_ << " "
       << std::setprecision(7) << record.z_ << " "
       << std::setprecision(7) << record.r_ << " "
       << ((record.parent_id_ == -1) ? record.parent_id_ : record.parent_id_+1);

    return os;
}


//
// Utility functions
//

bool starts_with(const std::string &str, const std::string &prefix)
{
    return (str.find(prefix) == 0);
}

void check_parse_status(const std::istream &is, const swc_parser &parser)
{
    if (is.fail()) {
        // If we try to read past the eof; fail bit will also be set
        throw swc_parse_error("could not parse value", parser.lineno());
    }
}

template<typename T>
T parse_value_strict(std::istream &is, const swc_parser &parser)
{
    T val;
    check_parse_status(is >> val, parser);

    // everything's fine
    return val;
}

// specialize parsing for record types
template<>
swc_record::kind parse_value_strict(std::istream &is, const swc_parser &parser)
{
    swc_record::id_type val;
    check_parse_status(is >> val, parser);

    // Let swc_record's constructor check for the type validity
    return static_cast<swc_record::kind>(val);
}

//
// swc_parser implementation
//

std::istream &swc_parser::parse_record(std::istream &is, swc_record &record)
{
    while (!is.eof() && !is.bad()) {
        // consume empty and comment lines first
        std::getline(is, linebuff_);

        ++lineno_;
        if (!linebuff_.empty() &&
            !starts_with(linebuff_, comment_prefix_) &&
            !starts_with(linebuff_, "\r")) 
        {
            break;
        }
    }

    if (is.bad()) {
        // let the caller check for such events
        return is;
    }

    if (is.eof() &&
        (linebuff_.empty() || starts_with(linebuff_, comment_prefix_))) {
        // last line is either empty or a comment; don't parse anything
        return is;
    }

    if (is.fail()) {
        throw swc_parse_error("too long line detected", lineno_);
    }

    std::istringstream line(linebuff_);
    try {
        record = parse_record(line);
    } catch (std::invalid_argument &e) {
        // Rethrow as a parse error
        throw swc_parse_error(e.what(), lineno_);
    }

    return is;
}

swc_record swc_parser::parse_record(std::istringstream &is)
{
    auto id = parse_value_strict<int>(is, *this);
    auto type = parse_value_strict<swc_record::kind>(is, *this);
    auto x = parse_value_strict<swc_record::coord_type>(is, *this);
    auto y = parse_value_strict<swc_record::coord_type>(is, *this);
    auto z = parse_value_strict<swc_record::coord_type>(is, *this);
    auto r = parse_value_strict<swc_record::coord_type>(is, *this);
    auto parent_id = parse_value_strict<swc_record::id_type>(is, *this);

    // Convert to zero-based, leaving parent_id as-is if -1
    if (parent_id != -1) {
        parent_id--;
    }

    return swc_record(type, id-1, x, y, z, r, parent_id);
}


swc_record_range_clean::swc_record_range_clean(std::istream &is)
{
    std::unordered_set<swc_record::id_type> ids;

    std::size_t         num_trees = 0;
    swc_record::id_type last_id   = -1;
    bool                needsort  = false;

    swc_record curr_record;
    for (auto r : swc_get_records<swc_io_raw>(is)) {
        if (r.parent() == -1 && ++num_trees > 1) {
            // only a single tree is allowed
            break;
        }

        auto inserted = ids.insert(r.id());
        if (inserted.second) {
            // not a duplicate; insert record
            records_.push_back(r);
            if (!needsort && r.id() < last_id) {
                needsort = true;
            }

            last_id = r.id();
        }
    }

    if (needsort) {
        std::sort(records_.begin(), records_.end());
    }

    // Renumber records if necessary
    std::map<swc_record::id_type, swc_record::id_type> idmap;
    swc_record::id_type next_id = 0;
    for (auto &r : records_) {
        if (r.id() != next_id) {
            r.renumber(next_id, idmap);
        }

        ++next_id;
    }

    // Reject if branches are not contiguously numbered
    std::vector<swc_record::id_type> parent_list = { 0 };
    for (std::size_t i = 1; i < records_.size(); ++i) {
        parent_list.push_back(records_[i].parent());
    }

    if (!nest::mc::algorithms::has_contiguous_segments(parent_list)) {
        throw swc_parse_error("branches are not contiguously numbered", 0);
    }
}

cell swc_read_cell(std::istream &is)
{
    using namespace nest::mc;

    cell newcell;
    std::vector<swc_record::id_type> parent_index;
    std::vector<swc_record> swc_records;
    for (const auto &r : swc_get_records<swc_io_clean>(is)) {
        swc_records.push_back(r);
        parent_index.push_back(r.parent());
    }

    if (parent_index.empty()) {
        return newcell;
    }

    // The parent of soma must be 0, while in SWC files is -1
    parent_index[0] = 0;
    auto branch_index     = algorithms::branches(parent_index);
    auto new_parent_index = algorithms::make_parent_index(parent_index,
                                                          branch_index);

    // sanity check
    EXPECTS(new_parent_index.size() == branch_index.size() - 1);

    // Add the soma first; then the segments
    newcell.add_soma(swc_records[0].radius(), swc_records[0].coord());
    for (std::size_t i = 1; i < new_parent_index.size(); ++i) {
        auto b_start = std::next(swc_records.begin(), branch_index[i]);
        auto b_end   = std::next(swc_records.begin(), branch_index[i+1]);

        std::vector<swc_record::coord_type> radii;
        std::vector<nest::mc::point<swc_record::coord_type>> points;
        if (new_parent_index[i] != 0) {
            // include the parent of current record if not branching from soma
            auto p = parent_index[branch_index[i]];
            radii.push_back(swc_records[p].radius());
            points.push_back(swc_records[p].coord());
        }

        // extract the radii and the points
        std::for_each(b_start, b_end,
                      [&radii](const swc_record& r) {
                          radii.push_back(r.radius());
                      });

        std::for_each(b_start, b_end,
                      [&points](const swc_record& r) {
                          points.push_back(r.coord());
                      });

        // add the new cable
        newcell.add_cable(new_parent_index[i],
                          nest::mc::segmentKind::dendrite, radii, points);
    }

    return newcell;
}

} // namespace io
} // namespace mc
} // namespace nest

#include <algorithm>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/point.hpp>
#include <arbor/swcio.hpp>

#include "algorithms.hpp"
#include "util/span.hpp"

namespace arb {

// swc_record implementation

// helper function: return error message if inconsistent, or nullptr if ok.
const char* swc_record_error(const swc_record& r) {
    if (r.tag<0) {
        return "unknown record tag";
    }

    if (r.id < 0) {
        return "negative ids not allowed";
    }

    if (r.parent_id < -1) {
        return "parent_id < -1 not allowed";
    }

    if (r.parent_id >= r.id) {
        return "parent_id >= id is not allowed";
    }

    if (r.r < 0) {
        return "negative radii are not allowed";
    }

    return nullptr;
}


bool swc_record::is_consistent() const {
    return swc_record_error(*this)==nullptr;
}

void swc_record::assert_consistent() const {
    const char* error = swc_record_error(*this);
    if (error) {
        throw swc_error(error);
    }
}

static bool parse_record(const std::string& line, swc_record& record) {
    std::istringstream is(line);
    swc_record r;

    is >> r.id >> r.tag >> r.x >> r.y >> r.z >> r.r >> r.parent_id;
    if (is) {
        // Convert to zero-based, leaving parent_id as-is if -1
        --r.id;
        if (r.parent_id>=0) {
            --r.parent_id;
        }
        record = r;
        return true;
    }
    return false;
}

static bool is_comment(const std::string& line) {
    auto pos = line.find_first_not_of(" \f\n\r\t\v");
    return pos==std::string::npos || line[pos]=='#';
}

std::istream& operator>>(std::istream& is, swc_record& record) {
    std::string line;

    while (is) {
        std::getline(is, line);
        if (!is) {
            break;
        }

        if (is_comment(line)) {
            continue;
        }

        bool ok = parse_record(line, record);
        if (!ok) {
            is.setstate(std::ios::failbit);
        }
        break;
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const swc_record& record) {
    // output in one-based indexing
    os << record.id+1 << " "
       << record.tag << " "
       << std::setprecision(7) << record.x << " "
       << std::setprecision(7) << record.y << " "
       << std::setprecision(7) << record.z << " "
       << std::setprecision(7) << record.r << " "
       << ((record.parent_id == -1) ? record.parent_id : record.parent_id+1);

    return os;
}

std::vector<swc_record> parse_swc_file(std::istream& is) {
    constexpr auto eof = std::char_traits<char>::eof();
    std::vector<swc_record> records;
    std::size_t line_number = 1;
    std::string line;

    try {
        while (is && is.peek()!=eof) {
            std::getline(is, line);
            if (is_comment(line)) {
                continue;
            }

            swc_record record;
            bool ok = parse_record(line, record);
            if (!ok) {
                is.setstate(std::ios::failbit);
            }
            else {
                record.assert_consistent();
                records.push_back(std::move(record));
            }
            ++line_number;
        }
    }
    catch (swc_error& e) {
        e.line_number = line_number; // rethrow with saved line number
        throw e;
    }

    if (!is.eof()) {
        // parse error, so throw exception
        throw swc_error("SWC parse error", line_number);
    }

    swc_canonicalize(records);
    return records;
}

void swc_canonicalize(std::vector<swc_record>& swc_records) {
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
        std::sort(swc_records.begin(), swc_records.end(),
            [](const swc_record& a, const swc_record& b) { return a.id<b.id; });
    }

    // Renumber records if necessary.
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

    // Reject if branches are not contiguously numbered.
    std::vector<swc_record::id_type> parent_list = { 0 };
    for (std::size_t i = 1; i < swc_records.size(); ++i) {
        parent_list.push_back(swc_records[i].parent_id);
    }

    if (!arb::algorithms::has_contiguous_compartments(parent_list)) {
        throw swc_error("branches are not contiguously numbered", 0);
    }
}

} // namespace arb

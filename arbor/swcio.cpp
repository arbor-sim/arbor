#include <algorithm>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <unordered_set>

#include <arbor/assert.hpp>
#include <arbor/morphology.hpp>
#include <arbor/point.hpp>

#include "algorithms.hpp"
#include "swcio.hpp"

namespace arb {
namespace io {

// swc_record implementation


// helper function: return error message if inconsistent, or nullptr if ok.
const char* swc_record_error(const swc_record& r) {
    constexpr int max_type = static_cast<int>(swc_record::kind::custom);

    if (static_cast<int>(r.type) < 0 || static_cast<int>(r.type) > max_type) {
        return "unknown record type";
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

bool parse_record(const std::string& line, swc_record& record) {
    std::istringstream is(line);
    swc_record r;

    int type_as_int;
    is >> r.id >> type_as_int >> r.x >> r.y >> r.z >> r.r >> r.parent_id;
    r.type = static_cast<swc_record::kind>(type_as_int);
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

bool is_comment(const std::string& line) {
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
       << static_cast<int>(record.type) << " "
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

    swc_canonicalize_sequence(records);
    return records;
}

} // namespace io
} // namespace arb

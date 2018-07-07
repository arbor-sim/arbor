#pragma once

// Convenience wrappers for ostream formatting.

#include <locale>
#include <ostream>
#include <string>

namespace io {

template <typename T>
struct quoted_item {
    quoted_item(const T& item): item(item) {}
    const T& item;
};

template <typename T>
std::ostream& operator<<(std::ostream& o, const quoted_item<T>& q) {
    return o << '"' << q.item << '"';
}

// Print item enclosed in double quotes.

template <typename T>
quoted_item<T> quote(const T& item) { return quoted_item<T>(item); }

// Separator which prints nothing or a prefix the first time it is used,
// then prints the delimiter text each time thereafter.

struct separator {
    explicit separator(std::string d):
        delimiter(std::move(d)) {}

    separator(std::string p, std::string d):
        prefix(std::move(p)),  delimiter(std::move(d)) {}

    std::string prefix, delimiter;
    bool visited = false;

    friend std::ostream& operator<<(std::ostream& o, separator& sep) {
        o << (sep.visited? sep.delimiter: sep.prefix);
        sep.visited = true;
        return o;
    }

    void reset() { visited = false; }
};

// Reset locale on a stream to 'classic' for locale-independent formatting
// inline.

inline std::ios_base& classic(std::ios_base& ios) {
    ios.imbue(std::locale::classic());
    return ios;
}

} // namespace io

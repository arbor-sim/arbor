#pragma once

#include <ostream>

struct Location {
    int line;
    int column;

    Location() : Location(1, 1)
    {}

    Location(int ln, int col)
    :   line(ln),
        column(col)
    {}
};

inline std::ostream& operator<< (std::ostream& os, Location const& L) {
    return os << "(line " << L.line << ",col " << L.column << ")";
}



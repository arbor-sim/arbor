#pragma once

#include <stdexcept>

#include "msparse.hpp"

// Symbolic sparse matrix manipulation for symbolic Gauss-Jordan elimination
// (used in `sparse` solver).

namespace symge {

struct symbol_error: public std::runtime_error {
    symbol_error(const std::string& what): std::runtime_error(what) {}
};

// Abstract symbols:

class symbol_table;

class symbol {
private:
    unsigned index_;
    const symbol_table* table_;

    // Valid symbols are constructed via a symbol table.
    friend class symbol_table;
    symbol(unsigned index, const symbol_table* table):
        index_(index), table_(table) {}

public:
    symbol(): index_(0), table_(nullptr) {}

    // true => valid symbol.
    operator bool() const { return table_; }

    bool operator==(symbol other) const { return index_==other.index_ && table_==other.table_; }
    bool operator!=(symbol other) const { return !(*this==other); }

    const symbol_table* table() const { return table_; }
};

// A `symbol_term` is either zero or a product of symbols.

struct symbol_term {
    symbol left, right;

    symbol_term() = default;
    bool is_zero() const { return !left || !right; }
    operator bool() const { return !is_zero(); }
};

struct symbol_term_diff {
    symbol_term left, right;

    symbol_term_diff() = default;
    symbol_term_diff(const symbol_term& left): left(left), right{} {}
    symbol_term_diff(const symbol_term& left, const symbol_term& right):
        left(left), right(right) {}
};

inline symbol_term operator*(symbol a, symbol b) {
    return symbol_term{a, b};
}

inline symbol_term_diff operator-(symbol_term l, symbol_term r) {
    return symbol_term_diff{l, r};
}

inline symbol_term_diff operator-(symbol_term r) {
    return symbol_term_diff{symbol_term{}, r};
}

// Symbols are not re-assignable; they are created as primitive, or
// have a definition in terms of a `symbol_term_diff`.

class symbol_table {
private:
    struct table_entry {
        std::string name;
        symbol_term_diff def;
        bool defined;
    };

    std::vector<table_entry> entries_;

public:
    // make new primitive symbol
    symbol define(const std::string& name="") {
        symbol s(size(), this);
        entries_.push_back({name, symbol_term_diff{}, false});
        return s;
    }

    // make new symbol with definition
    symbol define(const std::string& name, const symbol_term_diff& def) {
        symbol s(size(), this);
        entries_.push_back({name, def, true});
        return s;
    }

    symbol define(const symbol_term_diff& def) {
        return define("", def);
    }

    symbol_term_diff get(symbol s) const {
        if (!defined(s)) throw symbol_error("symbol is primitive");
        return entries_[s.index_].def;
    }

    bool defined(symbol s) const {
        if (!valid(s)) throw symbol_error("symbol not present in this table");
        return entries_[s.index_].defined;
    }

    bool primitive(symbol s) const { return !defined(s); }

    const std::string& name(symbol s) const {
        if (!valid(s)) throw symbol_error("symbol not present in this table");
        return entries_[s.index_].name;
    }

    void name(symbol s, const std::string& n) {
        if (!valid(s)) throw symbol_error("symbol not present in this table");
        entries_[s.index_].name = n;
    }

    std::size_t size() const { return entries_.size(); }

    symbol operator[](unsigned i) const { return symbol{i, this}; }

    bool valid(symbol s) const { return s.table_==this && s.index_<size(); }

    // Existing symbols associated with this table are invalidated by clear().
    void clear() { entries_.clear(); }
};

inline std::string name(symbol s) {
    return s? s.table()->name(s): "";
}

inline symbol_term_diff definition(symbol s) {
    if (!s) throw symbol_error("invalid symbol");
    return s.table()->get(s);
}

inline bool primitive(symbol s) {
    return s && s.table()->primitive(s);
}

using sym_row = msparse::row<symbol>;
using sym_matrix = msparse::matrix<symbol>;

// Perform Gauss-Jordan reduction on a (possibly augmented) symbolic matrix, with
// pivots taken from the diagonal elements. New symbol definitions due to fill-in
// will be added via the provided symbol table.
// Returns a vector of vectors of symbols, partitioned by row of the matrix
std::vector<std::vector<symge::symbol>> gj_reduce(sym_matrix& A, symbol_table& table);

} // namespace symge

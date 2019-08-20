#pragma once

// (Possibly augmented) matrix implementation, represented as a vector of sparse rows.

#include <algorithm>
#include <utility>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace msparse {

struct msparse_error: std::runtime_error {
    msparse_error(const std::string &what): std::runtime_error(what) {}
};

constexpr unsigned row_npos = unsigned(-1);

// `msparse::row` represents one sparse matrix row as a vector of
// (column, value) pairs, ordered by (unsigned) column. `row_npos`
// is used to represent an invalid column number.

template <typename X>
class row {
public:
    struct entry {
        unsigned col;
        X value;
    };
    static constexpr unsigned npos = row_npos;

private:
    std::vector<entry> data_;

    // Entries must have strictly monotonically increasing column numbers.
    bool check_invariant() const {
        for (unsigned i = 1; i<data_.size(); ++i) {
            if (data_[i].col<=data_[i-1].col) return false;
        }
        return true;
    }

public:
    row() = default;
    row(const row&) = default;

    row(std::initializer_list<entry> il): data_(il) {
        if (!check_invariant())
            throw msparse_error("improper row element list");
    }

    template <typename InIter>
    row(InIter b, InIter e): data_(b, e) {
        if (!check_invariant())
            throw msparse_error("improper row element list");
    }

    unsigned size() const { return data_.size(); }
    bool empty() const { return size()==0; }

    // Iterators present row as sequence of `entry` objects.
    auto begin() -> decltype(data_.begin()) { return data_.begin(); }
    auto begin() const -> decltype(data_.cbegin()) { return data_.cbegin(); }
    auto end() -> decltype(data_.end()) { return data_.end(); }
    auto end() const -> decltype(data_.cend()) { return data_.cend(); }

    // Return column of first (left-most) entry.
    unsigned mincol() const {
        return empty()? npos: data_.front().col;
    }

    // Return column of first entry with column greater than `c`.
    unsigned mincol_after(unsigned c) const {
        auto i = std::upper_bound(data_.begin(), data_.end(), c,
            [](unsigned a, const entry& b) { return a<b.col; });

        return i==data_.end()? npos: i->col;
    }

    // Return column of last (right-most) entry.
    unsigned maxcol() const {
        return empty()? npos: data_.back().col;
    }

    // As opposed to [] indexing (see below), retrieve `i'th entry from
    // the list of entries.
    const entry& get(unsigned i) const {
        return data_[i];
    }

    void push_back(const entry& e) {
        if (!empty() && e.col <= data_.back().col)
            throw msparse_error("cannot push_back row elements out of order");
        data_.push_back(e);
    }

    void clear() {
        data_.clear();
    }

    // Return index into entry list which has column `c`.
    unsigned index(unsigned c) const {
        auto i = std::lower_bound(data_.begin(), data_.end(), c,
            [](const entry& a, unsigned b) { return a.col<b; });

        return (i==data_.end() || i->col!=c)? npos: std::distance(data_.begin(), i);
    }

    // Remove all entries from column `c` onwards.
    void truncate(unsigned c) {
        auto i = std::lower_bound(data_.begin(), data_.end(), c,
            [](const entry& a, unsigned b) { return a.col<b; });
        data_.erase(i, data_.end());
    }

    // Return value at column `c`; if no entry in row, return default-constructed `X`,
    // i.e. 0 for numeric types.
    X operator[](unsigned c) const {
        auto i = index(c);
        return i==npos? X{}: data_[i].value;
    }

    // Proxy object to allow assigning elements with the syntax `row[c] = value`.
    struct assign_proxy {
        row<X>& row_;
        unsigned c;

        assign_proxy(row<X>& r, unsigned c): row_(r), c(c) {}

        operator X() const { return const_cast<const row<X>&>(row_)[c]; }
        assign_proxy& operator=(const X& x) {
            auto i = std::lower_bound(row_.data_.begin(), row_.data_.end(), c,
                [](const entry& a, unsigned b) { return a.col<b; });

            if (i==row_.data_.end() || i->col!=c) {
                row_.data_.insert(i, {c, x});
            }
            else if (x == X{}) {
                row_.data_.erase(i);
            }
            else {
                i->value = x;
            }

            return *this;
        }
    };

    assign_proxy operator[](unsigned c) {
        return assign_proxy{*this, c};
    }
};

// `msparse::matrix` represents a matrix by a size (number of rows,
// columns) and vector of sparse `mspase::row` rows.
//
// The matrix may also be 'augmented', with columns corresponding to a second
// matrix appended on the right.

template <typename X>
class matrix {
private:
    std::vector<row<X>> rows;
    unsigned cols = 0;
    unsigned aug = row_npos;

public:
    static constexpr unsigned npos = row_npos;

    matrix() = default;
    matrix(unsigned n, unsigned c): rows(n), cols(c) {}

    row<X>& operator[](unsigned i) { return rows[i]; }
    const row<X>& operator[](unsigned i) const { return rows[i]; }

    unsigned size() const { return rows.size(); }
    unsigned nrow() const { return size(); }
    unsigned ncol() const { return cols; }

    // First column corresponding to the augmented submatrix.
    unsigned augcol() const { return aug; }

    bool empty() const { return size()==0; }
    bool augmented() const { return aug!=npos; }

    void clear() {
        rows.clear();
        cols = 0;
        aug = row_npos;
    }

    // Add a column on the right as part of the augmented submatrix.
    // The new entries are provided by a (full, dense representation)
    // sequence of values.
    template <typename Seq>
    void augment(const Seq& col_dense) {
        unsigned r = 0;
        for (const auto& v: col_dense) {
            if (r>=rows.size()) throw msparse_error("augmented column size mismatch");
            rows[r++].push_back({cols, v});
        }
        if (aug==npos) aug=cols;
        ++cols;
    }

    // Remove all augmented columns.
    void diminish() {
        if (aug==npos) return;
        for (auto& row: rows) row.truncate(aug);
        cols = aug;
        aug = npos;
    }
};

} // namespace msparse

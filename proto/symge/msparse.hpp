#pragma once

#include <algorithm>
#include <utility>
#include <initializer_list>
#include <iterator>
#include <vector>

#include <util/compat.hpp>
#include <util/iterutil.hpp>
#include <util/rangeutil.hpp>

namespace msparse {

namespace util = nest::mc::util;

struct msparse_error: std::runtime_error {
    msparse_error(const std::string &what): std::runtime_error(what) {}
};

constexpr unsigned mrow_npos = unsigned(-1);

template <typename X>
class mrow {
public:
    using entry = std::pair<unsigned, X>;
    static constexpr unsigned npos = mrow_npos;

private:
    std::vector<entry> data;

    bool check_invariant() {
        for (unsigned i = 1; i<data.size(); ++i) {
            if (data[i].first<=data[i-1].first) return false;
        }
        return true;
    }

public:
    mrow() = default;
    mrow(const mrow&) = default;

    mrow(std::initializer_list<entry> il): data(il) {
        if (!check_invariant())
            throw msparse_error("improper row element list");
    }

    template <typename InIter>
    mrow(InIter b, InIter e): data(b, e) {
        if (!check_invariant())
            throw msparse_error("improper row element list");
    }

    unsigned size() const { return data.size(); }
    bool empty() const { return size()==0; }

    auto begin() -> decltype(data.begin()) { return data.begin(); }
    auto begin() const -> decltype(data.cbegin()) { return data.cbegin(); }
    auto end() -> decltype(data.end()) { return data.end(); }
    auto end() const -> decltype(data.cend()) { return data.cend(); }

    unsigned mincol() const {
        return empty()? npos: data.front().first;
    }

    unsigned mincol_after(unsigned c) const {
        auto i = std::upper_bound(data.begin(), data.end(), c,
            [](unsigned a, const entry& b) { return a<b.first; });

        return i==data.end()? npos: i->first;
    }

    unsigned maxcol() const {
        return empty()? npos: data.back().first;
    }

    const entry& get(unsigned i) const {
        return data[i];
    }

    void push_back(const entry& e) {
        if (!empty() && e.first <= data.back().first)
            throw msparse_error("cannot push_back row elements out of order");
        data.push_back(e);
    }

    unsigned index(unsigned c) const {
        auto i = std::lower_bound(data.begin(), data.end(), c,
            [](const entry& a, unsigned b) { return a.first<b; });

        return (i==data.end() || i->first!=c)? npos: std::distance(data.begin(), i);
    }

    // remove all entries from column c onwards
    void truncate(unsigned c) {
        auto i = std::lower_bound(data.begin(), data.end(), c,
            [](const entry& a, unsigned b) { return a.first<b; });
        data.erase(i, data.end());
    }

    X operator[](unsigned c) const {
        auto i = index(c);
        return i==npos? X{}: data[i].second;
    }

    struct assign_proxy {
        mrow<X>& row;
        unsigned c;

        assign_proxy(mrow<X>& row, unsigned c): row(row), c(c) {}

        operator X() const { return const_cast<const mrow<X>&>(row)[c]; }
        assign_proxy& operator=(const X& x) {
            auto i = std::lower_bound(row.data.begin(), row.data.end(), c,
                [](const entry& a, unsigned b) { return a.first<b; });

            if (i==row.data.end() || i->first!=c) {
                row.data.insert(i, {c, x});
            }
            else if (x == X{}) {
                row.data.erase(i);
            }
            else {
                i->second = x;
            }

            return *this;
        }
    };

    assign_proxy operator[](unsigned c) {
        return assign_proxy{*this, c};
    }

    template <typename RASeq>
    auto dot(const RASeq& v) const -> decltype(X{}*util::front(v)) {
        using result_type = decltype(X{}*util::front(v));
        result_type s{};

        auto nv = util::size(v);
        for (const auto& e: data) {
            if (e.first>=nv) throw msparse_error("right multiplicand too short");
            s += e.second*v[e.first];
        }
        return s;
    }
};

template <typename X>
class matrix {
private:
    std::vector<mrow<X>> rows;
    unsigned cols = 0;
    unsigned aug = mrow_npos;

public:
    static constexpr unsigned npos = mrow_npos;

    matrix() = default;
    matrix(unsigned n, unsigned c): rows(n), cols(c) {}

    mrow<X>& operator[](unsigned i) { return rows[i]; }
    const mrow<X>& operator[](unsigned i) const { return rows[i]; }

    unsigned size() const { return rows.size(); }
    unsigned nrow() const { return size(); }
    unsigned ncol() const { return cols; }
    unsigned augcol() const { return aug; }

    bool empty() const { return size()==0; }
    bool augmented() const { return aug!=npos; }

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

    void diminish() {
        if (aug==npos) return;
        for (auto& row: rows) row.truncate(aug);
        cols = aug;
        aug = npos;
    }
};

// sparse * dense vector muliply:

template <typename AT, typename RASeqX, typename SeqB>
void mul_dense(const matrix<AT>& A, const RASeqX& x, SeqB& b) {
    auto bi = std::begin(b);
    unsigned n = A.nrow();
    for (unsigned i = 0; i<n; ++i) {
        if (bi==compat::end(b)) throw msparse_error("output sequence b too short");
        *bi++ = A[i].dot(x);
    }
}

} // namespace msparse

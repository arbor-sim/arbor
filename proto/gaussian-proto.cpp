#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <util/compat.hpp>
#include <util/optional.hpp>
#include <util/rangeutil.hpp>
#include <util/span.hpp>

namespace util = nest::mc::util;
using util::optional;
using util::nothing;
using util::span;

// identifier name picking helper routines

void join_impl(std::ostream& ss, const std::string& sep) {}

template <typename Head, typename... Args>
void join_impl(std::ostream& ss, const std::string& sep, Head&& head, Args&&... tail) {
    if (sizeof...(tail)==0)
        ss << std::forward<Head>(head);
    else
        join_impl(ss << std::forward<Head>(head) << sep, sep, std::forward<Args>(tail)...);
}

template <typename... Args>
std::string join(const std::string& sep, Args&&... items) {
    std::stringstream ss;
    join_impl(ss, sep, std::forward<Args>(items)...);
    return ss.str();
}

std::string next_id(std::string s) {
    unsigned l = s.size();
    if (l==0) return "a";

    unsigned i = l-1;

    char l0='a', l1='z';
    char u0='A', u1='Z';
    char d0='0', d1='9';
    for (;;) {
        char& c = s[i];
        if (c>=l0 && c<l1 || c>=u0 && c<u1 || c>=d0 && c<d1) {
            ++c;
            return s;
        }
        if (c==l1) c=l0;
        if (c==u1) c=u0;
        if (c==d1) c=d0;
        if (i==0) break;
        --i;
    }

    // prepend a character based on class of first
    if (s[0]==u0) return u0+s;
    if (s[0]==d0) return d0+s;
    return l0+s;
}

// store represents map from symbols to values

using symidx = unsigned;
constexpr symidx no_sym = 0u; // 0 => invalid symbol

struct store {
    std::vector<optional<double>> data;

    optional<double>& operator[](symidx i) {
        if (i==no_sym) throw std::runtime_error("no such symbol");

        if (i>data.size()) data.resize(i);
        return data[i-1];
    }

    optional<double> operator[](symidx i) const {
        return i==no_sym || i>data.size()? nothing: data[i-1];
    }
};

// termexpr represents a difference of two terms, where
// each term is either zero or a product of two symbols.

struct termexpr {
    // a*b or zero
    struct term {
        bool zero;
        symidx a, b;

        term(): zero(true) {}
        term(symidx a, symidx b): zero(false), a(a), b(b) {}
    };

    term left, right;
};

// symtbl represents map from symbols to expressions

struct symtbl {
    std::vector<termexpr> def;
    std::vector<std::string> names;
    std::unordered_map<std::string, symidx> name_index;

    symidx newsym(const termexpr& expr, const std::string& name="") {
        def.push_back(expr);
        symidx s = def.size(); // valid syms have indices starting from 1.
        names.push_back(name);
        name_index[name] = s;
        return s;
    }

    template <typename... Args>
    std::string make_id(std::string prefix, Args&&... suffixes) {
        std::string suffix = join("",std::forward<Args>(suffixes)...);

        while (name_index.count(prefix+suffix)) prefix = next_id(prefix);
        return prefix+suffix;
    }

    const termexpr& operator[](symidx i) const {
        --i;
        if (i>=def.size()) throw std::runtime_error("no such symbol");
        return def[i];
    }

    const std::string& id(symidx i) const {
        --i;
        if (i>=names.size()) throw std::runtime_error("no such symbol");
        return names[i];
    }

    std::size_t size() const {
        return def.size();
    }

    span<symidx> all() const {
        return span<symidx>(1u, size()+1);
    }
};

std::string pprint(const termexpr& expr, const symtbl& syms) {
    std::string s;

    if (!expr.left.zero) {
        s = syms.id(expr.left.a)+"*"+syms.id(expr.left.b);
    }
    if (!expr.right.zero) {
        s += '-';
        s += syms.id(expr.right.a)+"*"+syms.id(expr.right.b);
    }

    if (s.empty()) s = "0";
    return s;
}

double eval(symidx s, const symtbl& syms, store& vals);

double eval(const termexpr& expr, const symtbl& syms, store& vals) {
    auto expand = [&syms,&vals](symidx s) -> double { return eval(s, syms, vals); };

    double left = expr.left.zero? 0.: expand(expr.left.a)*expand(expr.left.b);
    double right = expr.right.zero? 0.: expand(expr.right.a)*expand(expr.right.b);
    return left-right;
}

double eval(symidx s, const symtbl& syms, store& vals) {
    auto& v = vals[s];
    if (!v) v=eval(syms[s], syms, vals);
    return *v;
};

template <typename X>
std::ostream &operator<<(std::ostream& o, const optional<X>& x) {
    return x? o << *x: o << "nothing";
}

void demo_store_eval() {
    store vals;
    symtbl syms;

    auto a1 = syms.newsym(termexpr{}, syms.make_id("a",1));
    auto a2 = syms.newsym(termexpr{}, syms.make_id("a",2));
    auto a3 = syms.newsym(termexpr{}, syms.make_id("a",3));
    auto b =  syms.newsym(termexpr{{a1,a2},{a2,a3}}, syms.make_id("b"));
    auto c =  syms.newsym(termexpr{{a1,a2},{a1,b}}, syms.make_id("c"));

    std::cout << syms.id(b) << "=" << pprint(syms[b], syms) << "\n";
    std::cout << syms.id(c) << "=" << pprint(syms[c], syms) << "\n";
    vals[a1] = 2;
    vals[a2] = 3;
    vals[a3] = 5;

    double cval = eval(c, syms, vals); // should force eval of b, too.
    std::cout << syms.id(c) << "=" << cval << "\n";

    std::cout << "value store\n";
    for (symidx s: syms.all()) {
        std::cout << syms.id(s) << "=" << vals[s] << "\n";
    }
}

// sparse matrix repn

template <typename X>
struct mrow {
    using entry = std::pair<unsigned, X>;
    std::vector<entry> data;

    static constexpr unsigned npos = unsigned(-1);

    mrow() = default;
    mrow(const mrow&) = default;
    mrow(std::initializer_list<entry> il): data(il) {
        util::sort_by(data, [](const entry& e) { return e.first; });
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
            throw std::runtime_error("cannot push_back row elements out of order");
        data.push_back(e);
    }

    unsigned index(unsigned c) const {
        auto i = std::lower_bound(data.begin(), data.end(), c,
            [](const entry& a, unsigned b) { return a.first<b; });

        return (i==data.end() || i->first!=c)? npos: std::distance(data.begin(), i);
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
            if (e.first>=nv) throw std::runtime_error("multiplicand v too short");
            s += e.second*v[e.first];
        }
        return s;
    }
};

template <typename X>
struct msparse {
    std::vector<mrow<X>> rows;

    msparse() = default;
    explicit msparse(unsigned n): rows(n) {}

    mrow<X>& operator[](unsigned i) { return rows[i]; }
    const mrow<X>& operator[](unsigned i) const { return rows[i]; }

    unsigned size() const { return rows.size(); }
    unsigned nrow() const { return size(); }
    bool empty() const { return size()==0; }
};

// compute A*x -> b for matrix A and vector x

template <typename AT, typename RASeqX, typename SeqB>
void mul_msparse_vec(const msparse<AT>& A, const RASeqX& x, SeqB& b) {
    auto bi = std::begin(b);
    for (const auto& row: A.rows) {
        if (bi==compat::end(b)) throw std::runtime_error("output sequence b too short");
        *bi++ = row.dot(x);
    }
}

// make random test matrices/sparsity patterns

// density is expected proportion of non-zero non-diagonal elements
template <typename Rng>
msparse<double> make_random_matrix(unsigned n, double density, Rng& R) {
    std::uniform_real_distribution<double> U;
    msparse<double> M(n);

    for (unsigned i = 0; i<n; ++i) {
        for (unsigned j = 0; j<n; ++j) {
            if (i!=j && U(R)>density) continue;
            double u = U(R);
            M[i][j] = i==j? n*(1+u): u-0.5;
        }
    }
    return M;
}

template <typename X>
std::string pprint(const msparse<X>& m, unsigned ncol, unsigned width=10) {
    std::stringstream ss;
    for (unsigned r = 0; r<m.nrow(); ++r) {
        ss << '|';
        for (unsigned c = 0; c<ncol; ++c)
            ss << std::setw(width) << m[r][c];
        ss << " |\n";
    }
    return ss.str();
}

template <typename Sep, typename V>
struct sepval_t {
    const Sep& sep;
    const V& v;

    sepval_t(const Sep& sep, const V& v): sep(sep), v(v) {}

    friend std::ostream& operator<<(std::ostream& O, const sepval_t& sv) {
        bool first = true;
        for (const auto& x: sv.v) {
            if (!first) O << sv.sep;
            first = false;
            O << x;
        }
        return O;
    }
};

template <typename Sep, typename V>
sepval_t<Sep,V> sepval(const Sep& sep, const V& v) { return sepval_t<Sep,V>(sep, v); }

void demo_msparse_random() {
    std::minstd_rand R;
    msparse<double> M = make_random_matrix(5, 0.3, R);

    std::cout << "M:\n" << pprint(M,5);

    int x[] = { 1, 2, 3, 4, 5 };
    std::cout << "x: " << sepval(',', x) << "\n";

    std::vector<double> b(5);
    mul_msparse_vec(M, x, b);
    std::cout << "Mx: " << sepval(',', b) << "\n";
}

// GE

std::string pprint(const msparse<symidx>& m, unsigned ncol, const symtbl& syms, unsigned width=10) {
    std::stringstream ss;
    for (unsigned r = 0; r<m.nrow(); ++r) {
        ss << '|';
        for (unsigned c = 0; c<ncol; ++c) {
            ss << std::setw(width);
            auto s = m[r][c];
            if (s!=no_sym) {
                ss << syms.id(s);
            }
            else {
                ss << '.';
            }
        }
        ss << " |\n";
    }
    return ss.str();
}

// return q[c]*p - p[c]*q (rownum used only for symbol names)
mrow<symidx> row_ge(unsigned c, const mrow<symidx>& p, const mrow<symidx>& q, symtbl& syms, unsigned rownum=0) {
    if (p.index(c)==p.npos || q.index(c)==q.npos) throw std::runtime_error("improper row GE");

    mrow<symidx> u;
    symidx x = q[c];
    symidx y = p[c];

    auto piter = p.begin();
    auto qiter = q.begin();
    unsigned pj = piter->first;
    unsigned qj = qiter->first;

    while (piter!=p.end() || qiter!=q.end()) {
        unsigned j = std::min(pj, qj);
        termexpr::term t1, t2;

        if (j==pj) {
            t1 = termexpr::term{x, piter->second};
            ++piter;
            pj = piter==p.end()? p.npos: piter->first;
        }
        if (j==qj) {
            t2 = termexpr::term{y, qiter->second};
            ++qiter;
            qj = qiter==q.end()? q.npos: qiter->first;
        }
        if (j!=c) {
            auto id = syms.make_id("c", rownum, j);
            symidx s = syms.newsym({t1, t2}, id);
            u.push_back({j, s});
        }
    }
    return u;
}

// ncol: number of columns before augmentation
void gj_reduce(msparse<symidx>& A, unsigned ncol, symtbl& syms) {
    constexpr auto npos = mrow<symidx>::npos;

    struct pq_entry {
        unsigned key;
        unsigned mincol;
        unsigned row;
    };

    struct pq_order_t {
        bool operator()(const pq_entry& a, const pq_entry& b) const {
            return a.key>b.key || a.key==b.key && a.mincol<b.mincol;
        }
    } pq_order;

    std::priority_queue<pq_entry, std::vector<pq_entry>, pq_order_t> pq;

    unsigned n = A.nrow();
    for (unsigned i: span<unsigned>(0, n)) {
        unsigned c = A[i].mincol();
        if (c<ncol) pq.push({c, c, i});
    }

    while (!pq.empty()) {
        pq_entry pick = pq.top();
        pq.pop();

        unsigned col = pick.key;
        auto r1 = pick.row;

        //std::cout << "picked row: " << r1 << "\n";

        while (!pq.empty() && pq.top().key==pick.key) {
            pq_entry top = pq.top();
            pq.pop();

            auto r2 = top.row;

            A[r2] = row_ge(col, A[r2], A[r1], syms, r2);
            //std::cout << "reduce " << r2 << " by " << r1 << ":\n" << pprint(A, ncol, syms);

            unsigned c = A[r2].mincol_after(col);
            if (c<ncol) pq.push({c, A[r2].mincol(), r2});
        }
        unsigned c = A[r1].mincol_after(col);
        if (c<ncol) pq.push({c, pick.mincol, r1});
    }
}

void demo_sym_matrix() {
    std::minstd_rand R;
    unsigned n = 8;
    msparse<double> M = make_random_matrix(n, 0.3, R);

    store vals;
    symtbl syms;
    msparse<symidx> S;

    for (unsigned i = 0; i<M.nrow(); ++i) {
        const auto& row = M.rows[i];
        mrow<symidx> r;
        mrow<std::string> r_print;
        for (const auto& el: row) {
            unsigned j = el.first;
            auto a = syms.newsym(termexpr{}, syms.make_id("a", i, j));
            vals[a] = el.second;
            r.push_back({j, a});
        }
        S.rows.push_back(r);
    }

    std::cout << "M:\n" << pprint(M, n);
    std::cout << "S:\n" << pprint(S, n, syms);

    std::cout << "symbols:\n";
    for (symidx s: syms.all()) {
        std::cout << syms.id(s) << ": ";
        auto v = vals[s];
        if (v) std::cout << *v << "\n";
        else std::cout << pprint(syms[s], syms) << "\n";
    }

    // Two GE step: reduce row 4 by row 0 then by row 1
    //S[4] = row_ge(0, S[4], S[0], syms, 4);
    //S[4] = row_ge(1, S[4], S[1], syms, 4);
    gj_reduce(S, n, syms);
    std::cout << "S:\n" << pprint(S, n, syms);

    std::cout << "symbols:\n";
    for (symidx s: syms.all()) {
        std::cout << syms.id(s) << ": ";
        auto v = vals[s];
        if (v) std::cout << *v << "\n";
        else std::cout << pprint(syms[s], syms) << "\n";
    }
}


int main() {
    //demo_store_eval();
    //demo_msparse_random();
    demo_sym_matrix();
}

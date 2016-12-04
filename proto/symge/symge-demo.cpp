// bloody CMake
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include <util/optional.hpp>

#include "symbolic.hpp"
#include "msparse.hpp"

using namespace sym;

// Identifier name picking helper routines

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

class id_maker {
private:
    std::unordered_set<std::string> ids;

public:
    static std::string next_id(std::string s) {
        unsigned l = s.size();
        if (l==0) return "a";

        unsigned i = l-1;

        char l0='a', l1='z';
        char u0='A', u1='Z';
        char d0='0', d1='9';
        for (;;) {
            char& c = s[i];
            if ((c>=l0 && c<l1) || (c>=u0 && c<u1) || (c>=d0 && c<d1)) {
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

    template <typename... Args>
    std::string operator()(Args&&... elements) {
        std::string name = join("",std::forward<Args>(elements)...);
        if (name.empty()) name = "a";

        while (ids.count(name)) {
            name = next_id(name);
        }
        ids.insert(name);
        return name;
    }

    void reserve(std::string name) {
        ids.insert(std::move(name));
    }
};

// Output helper functions

template <typename X>
std::ostream& operator<<(std::ostream& o, const optional<X>& x) {
    return x? o << *x: o << "nothing";
}

template <typename X>
std::ostream& operator<<(std::ostream& o, const msparse::matrix<X>& m) {
    for (unsigned r = 0; r<m.nrow(); ++r) {
        o << '|';
        for (unsigned c = 0; c<m.ncol(); ++c) {
            if (c==m.augcol()) o << " | ";
            o << std::setw(12) << m[r][c];
        }
        o << " |\n";
    }
    return o;
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

std::ostream& operator<<(std::ostream& o, const symbol_table& syms) {
    for (unsigned i = 0; i<syms.size(); ++i) {
        symbol s = syms[i];
        if (s.def()) o << s << ": " << s.def() << "\n";
    }
    return o;
}

// Symbolic GE

using symmrow = msparse::mrow<symbol>;
using symmatrix = msparse::matrix<symbol>;

// Returns q[c]*p - p[c]*q
template <typename DefineSym>
symmrow row_reduce(unsigned c, const symmrow& p, const symmrow& q, DefineSym define_sym) {
    if (p.index(c)==p.npos || q.index(c)==q.npos) throw std::runtime_error("improper row GE");

    symmrow u;
    symbol x = q[c];
    symbol y = p[c];

    auto piter = p.begin();
    auto qiter = q.begin();
    unsigned pj = piter->first;
    unsigned qj = qiter->first;

    while (piter!=p.end() || qiter!=q.end()) {
        unsigned j = std::min(pj, qj);
        symbol_term t1, t2;

        if (j==pj) {
            t1 = x*piter->second;
            ++piter;
            pj = piter==p.end()? p.npos: piter->first;
        }
        if (j==qj) {
            t2 = y*qiter->second;
            ++qiter;
            qj = qiter==q.end()? q.npos: qiter->first;
        }
        if (j!=c) {
            u.push_back({j, define_sym(t1-t2)});
        }
    }
    return u;
}

// Actual GE reduction.
// Note: ncol: number of columns before augmentation.
// Another note: if we're going to preference diagonal pivots
// (owing to application), then priority queue is overkill.
template <typename DefineSym>
void gj_reduce(symmatrix& A, unsigned ncol, DefineSym define_sym) {
    struct pq_entry {
        unsigned key; // first non-zero column after pivot column
        unsigned mincol;
        unsigned row;
    };

    struct pq_order_t {
        bool operator()(const pq_entry& a, const pq_entry& b) const {
            // The last condition preferences pivots on diagonal elements.
            return a.key>b.key ||
                  (a.key==b.key && a.mincol<b.mincol) ||
                  (a.key==b.key && a.mincol==b.mincol && a.row!=a.key);
        }
    };

    std::priority_queue<pq_entry, std::vector<pq_entry>, pq_order_t> pq;

    unsigned n = A.nrow();
    for (unsigned i = 0; i<n; ++i) {
        unsigned c = A[i].mincol();
        if (c<ncol) pq.push({c, c, i});
    }

    while (!pq.empty()) {
        pq_entry pick = pq.top();
        pq.pop();

        unsigned col = pick.key;
        auto r1 = pick.row;

        while (!pq.empty() && pq.top().key==pick.key) {
            pq_entry top = pq.top();
            pq.pop();

            auto r2 = top.row;
            A[r2] = row_reduce(col, A[r2], A[r1], define_sym);

            unsigned c = A[r2].mincol_after(col);
            if (c<ncol) pq.push({c, A[r2].mincol(), r2});
        }
        unsigned c = A[r1].mincol_after(col);
        if (c<ncol) pq.push({c, pick.mincol, r1});
    }
}

// Validation

template <typename Rng>
msparse::matrix<double> make_random_matrix(unsigned n, double density, Rng& R) {
    std::uniform_real_distribution<double> U;
    msparse::matrix<double> M(n, n);

    for (unsigned i = 0; i<n; ++i) {
        for (unsigned j = 0; j<n; ++j) {
            if (i!=j && U(R)>density) continue;
            double u = U(R);
            M[i][j] = i==j? n*(1+u): u-0.5;
        }
    }
    return M;
}

struct symge_stats {
    unsigned n;
    unsigned nnz;
    unsigned nmul;
    unsigned nsub;
    unsigned nsym;
    double relerr;
};

template <typename Rng>
symge_stats run_symge_validation(Rng& R, unsigned n, bool debug = false) {
    std::uniform_real_distribution<double> U;
    symge_stats stats = { n, 0, 0, 0, 0, 0. };

    msparse::matrix<double> A = make_random_matrix(n, 2.0/(n+1), R);
    symmatrix S(n, n);

    symbol_table syms;
    store values(syms);
    id_maker make_id;

    // make symbolic matrix with same sparsity pattern as M
    for (unsigned i = 0; i<A.nrow(); ++i) {
        symmrow srow;
        for (const auto& a: A[i]) {
            unsigned j = a.first;
            auto s = syms.define(make_id("a", i, j));
            values[s] = a.second;
            srow.push_back({j, s});
            ++stats.nnz;
        }
        S[i] = srow;
    }

    // pick x and compute b = Ax
    std::vector<double> x(n);
    for (auto& elem: x) elem = 10*U(R);

    std::vector<double> b(n);
    mul_dense(A, x, b);

    // augment M and S with rhs
    A.augment(b);
    std::vector<symbol> rhs;
    for (unsigned i = 0; i<n; ++i) {
        auto s = syms.define(make_id("b", i));
        rhs.push_back(s);
        values[s] = b[i];
    }
    S.augment(rhs);

    if (debug) {
        std::cerr << "A|b:\n" << A << "\n";
        std::cerr << "S:\n" << S << "\n";
    }

    // perform GE
    auto nprim_sym = syms.size();
    gj_reduce(S, n, [&](const symbol_def& def) { return syms.define(make_id("t00"), def); });
    stats.nsym = syms.size()-nprim_sym;
    if (debug) {
        std::cerr << "reduced S:\n" << S << "\n";
        std::cerr << "definitions:\n" << syms << "\n";
    }

    // validate: compute solution y from reduced S and symbol defs
    std::vector<double> y(n);
    double maxerr = 0;
    double maxx = 0;
    for (unsigned i = 0; i<n; ++i) {
        const symmrow& row = S[i];
        if (row.size()!=2 || row.maxcol()!=n)
            throw std::runtime_error("unexpected matrix layout!");

        unsigned idx = row.get(0).first;
        symbol coeff = row.get(0).second;
        symbol rhs = row.get(1).second;

        if (debug) {
            std::cerr << "y" << idx << " = " << rhs << "/" << coeff << "\n";
            std::cerr << "   = " << values.evaluate(rhs).get() << "/" << values.evaluate(coeff).get() << "\n";
        }
        y[idx] = values.evaluate(rhs).get()/values.evaluate(coeff).get();
        maxerr = std::max(maxerr, std::abs(y[idx]-x[idx]));
        maxx = std::max(maxx, std::abs(x[idx]));
    }
    stats.relerr = maxerr/maxx;
    stats.nsub = values.sub_count;
    stats.nmul = values.mul_count;
    if (debug) {
        std::cerr << "x:\n" << sepval(", ", x) << "\n";
        std::cerr << "computed solution:\n" << sepval(", ", y) << "\n";
        std::cerr << "operation count: " << values.mul_count << " mul; " << values.sub_count << " sub\n";
    }
    return stats;
}

int main(int argc, const char** argv) {
    bool debug = false;
    bool verbose = false;

    for (const char** arg=argv+1; arg!=argv+argc; ++arg) {
        if (!std::strcmp("-v", *arg)) {
            verbose = true;
        }
        if (!std::strcmp("-d", *arg)) {
            debug = true;
        }
        if (!std::strcmp("-h", *arg)) {
            std::cerr << "usage: symge-demo [-v] [-d]\n";
            return 0;
        }
    }

    if (verbose) {
        char line[80];
        std::snprintf(line, sizeof(line), "%10s%10s%10s%10s%10s%10s\n",
            "n", "nnz", "nmul", "nsub", "nsym", "relerr");
        std::cout << line;
    }

    std::minstd_rand R;
    for (unsigned n = 1; n<=10; ++n) {
        for (unsigned k =1; k<=100; ++k) {
            auto stats = run_symge_validation(R, n, debug);
            if (verbose) {
                char line[80];
                std::snprintf(line, sizeof(line), "%10d%10d%10d%10d%10d%10lf\n",
                    stats.n, stats.nnz, stats.nmul, stats.nsub, stats.nsym, stats.relerr);

                std::cout << line;
            }
            assert(stats.relerr<1e-6);
        }
    }

    return 0;
}

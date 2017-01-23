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

// Return string representation of all arguments, separated by `sep`.
template <typename... Args>
std::string join(const std::string& sep, Args&&... items) {
    std::stringstream ss;
    join_impl(ss, sep, std::forward<Args>(items)...);
    return ss.str();
}

// Maintain a collection of unique identifiers.
class id_maker {
private:
    std::unordered_set<std::string> ids;

public:
    // Find the next string lexicographically after argument, allowing only
    // ASCII letters and digits to change.
    static std::string next_id(std::string s) {
        unsigned l = s.size();
        if (l==0) return "a";

        unsigned i = l-1;

        constexpr char l0='a', l1='z';
        constexpr char u0='A', u1='Z';
        constexpr char d0='0', d1='9';
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

    // Make a new identifier based on the given arguments.
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
struct sepval_proxy {
    const Sep& sep;
    const V& v;

    sepval_proxy(const Sep& sep, const V& v): sep(sep), v(v) {}

    friend std::ostream& operator<<(std::ostream& O, const sepval_proxy& sv) {
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
sepval_proxy<Sep,V> sepval(const Sep& sep, const V& v) { return sepval_proxy<Sep,V>(sep, v); }

std::ostream& operator<<(std::ostream& o, const symbol_table& syms) {
    for (unsigned i = 0; i<syms.size(); ++i) {
        symbol s = syms[i];
        if (s.def()) o << s << ": " << s.def() << "\n";
    }
    return o;
}

// Symbolic GE

using sym_row = msparse::row<symbol>;
using sym_matrix = msparse::matrix<symbol>;

// Returns q[c]*p - p[c]*q
template <typename DefineSym>
sym_row row_reduce(unsigned c, const sym_row& p, const sym_row& q, DefineSym& define_sym) {
    if (p.index(c)==p.npos || q.index(c)==q.npos) throw std::runtime_error("improper row GE");

    sym_row u;
    symbol x = q[c];
    symbol y = p[c];

    auto piter = p.begin();
    auto qiter = q.begin();
    unsigned pj = piter->col;
    unsigned qj = qiter->col;

    while (piter!=p.end() || qiter!=q.end()) {
        unsigned j = std::min(pj, qj);
        symbol_term t1, t2;

        if (j==pj) {
            t1 = x*piter->value;
            ++piter;
            pj = piter==p.end()? p.npos: piter->col;
        }
        if (j==qj) {
            t2 = y*qiter->value;
            ++qiter;
            qj = qiter==q.end()? q.npos: qiter->col;
        }
        if (j!=c) {
            u.push_back({j, define_sym(t1-t2)});
        }
    }
    return u;
}

// Actual GE reduction.
template <typename DefineSym>
void gj_reduce_simple(sym_matrix& A, unsigned ncol, const DefineSym& define_sym) {
    // Expect A to be a (possibly column-augmented) diagonally dominant
    // matrix; take diagonal elements as pivots.
    if (A.nrow()>A.ncol()) throw std::runtime_error("improper matrix for reduction");

    for (unsigned pivrow = 0; pivrow<A.nrow(); ++pivrow) {
        unsigned pivcol = pivrow;

        for (unsigned i = 0; i<A.nrow(); ++i) {
            if (i==pivrow || A[i].index(pivcol)==msparse::row_npos) continue;

            A[i] = row_reduce(pivcol, A[i], A[pivrow], define_sym);
        }
    }
}

// simple greedy estimate based on immediate fill cost
double estimate_cost(const sym_matrix& A, unsigned p) {
    struct count_sym_mul {
        mutable unsigned nmul = 0;
        mutable unsigned nfill = 0;
        symbol operator()(symbol_term_diff t) {
            bool l = t.left, r = t.right;
            nmul += l+r;
            nfill += r&!l;
            return symbol{};
        }
    };

    count_sym_mul counter;
    for (unsigned i = 0; i<A.nrow(); ++i) {
        if (i==p || A[i].index(p)==msparse::row_npos) continue;

        row_reduce(p, A[i], A[p], counter);
    }
    return counter.nfill;
}

template <typename DefineSym>
void gj_reduce(sym_matrix& A, unsigned ncol, const DefineSym define_sym) {
    // Expect A to be a (possibly column-augmented) diagonally dominant
    // matrix; take diagonal elements as pivots.
    if (A.nrow()>A.ncol()) throw std::runtime_error("improper matrix for reduction");

    std::vector<unsigned> pivots;
    for (unsigned r = 0; r<A.nrow(); ++r) {
        pivots.push_back(r);
    }

    std::vector<double> cost(pivots.size());

    while (!pivots.empty()) {
        for (unsigned i = 0; i<pivots.size(); ++i) {
            cost[pivots[i]] = estimate_cost(A, pivots[i]);
        }

        std::sort(pivots.begin(), pivots.end(),
            [&](unsigned r1, unsigned r2) { return cost[r1]>cost[r2]; });

        unsigned pivrow = pivots.back();
        pivots.erase(std::prev(pivots.end()));

        unsigned pivcol = pivrow;

        for (unsigned i = 0; i<A.nrow(); ++i) {
            if (i==pivrow || A[i].index(pivcol)==msparse::row_npos) continue;

            A[i] = row_reduce(pivcol, A[i], A[pivrow], define_sym);
        }
    }
}

// Validation

template <typename Rng>
msparse::matrix<double> make_random_matrix(unsigned n, double row_nnz, Rng& R) {
    std::uniform_real_distribution<double> U;
    msparse::matrix<double> M(n, n);

    row_nnz = std::min(row_nnz, n-1.5); // always allow some chance of zeroes.
    double density = n==1? 1: row_nnz/(n-1);
    for (unsigned i = 0; i<n; ++i) {
        for (unsigned j = 0; j<n; ++j) {
            if (i!=j && U(R)>density) continue;
            double u = U(R);
            M[i][j] = i==j? 1+0.2*u: (u-0.5)/n;
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
symge_stats run_symge_validation(Rng& R, unsigned n, double row_nnz, bool use_estimator, bool debug = false) {
    std::uniform_real_distribution<double> U;
    symge_stats stats = { n, 0, 0, 0, 0, 0. };

    msparse::matrix<double> A = make_random_matrix(n, row_nnz, R);
    sym_matrix S(n, n);

    symbol_table syms;
    store values(syms);
    id_maker make_id;

    // make symbolic matrix with same sparsity pattern as M
    for (unsigned i = 0; i<A.nrow(); ++i) {
        sym_row srow;
        for (const auto& a: A[i]) {
            unsigned j = a.col;
            auto s = syms.define(make_id("a", i, j));
            values[s] = a.value;
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
    auto define_sym = [&](const symbol_def& def) { return syms.define(make_id("t00"), def); };
    if (use_estimator) {
        gj_reduce(S, n, define_sym);
    }
    else {
        gj_reduce_simple(S, n, define_sym);
    }
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
        const sym_row& row = S[i];
        if (row.size()!=2 || row.maxcol()!=n)
            throw std::runtime_error("unexpected matrix layout!");

        unsigned idx = row.get(0).col;
        symbol coeff = row.get(0).value;
        symbol rhs = row.get(1).value;

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

void usage() {
    std::cout << "Usage: symge-demo [-v] [-d] [-e]\n"
              << "Options:\n"
              << "  -v      Verbose output: display statistics as table\n"
              << "  -d      Debug output to stderr\n"
              << "  -e      Use a simple cost estimator to select pivots\n"
              << "  -n N    Specify max matrix size (default 10)\n"
              << "  -k N    Number of matrices to test of each size (default 100)\n"
              << "  -r R    Mean number of off-diagonal nonzero elements per row (default 2.0)\n";
}

void usage_error(const std::string& msg) {
    std::cerr << "symge-demo: " << msg << "\n"
              << "Try 'symge-demo -h' for more information.\n";
}

int main(int argc, const char** argv) {
    bool debug = false;
    bool verbose = false;
    bool use_estimator = false;
    int max_n = 10;
    int count = 100;
    double row_mean_nnz = 2.0;

    for (const char** arg=argv+1; arg!=argv+argc; ++arg) {
        if (!std::strcmp("-v", *arg)) {
            verbose = true;
        }
        else if (!std::strcmp("-d", *arg)) {
            debug = true;
        }
        else if (!std::strcmp("-e", *arg)) {
            use_estimator = true;
        }
        else if (!std::strcmp("-n", *arg)) {
            if (!arg[1]) {
                usage_error("expected argument for '-n'");
                return 2;
            }
            max_n = std::atoi(*++arg);
        }
        else if (!std::strcmp("-r", *arg)) {
            if (!arg[1]) {
                usage_error("expected argument for '-r'");
                return 2;
            }
            row_mean_nnz = std::stod(*++arg);
        }
        else if (!std::strcmp("-k", *arg)) {
            if (!arg[1]) {
                usage_error("expected argument for '-k'");
                return 2;
            }
            count = std::atoi(*++arg);
        }
        else if (!std::strcmp("-h", *arg)) {
            usage();
            return 0;
        }
        else {
            std::cerr << "symge-demo: unrecognized argument\n"
                      << "Try 'symge-demo -h' for more information.\n";
            return 2;
        }
    }

    if (verbose) {
        char line[80];
        std::snprintf(line, sizeof(line), "%10s%10s%10s%10s%10s%10s\n",
            "n", "nnz", "nmul", "nsub", "nsym", "relerr");
        std::cout << line;
    }

    std::minstd_rand R;
    for (unsigned n = 1; n<=max_n; ++n) {
        for (unsigned k = 1; k<=count; ++k) {
            auto stats = run_symge_validation(R, n, row_mean_nnz, use_estimator, debug);
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

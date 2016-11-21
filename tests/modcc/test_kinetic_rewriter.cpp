#include "test.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "expression.hpp"
#include "kinrewriter.hpp"
#include "parser.hpp"

// Simple algebraic term expansion/collection routines.

namespace alg {
    template <typename Prim, typename Num>
    struct collectable {
        Prim prim;
        Num n;

        collectable(): n(0) {}
        collectable(const Prim& prim): prim(prim), n(1) {}
        collectable(const Prim& prim, Num n): prim(prim), n(n) {}

        friend bool operator<(const collectable& a, const collectable& b) {
            return a.prim<b.prim || (a.prim==b.prim && a.n<b.n);
        }

        friend bool operator==(const collectable& a, const collectable& b) {
            return a.prim==b.prim && a.n==b.n;
        }

        friend bool operator!=(const collectable& a, const collectable& b) {
            return !(a==b);
        }

        void invert() { n = -n; }
    };

    template <typename Prim, typename Num>
    void collect(std::vector<collectable<Prim, Num>>& xs) {
        std::sort(xs.begin(), xs.end());
        if (xs.size()<2) return;

        std::vector<collectable<Prim, Num>> coll;
        coll.push_back(xs[0]);

        for (unsigned j=1; j<xs.size(); ++j) {
            const auto& x = xs[j];
            if (coll.back().prim!=x.prim) {
                coll.push_back(x);
            }
            else {
                coll.back().n += x.n;
            }
        }

        xs.clear();
        for (auto& t: coll) {
            if (t.n!=0) xs.push_back(std::move(t));
        }
    }

    template <typename Prim, typename Num>
    void invert(std::vector<collectable<Prim, Num>>& xs) {
        for (auto& x: xs) x.invert();
    }

    struct prodterm {
        using factor = collectable<std::string, double>;

        std::vector<factor> factors;

        prodterm() {}
        explicit prodterm(factor f): factors(1, f) {}
        explicit prodterm(const std::vector<factor>& factors): factors(factors) {}

        void collect() { alg::collect(factors); }
        void invert() { alg::invert(factors); }
        bool empty() const { return factors.empty(); }

        prodterm& operator*=(const prodterm& x) {
            factors.insert(factors.end(), x.factors.begin(), x.factors.end());
            collect();
            return *this;
        }

        prodterm& operator/=(const prodterm& x) {
            prodterm recip(x);
            recip.invert();
            return *this *= recip;
        }

        prodterm& pow(double n) {
            for (auto& f: factors) f.n *= n;
            return *this;
        }

        friend prodterm pow(const prodterm& pt, double n) {
            prodterm x(pt);
            return x.pow(n);
        }

        friend prodterm operator*(const prodterm& a, const prodterm& b) {
            prodterm p(a);
            return p *= b;
        }

        friend prodterm operator/(const prodterm& a, const prodterm& b) {
            prodterm p(a);
            return p /= b;
        }

        friend bool operator<(const prodterm& p, const prodterm& q) {
            return p.factors<q.factors;
        }

        friend bool operator==(const prodterm& p, const prodterm& q) {
            return p.factors==q.factors;
        }

        friend bool operator!=(const prodterm& p, const prodterm& q) {
            return !(p==q);
        }

        friend std::ostream& operator<<(std::ostream& o, const prodterm& x) {
            if (x.empty()) return o << "1";

            int nf = 0;
            for (const auto& f: x.factors) {
                o << (nf++?"*":"") << f.prim;
                if (f.n!=1) o << '^' << f.n;
            }
            return o;
        }
    };

    struct prodsum {
        using term = collectable<prodterm, double>;
        std::vector<term> terms;

        prodsum() {}

        prodsum(const prodterm& pt): terms(1, pt) {}
        prodsum(prodterm&& pt): terms(1, std::move(pt)) {}
        explicit prodsum(double x, const prodterm& pt = prodterm()): terms(1, term(pt, x)) {}

        void collect() { alg::collect(terms); }
        void invert() { alg::invert(terms); }
        bool empty() const { return terms.empty(); }

        prodsum& operator+=(const prodsum& x) {
            terms.insert(terms.end(), x.terms.begin(), x.terms.end());
            collect();
            return *this;
        }

        prodsum& operator-=(const prodsum& x) {
            prodsum neg(x);
            neg.invert();
            return *this += neg;
        }

        prodsum operator-() const {
            prodsum neg(*this);
            neg.invert();
            return neg;
        }

        // Distribution:
        prodsum& operator*=(const prodsum& x) {
            if (terms.empty()) return *this;
            if (x.empty()) {
                terms.clear();
                return *this;
            }

            std::vector<term> distrib;
            for (const auto& a: terms) {
                for (const auto& b: x.terms) {
                    distrib.emplace_back(a.prim*b.prim, a.n*b.n);
                }
            }

            terms = distrib;
            collect();
            return *this;
        }

        prodsum recip() const {
            prodterm rterm;
            double rcoef = 1;

            if (terms.size()==1) {
                rcoef = terms.front().n;
                rterm = terms.front().prim;
            }
            else {
                // Make an opaque term from denominator if not a simple product.
                rterm = as_opaque_term();
            }
            rterm.invert();
            return prodsum(1.0/rcoef, rterm);
        }

        prodsum& operator/=(const prodsum& x) {
            return *this *= x.recip();
        }

        prodterm as_opaque_term() const {
            return prodterm("("+to_string(*this)+")");
        }

        friend prodsum operator+(const prodsum& a, const prodsum& b) {
            prodsum p(a);
            return p += b;
        }

        friend prodsum operator-(const prodsum& a, const prodsum& b) {
            prodsum p(a);
            return p -= b;
        }

        friend prodsum operator*(const prodsum& a, const prodsum& b) {
            prodsum p(a);
            return p *= b;
        }

        friend prodsum operator/(const prodsum& a, const prodsum& b) {
            prodsum p(a);
            return p /= b;
        }

        friend std::ostream& operator<<(std::ostream& o, const prodsum& x) {
            if (x.terms.empty()) return o << "0";

            bool first = true;
            for (const auto& t: x.terms) {
                double coef = t.n;
                const prodterm& pd = t.prim;

                const char* prefix = coef<0? "-": first? "": "+";
                if (coef<0) coef = -coef;

                o << prefix;
                if (pd.empty()) {
                    o << coef;
                }
                else {
                    if (coef!=1) o << coef << '*';
                    o << pd;
                }
                first = false;
            }
            return o;
        }

        bool is_scalar() const {
            return terms.empty() || (terms.size()==1 && terms.front().prim.empty());
        }

        double first_coeff() const {
            return terms.empty()? 0: terms.front().n;
        }

        friend bool operator<(const prodsum& p, const prodsum& q) {
            return p.terms<q.terms;
        }

        friend bool operator==(const prodsum& p, const prodsum& q) {
            return p.terms==q.terms;
        }

        friend bool operator!=(const prodsum& p, const prodsum& q) {
            return !(p==q);
        }

    };

    prodsum int_pow_expand(const prodsum& base, unsigned n) {
        switch (n) {
        case 0:
            return prodsum(1);
        case 1:
            return base;
        default:
            return int_pow_expand(base, n/2)*int_pow_expand(base, n/2)*int_pow_expand(base, n%2);
        }
    }

    prodsum pow_expand(const prodsum& base, double n) {
        if (n==0) {
            return prodsum(1);
        }
        else if (n==1) {
            return base;
        }
        else if (base.is_scalar()) {
            return prodsum(std::pow(base.first_coeff(), n));
        }
        else if (base.terms.size()==1) {
            double c = std::pow(base.terms.front().n, n);
            prodterm t = pow(base.terms.front().prim, n);
            return prodsum(c, t);
        }
        else if (n<0) {
            return pow_expand(base.recip(), -n);
        }
        else if (n!=std::floor(n)) {
            prodterm pt = base.as_opaque_term();
            return pow(base.as_opaque_term(),n);
        }
        else {
            return int_pow_expand(base, static_cast<unsigned>(n));
        }
    }
} // namespace alg


using id_prodsum_map = std::map<std::string, alg::prodsum>;

alg::prodsum expand_expression(Expression* e, const id_prodsum_map& exmap) {
    using namespace alg;

    if (const auto& n = e->is_number()) {
        return prodsum(n->value());
    }
    else if (const auto& c = e->is_function_call()) {
        std::stringstream rep(c->name());
        rep << '(';
        bool first = true;
        for (const auto& arg: c->args()) {
            if (!first) rep << ',';
            rep << expand_expression(arg.get(), exmap);
            first = false;
        }
        rep << ')';
        return prodterm(rep.str());
    }
    else if (const auto& i = e->is_identifier()) {
        std::string k = i->spelling();
        auto x = exmap.find(k);
        return x!=exmap.end()? x->second: prodterm(k);
    }
    else if (const auto& b = e->is_binary()) {
        prodsum lhs = expand_expression(b->lhs(), exmap);
        prodsum rhs = expand_expression(b->rhs(), exmap);

        switch (b->op()) {
        case tok::plus:
            return lhs+rhs;
        case tok::minus:
            return lhs-rhs;
        case tok::times:
            return lhs*rhs;
        case tok::divide:
            return lhs/rhs;
        case tok::pow:
            if (!rhs.is_scalar()) {
                // make an opaque term for this case (i.e. too hard to simplify)
                return prodterm("("+to_string(lhs)+")^("+to_string(rhs)+")");
            }
            else return pow_expand(lhs, rhs.first_coeff());
        default:
            throw std::runtime_error("unrecognized binop");
        }
    }
    else if (const auto& u = e->is_unary()) {
        prodsum inner = expand_expression(u->expression(), exmap);
        switch (u->op()) {
        case tok::minus:
            return -inner;
        case tok::exp:
            return prodterm("exp("+to_string(inner)+")");
        case tok::log:
            return prodterm("log("+to_string(inner)+")");
        case tok::sin:
            return prodterm("sin("+to_string(inner)+")");
        case tok::cos:
            return prodterm("cos("+to_string(inner)+")");
        default:
            throw std::runtime_error("unrecognized binop");
        }
    }
    else {
        throw std::runtime_error("unexpected expression type");
    }
}

std::map<std::string, alg::prodsum> expand_assignments(stmt_list_type& stmts) {
    using namespace alg;

    // This is 'just a test', so don't try to be complete: functions are
    // left unexpanded; procedure calls are ignored.

    std::map<std::string, prodsum> exmap;

    for (const auto& stmt: stmts) {
        if (auto assign = stmt->is_assignment()) {
            auto lhs = assign->lhs();
            std::string key;
            if (auto deriv = lhs->is_derivative()) {
                key = deriv->spelling()+"'";
            }
            else if (auto id = lhs->is_identifier()) {
                key = id->spelling();
            }
            else {
                // don't know what we have here! skip.
                continue;
            }

            exmap[key] = expand_expression(assign->rhs(), exmap);
        }
    }
    return exmap;
}

stmt_list_type& proc_statements(Expression *e) {
    if (!e || !e->is_symbol() || ! e->is_symbol()->is_procedure()) {
        throw std::runtime_error("not a procedure");
    }

    return e->is_symbol()->is_procedure()->body()->statements();
}


inline symbol_ptr state_var(const char* name) {
    auto v = make_symbol<VariableExpression>(Location(), name);
    v->is_variable()->state(true);
    return v;
}

inline symbol_ptr assigned_var(const char* name) {
    return make_symbol<VariableExpression>(Location(), name);
}

static const char* kinetic_abc =
    "KINETIC kin {             \n"
    "    u = 3                 \n"
    "    ~ a <-> b (u, v)      \n"
    "    u = 4                 \n"
    "    ~ b <-> 3b + c (u, v) \n"
    "}                         \n";

static const char* derivative_abc =
    "DERIVATIVE deriv {        \n"
    "    a' = -3*a + b*v       \n"
    "    LOCAL rev2            \n"
    "    rev2 = c*b^3*v        \n"
    "    b' = 3*a - v*b + 8*b - 2*rev2\n"
    "    c' = 4*b - rev2       \n"
    "}                         \n";

TEST(KineticRewriter, equiv) {
    auto visitor = make_unique<KineticRewriter>();
    auto kin = Parser(kinetic_abc).parse_procedure();
    auto deriv = Parser(derivative_abc).parse_procedure();

    ASSERT_NE(nullptr, kin);
    ASSERT_NE(nullptr, deriv);
    ASSERT_TRUE(kin->is_symbol() && kin->is_symbol()->is_procedure());
    ASSERT_TRUE(deriv->is_symbol() && deriv->is_symbol()->is_procedure());

    auto kin_weak = kin->is_symbol()->is_procedure();
    scope_type::symbol_map globals;
    globals["kin"] = std::move(kin);
    globals["a"] = state_var("a");
    globals["b"] = state_var("b");
    globals["c"] = state_var("c");
    globals["u"] = assigned_var("u");
    globals["v"] = assigned_var("v");

    kin_weak->semantic(globals);
    kin_weak->accept(visitor.get());

    auto kin_deriv = visitor->as_procedure();

    if (g_verbose_flag) {
        std::cout << "derivative procedure:\n" << deriv->to_string() << "\n";
        std::cout << "kin procedure:\n" << kin_weak->to_string() << "\n";
        std::cout << "rewritten kin procedure:\n" << kin_deriv->to_string() << "\n";
    }

    auto deriv_map = expand_assignments(proc_statements(deriv.get()));
    auto kin_map = expand_assignments(proc_statements(kin_deriv.get()));

    if (g_verbose_flag) {
        std::cout << "derivtive assignments (canonical):\n";
        for (const auto&p: deriv_map) {
            std::cout << p.first << ": " << p.second << "\n";
        }
        std::cout << "rewritten kin assignments (canonical):\n";
        for (const auto&p: kin_map) {
            std::cout << p.first << ": " << p.second << "\n";
        }
    }

    EXPECT_EQ(deriv_map["a'"], kin_map["a'"]);
    EXPECT_EQ(deriv_map["b'"], kin_map["b'"]);
    EXPECT_EQ(deriv_map["c'"], kin_map["c'"]);
}


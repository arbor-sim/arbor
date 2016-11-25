#pragma once

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

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

    prodterm pow(double n) const {
        prodterm x(*this);
        for (auto& f: x.factors) f.n *= n;
        return x;
    }

    friend prodterm pow(const prodterm& pt, double n) {
        return pt.pow(n);
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
        std::stringstream s;
        s << '(' << *this << ')';
        return prodterm(s.str());
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

    prodsum int_pow(unsigned n) const {
        switch (n) {
        case 0:
            return prodsum(1);
        case 1:
            return *this;
        default:
            return int_pow(n/2)*int_pow(n/2)*int_pow(n%2);
        }
    }

    prodsum pow(double n) const {
        if (n==0) {
            return prodsum(1);
        }
        else if (n==1) {
            return *this;
        }
        else if (is_scalar()) {
            return prodsum(std::pow(first_coeff(), n));
        }
        else if (terms.size()==1) {
            const auto& t = terms.front();
            return prodsum(std::pow(t.n, n), t.prim.pow(n));
        }
        else if (n<0) {
            return recip().pow(-n);
        }
        else if (n!=std::floor(n)) {
            return as_opaque_term().pow(n);
        }
        else {
            return int_pow(static_cast<unsigned>(n));
        }
    }
};

} // namespace alg

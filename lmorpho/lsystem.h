#pragma once

#include <random>

#include "morphology.h"

struct lsys_param;

using lsys_generator = std::minstd_rand;

morphology generate_morphology(const lsys_param& P, lsys_generator& g);


// Random distribution used in the specification of L-system parameters.
// It can be a constant, uniform over an interval, or truncated normal.
// Note that mixture distributions of the above should be, but isn't yet,
// implemented.

struct lsys_distribution {
    enum lsys_kind {
        constant, uniform, normal
    };

    using result_type = double;
    struct param_type {
        lsys_kind kind;
        result_type lb; // lower bound
        result_type ub; // upper bound (used for only non-constant)
        result_type mu; // mean (used only for truncated Gaussian)
        result_type sigma; // s.d. (used only for truncated Gaussian)

        bool operator==(const param_type &p) const {
            if (kind!=p.kind) return false;
            switch (kind) {
            case lsys_kind::constant:
                return lb == p.lb;
            case lsys_kind::uniform:
                return lb == p.lb && ub == p.ub;
            case lsys_kind::normal:
                return lb == p.lb && ub == p.ub && mu == p.mu && sigma == p.sigma;
            }
            return false;
        }

        bool operator!=(const param_type &p) const {
            return !(*this==p);
        }

        // zero or one parameter => constant
        param_type():
            kind(lsys_kind::constant), lb(0.0) {}

        param_type(result_type x):
            kind(lsys_kind::constant), lb(x) {}

        // two paramter => uniform
        param_type(result_type lb, result_type ub):
            kind(lsys_kind::uniform), lb(lb), ub(ub) {}

        // four paramter => truncated normal
        param_type(result_type lb, result_type ub, result_type mu, result_type sigma):
            kind(lsys_kind::normal), lb(lb), ub(ub), mu(mu), sigma(sigma) {}
    };

    lsys_distribution(const param_type& p): p_(p) { init(); }

    lsys_distribution(): p_() { init(); }
    lsys_distribution(result_type x): p_(x) { init(); }
    lsys_distribution(result_type lb, result_type ub): p_(lb, ub) { init(); }
    lsys_distribution(result_type lb, result_type ub, result_type mu, result_type sigma):
        p_(lb, ub, mu, sigma)
    {
        init();
    }

    param_type param() const { return p_; }
    void param(const param_type& pbis) { p_=pbis; }

    result_type min() const { return p_.lb; }
    result_type max() const {
        return p_.kind==lsys_kind::constant? min(): p_.ub;
    }

    bool operator==(const lsys_distribution& y) const {
        return p_==y.p_;
    }

    bool operator!=(const lsys_distribution& y) const {
        return p_!=y.p_;
    }

    // omitting out ostream, istream methods
    // (...)

    template <typename Gen>
    result_type operator()(Gen& g, const param_type &p) const {
        lsys_distribution dist(p);
        return dist(g);
    }

    template <typename Gen>
    result_type operator()(Gen& g) const {
        switch (p_.kind) {
        case constant:
            return p_.lb;
        case uniform:
            return uniform_dist(g);
        case normal:
            // TODO: replace with a robust truncated normal generator!
            for (;;) {
                result_type r = normal_dist(g);
                if (r>=p_.lb && r<=p_.ub) return r;
            }
        }
        return 0;
    }

private:
    param_type p_;
    mutable std::uniform_real_distribution<double> uniform_dist;
    mutable std::normal_distribution<double> normal_dist;

    void init() {
        switch (p_.kind) {
        case uniform:
            uniform_dist = std::uniform_real_distribution<double>(p_.lb, p_.ub);
            break;
        case normal:
            normal_dist = std::normal_distribution<result_type>(p_.mu, p_.sigma);
            break;
        default: ;
        }
    }
};


// Fields below refer to equivalent L-Neuron parameters in parantheses.
// Note that for consistency with literature, parameters describe diameters,
// while the internal representation uses radii.
//
// Default parameters are taken from Burke 1992 and Ascoli 2001, with
// the caveat that mixture representations have been simplified.

struct lsys_param {
    // Soma diameter [µm].
    lsys_distribution diam_soma = { 47.0, 65.5, 57.6, 5.0 };

    // Number of dendritic trees (rounded to nearest int). (Ntree)
    lsys_distribution n_tree = { 1.0 };

    // Initial dendrite diameter [µm]. (Dstem)
    lsys_distribution diam_initial = { 5.0, 8.0 };

    // Dendrite step length [µm]. (ΔL)
    lsys_distribution length_step = { 25 };

    // Dendrites grow along x-axis after application of (instrinsic) rotations;
    // roll (about x-axis) is applied first, followed by pitch (about y-axis).

    // Initial roll (intrinsic rotation about x-axis) [degrees]. (Taz)
    lsys_distribution roll_initial = { -180.0, 180.0 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees]. (Tel)
    lsys_distribution pitch_initial = { 0.0, 180.0 };

    // Tortuousness: roll within segment over ΔL [degrees]. (Eaz)
    lsys_distribution roll_segment = { -180.0, 180.0 };

    // Tortuousness: pitch within segment over ΔL [degrees]. (Eel)
    lsys_distribution pitch_segment = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate. (TPRB)
    lsys_distribution taper = { -1.25e-3 };

    // Branching torsion: roll at branch point [degrees]. (Btor)
    lsys_distribution roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees]. (Bamp)
    lsys_distribution branch_angle = { 1.0, 172.0, 45.0, 20.0 };

    // Child branch diameter ratios are typically anticorrelated.
    // The Burke 1992 parameterization describes the two child ratios
    // as d1 = r1 + a*r2 and d2 = r2 + a*r1 where a is a constant
    // determined by the correlation, and r1 and r2 are drawn from
    // a common distribution (`d_child_r` below).
    double diam_child_a = -0.2087;
    lsys_distribution diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

    // Bifurcation and termination probabilities per unit-length below are given by
    // P = k1 * exp(k2*diameter), described by the paremeters k1 and k2 [1/µm].

    // Termination probability parameters [1/µm]. (k1trm, k2trm)
    double pterm_k1 = 2.62e-2;
    double pterm_k2 = -2.955;

    // Bifurcation probability is given by minimum of two probabilities, `pbranch_ov' and
    // `pbranch_nov' below.
    double pbranch_ov_k1 = 2.58e-5;
    double pbranch_ov_k2 = 2.219;
    double pbranch_nov_k1 = 2.34e-3;
    double pbranch_nov_k2 = 0.194;

    // Absolute maximum dendritic extent [µm]. (Forces termination of algorithm)
    double max_extent = 2000; 
};

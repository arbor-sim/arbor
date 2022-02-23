#pragma once

#include <random>

#include <arbor/morph/segment_tree.hpp>

struct lsys_param;

using lsys_generator = std::minstd_rand;

struct lsys_distribution_param;
arb::segment_tree generate_morphology(const lsys_distribution_param& soma, std::vector<lsys_param> Ps, lsys_generator& g);

// The distribution parameters used in the specification of the L-system parameters.
// The distribution can be a constant, uniform over an interval, or truncated normal.
// Note that mixture distributions of the above should be, but isn't yet,
// implemented.

struct lsys_distribution_param {
    using result_type = double;
    enum lsys_kind {
        constant, uniform, normal
    };

    lsys_kind kind;
    result_type lb; // lower bound
    result_type ub; // upper bound (used for only non-constant)
    result_type mu; // mean (used only for truncated Gaussian)
    result_type sigma; // s.d. (used only for truncated Gaussian)

    bool operator==(const lsys_distribution_param &p) const {
        if (kind!=p.kind) return false;
        switch (kind) {
        case constant:
            return lb == p.lb;
        case uniform:
            return lb == p.lb && ub == p.ub;
        case normal:
            return lb == p.lb && ub == p.ub && mu == p.mu && sigma == p.sigma;
        }
        return false;
    }

    bool operator!=(const lsys_distribution_param &p) const {
        return !(*this==p);
    }

    // zero or one parameter => constant
    lsys_distribution_param():
        kind(constant), lb(0.0) {}

    lsys_distribution_param(result_type x):
        kind(constant), lb(x) {}

    // two paramter => uniform
    lsys_distribution_param(result_type lb, result_type ub):
        kind(uniform), lb(lb), ub(ub) {}

    // four paramter => truncated normal
    lsys_distribution_param(result_type lb, result_type ub, result_type mu, result_type sigma):
        kind(normal), lb(lb), ub(ub), mu(mu), sigma(sigma) {}
};


// Parameters as referenced in Ascoli 2001 are given in parentheses below.
// Defaults below are chosen to be safe, boring and not biological.
struct lsys_param {
    // Soma diameter [µm].
    lsys_distribution_param diam_soma = { 20 };

    // Number of dendritic trees (rounded to nearest int). (Ntree)
    lsys_distribution_param n_tree = { 1 };

    // Initial dendrite diameter [µm]. (Dstem)
    lsys_distribution_param diam_initial = { 2 };

    // Dendrite step length [µm]. (ΔL)
    lsys_distribution_param length_step = { 25 };

    // Dendrites grow along x-axis after application of (instrinsic) rotations;
    // roll (about x-axis) is applied first, followed by pitch (about y-axis).

    // Initial roll (intrinsic rotation about x-axis) [degrees]. (Taz)
    lsys_distribution_param roll_initial = { -45, 45 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees]. (Tel)
    lsys_distribution_param pitch_initial = { -45, 45 };

    // Tortuousness: roll within section over ΔL [degrees]. (Eaz)
    lsys_distribution_param roll_section = { -45, 45 };

    // Tortuousness: pitch within section over ΔL [degrees]. (Eel)
    lsys_distribution_param pitch_section = { -45, 45 };

    // Taper rate: diameter decrease per unit length. (TPRB)
    lsys_distribution_param taper = { 0 };

    // Branching torsion: roll at branch point [degrees]. (Btor)
    lsys_distribution_param roll_at_branch = { 0 };

    // Branch angle between siblings [degrees]. (Bamp)
    lsys_distribution_param branch_angle = { 45 };

    // Child branch diameter ratios are typically anticorrelated.
    // The Burke 1992 parameterization describes the two child ratios
    // as d1 = r1 + a*r2 and d2 = r2 + a*r1 where a is a constant
    // determined by the correlation, and r1 and r2 are drawn from
    // a common distribution (`d_child_r` below).
    double diam_child_a = 0;
    lsys_distribution_param diam_child_r = { 0.5 };

    // Bifurcation and termination probabilities per unit-length below are given by
    // P = k1 * exp(k2*diameter), described by the paremeters k1 and k2 [1/µm].

    // Termination probability parameters [1/µm]. (k1trm, k2trm)
    double pterm_k1 = 0.05;
    double pterm_k2 = -2;

    // Bifurcation probability is given by minimum of two probabilities, `pbranch_ov' and
    // `pbranch_nov' below.
    double pbranch_ov_k1 = 0.01;
    double pbranch_ov_k2 = 0;
    double pbranch_nov_k1 = 0.01;
    double pbranch_nov_k2 = 0;

    // Absolute maximum dendritic extent [µm]. (Forces termination of algorithm)
    double max_extent = 2000;

    // Absolute maximum number of unbranched sections. (Forces termination of algorithm)
    unsigned max_sections = 10000;

    size_t tag = 0;
};


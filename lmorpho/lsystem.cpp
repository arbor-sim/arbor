#include <cmath>
#include <iostream>
#include <random>
#include <stack>
#include <vector>
#include <map>

#include <arbor/math.hpp>

#include "lsystem.hpp"

using namespace arb::math;

// L-system implementation.

// Random distribution used in the specification of L-system parameters.
// It can be a constant, uniform over an interval, or truncated normal.
// Note that mixture distributions of the above should be, but isn't yet,
// implemented.

struct lsys_distribution {
    using result_type = double;
    using param_type = lsys_distribution_param;

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
        return p_.kind==param_type::constant? min(): p_.ub;
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
        case param_type::constant:
            return p_.lb;
        case param_type::uniform:
            return uniform_dist(g);
        case param_type::normal:
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
        case param_type::uniform:
            uniform_dist = std::uniform_real_distribution<double>(p_.lb, p_.ub);
            break;
        case param_type::normal:
            normal_dist = std::normal_distribution<result_type>(p_.mu, p_.sigma);
            break;
        default: ;
        }
    }
};

class burke_correlated_r {
    double a_;
    lsys_distribution r_;

public:
    struct result_type { double rho1, rho2; };

    burke_correlated_r(double a, lsys_distribution_param r): a_(a), r_(r) {}

    template <typename Gen>
    result_type operator()(Gen& g) const {
        double r1 = r_(g);
        double r2 = r_(g);
        return { r1+a_*r2, r2+a_*r1 };
    }
};

class burke_exp_test {
    double k1_;
    double k2_;

public:
    using result_type = int;

    burke_exp_test(double k1, double k2): k1_(k1), k2_(k2) {}

    template <typename Gen>
    result_type operator()(double delta_l, double radius, Gen& g) const {
        std::bernoulli_distribution B(delta_l*k1_*std::exp(k2_*radius*2.0));
        return B(g);
    }
};

class burke_exp2_test {
    double ak1_;
    double ak2_;
    double bk1_;
    double bk2_;

public:
    using result_type = int;

    burke_exp2_test(double ak1, double ak2, double bk1, double bk2):
        ak1_(ak1), ak2_(ak2), bk1_(bk1), bk2_(bk2) {}

    template <typename Gen>
    result_type operator()(double delta_l, double radius, Gen& g) const {
        double diam = 2.*radius;
        std::bernoulli_distribution B(
            delta_l*std::min(ak1_*std::exp(ak2_*diam), bk1_*std::exp(bk2_*diam)));

        return B(g);
    }
};

// L-system parameter set with instantiated distributions.
struct lsys_sampler {
    // Create distributions once from supplied parameter set.
    explicit lsys_sampler(const lsys_param& P):
        diam_soma(P.diam_soma),
        n_tree(P.n_tree),
        diam_initial(P.diam_initial),
        length_step(P.length_step),
        roll_initial(P.roll_initial),
        pitch_initial(P.pitch_initial),
        roll_section(P.roll_section),
        pitch_section(P.pitch_section),
        taper(P.taper),
        roll_at_branch(P.roll_at_branch),
        branch_angle(P.branch_angle),
        diam_child(P.diam_child_a, P.diam_child_r),
        term_test(P.pterm_k1, P.pterm_k2),
        branch_test(P.pbranch_ov_k1, P.pbranch_ov_k2,
            P.pbranch_nov_k1, P.pbranch_nov_k2),
        max_extent(P.max_extent)
    {}

    lsys_distribution diam_soma;
    lsys_distribution n_tree;
    lsys_distribution diam_initial;
    lsys_distribution length_step;
    lsys_distribution roll_initial;
    lsys_distribution pitch_initial;
    lsys_distribution roll_section;
    lsys_distribution pitch_section;
    lsys_distribution taper;
    lsys_distribution roll_at_branch;
    lsys_distribution branch_angle;
    burke_correlated_r diam_child;
    burke_exp_test term_test;
    burke_exp2_test branch_test;
    double max_extent;
};

struct section_point {
    double x;
    double y;
    double z;
    double r;
    friend std::ostream& operator<<(std::ostream& os, const section_point& obj);
};

std::ostream& operator<<(std::ostream& os, const section_point& obj) {
    os << obj.x << " " << obj.y << " " << obj.z << " " << obj.r << '\n';
    return os;
}

struct section_tip {
    section_point point = {0., 0., 0., 0.};
    quaternion rotation = {1., 0., 0., 0.};
    double somatic_distance = 0.;
};


struct section_geometry {
    unsigned id = 0; // ids should be contigously numbered from 1 in the morphology.
    unsigned parent_id = 0;
    bool terminal = false;
    std::vector<section_point> points;
    double length = 0; // µm

    section_geometry() = default;
    section_geometry(unsigned id, unsigned parent_id, bool terminal, std::vector<section_point> points, double length) :
        id(id), parent_id(parent_id), terminal(terminal), points(std::move(points)), length(length)
    {}

    friend std::ostream& operator<<(std::ostream& os, const section_geometry& obj);
};

std::ostream& operator<<(std::ostream& os, const section_geometry& obj) {
    os << "id: " << obj.id << '\n';
    os << "parent_id: " << obj.parent_id << '\n';
    os << "terminal: " << obj.terminal << '\n';
    os << "length: " << obj.length << '\n';
    os << "points: " << '\n';
    for (auto& p : obj.points) {
        os << p << '\n';
    }
    return os;
}

struct grow_result {
    std::vector<section_point> points;
    std::vector<section_tip> children;
    double length = 0.;
};

constexpr double deg_to_rad = 3.1415926535897932384626433832795/180.0;

template <typename Gen>
grow_result grow(section_tip tip, const lsys_sampler& S, Gen &g) {
    constexpr quaternion xaxis = {0, 1, 0, 0};
    static std::uniform_real_distribution<double> U;

    grow_result result;
    std::vector<section_point>& points = result.points;

    points.push_back(tip.point);
    for (;;) {
        quaternion step = xaxis^tip.rotation;
        double dl = S.length_step(g);
        tip.point.x += dl*step.x;
        tip.point.y += dl*step.y;
        tip.point.z += dl*step.z;
        tip.point.r += dl*0.5*S.taper(g);
        tip.somatic_distance += dl;
        result.length += dl;

        if (tip.point.r<0) tip.point.r = 0;

        double phi = S.roll_section(g);
        double theta = S.pitch_section(g);
        tip.rotation *= rotation_x(deg_to_rad*phi);
        tip.rotation *= rotation_y(deg_to_rad*theta);

        points.push_back(tip.point);

        if (tip.point.r==0 || tip.somatic_distance>=S.max_extent) {
            return result;
        }

        if (S.branch_test(dl, tip.point.r, g)) {
            auto r = S.diam_child(g);
            // ignore branch if we get non-positive radii
            if (r.rho1>0 && r.rho2>0) {
                double branch_phi = S.roll_at_branch(g);
                double branch_angle = S.branch_angle(g);
                double branch_theta1 = U(g)*branch_angle;
                double branch_theta2 = branch_theta1-branch_angle;

                tip.rotation *= rotation_x(deg_to_rad*branch_phi);

                section_tip t1 = tip;
                t1.point.r = t1.point.r * r.rho1;
                t1.rotation *= rotation_y(deg_to_rad*branch_theta1);

                section_tip t2 = tip;
                t2.point.r = t2.point.r * r.rho2;
                t2.rotation *= rotation_y(deg_to_rad*branch_theta2);

                result.children = {t1, t2};
                return result;
            }
        }

        if (S.term_test(dl, tip.point.r, g)) {
            return result;
        }
    }
}

std::vector<section_geometry> generate_sections(double soma_radius, lsys_param P, lsys_generator &g) {
    constexpr quaternion xaxis = {0, 1, 0, 0};

    lsys_sampler S(P);

    struct section_start {
        section_tip tip;
        unsigned parent_id;
    };
    std::stack<section_start> starts;
    unsigned next_id = 1u;

    int n = (int)std::round(S.n_tree(g));
    for (int i=0; i<n; ++i) {
        double phi = S.roll_initial(g);
        double theta = S.pitch_initial(g);
        double radius = 0.5*S.diam_initial(g);

        section_tip tip;
        tip.rotation = rotation_x(deg_to_rad*phi)*rotation_y(deg_to_rad*theta);
        tip.somatic_distance = 0.0;

        auto p = (soma_radius*xaxis)^tip.rotation;
        tip.point = {p.x, p.y, p.z, radius};

        starts.push({tip, 0u});
    }

    std::vector<section_geometry> sections;

    while (!starts.empty() && sections.size() < P.max_sections) {
        auto start = starts.top();
        starts.pop();

        auto branch = grow(start.tip, S, g);
        section_geometry section{next_id++, start.parent_id, branch.children.empty(), std::move(branch.points), branch.length};

        for (auto child: branch.children) {
            starts.push({child, section.id});
        }

        sections.push_back(std::move(section));
    }

    return sections;
}

arb::segment_tree generate_morphology(const lsys_distribution_param& soma, std::vector<lsys_param> Ps, lsys_generator &g) {

    double const soma_diameter = lsys_distribution(soma)(g);
    double const soma_radius = 0.5*soma_diameter;

    arb::segment_tree morph;
    morph.append(
                arb::mnpos, arb::mpoint{-soma_diameter, 0, 0, soma_diameter},
                arb::mpoint{soma_diameter, 0, 0, soma_diameter}, 1);

    for(auto P : Ps) {
        auto sections = generate_sections(soma_radius, P, g);

        // map from internal representation to Arbor ids
        std::map<size_t, arb::msize_t> parent_end_id;

        // soma
        parent_end_id[0] = 0;

        // dendrites:
        for (auto& sec: sections) {

            // the first section must have the soma as parent
            size_t parent = parent_end_id.at(sec.parent_id);

            const auto& points = sec.points;
            if (points.size() < 2) {
                throw std::runtime_error("segment has only 1 point");
            }

            // Include first point only for dendrites segments attached to soma.
            if (sec.parent_id==0) {
                parent = morph.append(parent,
                                      arb::mpoint{points[0].x, points[0].y, points[0].z, points[0].r},
                                      arb::mpoint{points[1].x, points[1].y, points[1].z, points[1].r}, P.tag);
            }

            // Remaining points.
            for (unsigned i = 1; i < points.size(); ++i) {
                const auto& p = points[i];
                parent = morph.append(parent, arb::mpoint{p.x, p.y, p.z, p.r}, P.tag);
            }

            parent_end_id[sec.id] = parent;
        }
    }

    return morph;
}


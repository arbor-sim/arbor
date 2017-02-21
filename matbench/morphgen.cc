#include <cmath>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

// 'generic' segment based generation for use with SWC or internal morphology targets.

struct segment_point {
    double x, y, z, r;
};

// move quaternion/versor stuff into separate header.

struct quaternion {
    double w, x, y, z;

    quaternion(): w(0), x(0), y(0), z(0) {}

    // scalar
    quaternion(double w): w(w), x(0), y(0), z(0) {}

    // vector (pure imaginary)
    quaternion(double x, double y, double z): w(0), x(x), y(y), z(z) {}

    // all 4-components
    quaternion(double w, double x, double y, double z): w(w), x(x), y(y), z(z) {}

    quaternion conjugate() const {
        return {w, -x, -y, -z};
    }

    quaternion operator*(quaternion q) const {
        return {w*q.w-x*q.x-y*q.y-z*q.z,
                w*q.x+x*q.w+y*q.z-z*q.y,
                w*q.y-x*q.z+y*q.w+z*q.x,
                w*q.z+x*q.y-y*q.x+z*q.w};
    }

    quaternion operator+(quaternion q) const {
        return {w+q.w, x+q.x, y+q.y, z+q.z};
    }

    quaternion operator-(quaternion q) const {
        return {w-q.w, x-q.x, y-q.y, z-q.z};
    }

    double sqnorm() const {
        return w*w+x*x+y*y+z*z;
    }

    double norm() const {
        return std::sqrt(sqnorm());
    }

    // add more as required...
};

// rotations about internal axes as quaternions

quaternion rotate_x(double phi) {
    return {std::cos(phi/2), std::sin(phi/2), 0, 0};
}

quaternion rotate_y(double theta) {
    return {std::cos(theta/2), 0, std::sin(theta/2), 0};
}

quaternion rotate_z(double psi) {
    return {std::cos(psi/2), 0, 0, std::sin(psi/2)};
}

// replace following with the more generic random distributions required,
// that can represent mixture models and truncated Gaussian.

enum class lparam_kind {
    constant, uniform, normal
};

struct lparam_distribution {
    using result_type = double;
    struct param_type {
        lparam_kind kind;
        result_type lb; // lower bound
        result_type ub; // upper bound (used for only non-constant)
        result_type mu; // mean (used only for truncated Gaussian)
        result_type sigma; // s.d. (used only for truncated Gaussian)

        bool operator==(const param_type &p) const {
            if (kind!=p.kind) return false;
            switch (kind) {
            case lparam_kind::constant:
                return lb == p.lb;
            case lparam_kind::uniform:
                return lb == p.lb && ub == p.ub;
            case lparam_kind::normal:
                return lb == p.lb && ub == p.ub && mu == p.mu && sigma == p.sigma;
            }
            return false;
        }

        bool operator!=(const param_type &p) const {
            return !(*this==p);
        }

        // zero or one parameter => constant
        param_type():
            kind(lparam_kind::constant), lb(0.0) {}

        param_type(result_type x):
            kind(lparam_kind::constant), lb(x) {}

        // two paramter => uniform
        param_type(result_type lb, result_type ub):
            kind(lparam_kind::uniform), lb(lb), ub(ub) {}

        // four paramter => truncated normal
        param_type(result_type lb, result_type ub, result_type mu, result_type sigma):
            kind(lparam_kind::normal), lb(lb), ub(ub), mu(mu), sigma(sigma) {}
    };

    lparam_distribution(const param_type& p): p_(p) { init(); }

    lparam_distribution(): p_() { init(); }
    lparam_distribution(result_type x): p_(x) { init(); }
    lparam_distribution(result_type lb, result_type ub): p_(lb, ub) { init(); }
    lparam_distribution(result_type lb, result_type ub, result_type mu, result_type sigma):
        p_(lb, ub, mu, sigma)
    {
        init();
    }

    param_type param() const { return p_; }
    void param(const param_type& pbis) { p_=pbis; }

    result_type min() const { return p_.lb; }
    result_type max() const {
        return p_.kind==lparam_kind::constant? min(): p_.ub;
    }

    bool operator==(const lparam_distribution& y) const {
        return p_==y.p_;
    }

    bool operator!=(const lparam_distribution& y) const {
        return p_!=y.p_;
    }

    // omitting out ostream, istream methods
    // (...)

    template <typename Gen>
    result_type operator()(Gen& g, const param_type &p) const {
        lparam_distribution dist(p);
        return dist(g);
    }

    template <typename Gen>
    result_type operator()(Gen& g) const {
        switch (p_.kind) {
        case lparam_kind::constant:
            return p_.lb;
        case lparam_kind::uniform:
            return uniform_dist(g);
        case lparam_kind::normal:
            // replace with a robust truncated normal generator!
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
        case lparam_kind::uniform:
            uniform_dist = std::uniform_real_distribution<double>(p_.lb, p_.ub);
            break;
        case lparam_kind::normal:
            normal_dist = std::normal_distribution<result_type>(p_.mu, p_.sigma);
            break;
        }
    }
};

// Fields below refer to equivalent L-Neuron parameters in parantheses.
// Note that for consistency with literature, parameters describe diameters,
// while the internal representation uses radii.
struct lsystem_param {
    // Soma diameter [µm].
    lparam_distribution diam_soma = { 47.0, 65.5, 57.6, 5.0 };

    // Number of dendritic trees (rounded to nearest int). (Ntree)
    lparam_distribution n_tree = { 1.0 };

    // Initial dendrite diameter [µm]. (Dstem)
    lparam_distribution diam_initial = { 5.0, 8.0 };

    // Dendrite step length [µm]. (ΔL)
    lparam_distribution length_step = { 25 };

    // Dendrites grow along x-axis after application of (instrinsic) rotations;
    // roll (about x-axis) is applied first, followed by pitch (about y-axis).

    // Initial roll (intrinsic rotation about x-axis) [degrees]. (Taz)
    lparam_distribution roll_initial = { -180.0, 180.0 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees]. (Tel)
    lparam_distribution pitch_initial = { 0.0, 180.0 };

    // Tortuousness: roll within segment over ΔL [degrees]. (Eaz)
    lparam_distribution roll_segment = { -180.0, 180.0 };

    // Tortuousness: pitch within segment over ΔL [degrees]. (Eel)
    lparam_distribution pitch_segment = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate. (TPRB)
    lparam_distribution taper = { -0.125 };

    // Branching torsion: roll at branch point [degrees]. (Btor)
    lparam_distribution roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees]. (Bamp)
    lparam_distribution branch_angle = { 1.0, 172.0, 45.0, 20.0 };

    // Child branch diameter ratios are typically anticorrelated.
    // The Burke 1992 parameterization describes the two child ratios
    // as d1 = r1 + a*r2 and d2 = r2 + a*r1 where a is a constant
    // determined by the correlation, and r1 and r2 are drawn from
    // a common distribution (`d_child_r` below).
    double diam_child_a = -0.2087;
    lparam_distribution diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

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

// L-system implementation

// common (flat) morphological description for SWC and cell morphology
// creation.

struct segment_tip {
    segment_point p = {0., 0., 0., 0.};
    quaternion rotation = {1., 0., 0., 0.};
    double somatic_distance = 0.;
};

struct grow_result {
    std::vector<segment_point> points;
    std::vector<segment_tip> children;
};

template <typename Gen>
grow_result grow(segment_tip tip, const lsystem_param& P, Gen &g) {
    static std::uniform_real_distribution<double> U;
    std::vector<segment_point> points;

    points.push_back(tip.p);
    for (;;) {
        quaternion step = tip.rotation*quaternion{0, 1, 0, 0}*tip.rotation.conjugate();
        double dl = P.length_step(g);
        tip.p.x += dl*step.x;
        tip.p.y += dl*step.y;
        tip.p.z += dl*step.z;
        tip.p.r += dl*0.5*P.taper(g);
        tip.somatic_distance += dl;

        double phi = P.roll_segment(g);
        double theta = P.pitch_segment(g);
        tip.rotation = rotate_y(theta)*rotate_x(phi)*tip.rotation;

        points.push_back(tip.p);

        double pbranch = std::min(
            P.pbranch_ov_k1*exp(P.pbranch_ov_k2*2.*tip.p.r),
            P.pbranch_nov_k1*exp(P.pbranch_nov_k2*2.*tip.p.r));

        if (U(g)<pbranch) {
            double branch_phi = P.roll_at_branch(g);
            double branch_angle = P.branch_angle(g);
            double branch_theta1 = U(g)*branch_angle;
            double branch_theta2 = branch_theta1-branch_angle;
            double r1 = P.diam_child_r(g);
            double r2 = P.diam_child_r(g);
            double a = P.diam_child_a;

            tip.rotation = rotate_x(branch_phi)*tip.rotation;

            segment_tip t1 = tip;
            t1.p.r = t1.p.r * (r1 + a*r2);
            t1.rotation = rotate_y(branch_theta1)*t1.rotation;

            segment_tip t2 = tip;
            t2.p.r = t2.p.r * (r2 + a*r1);
            t2.rotation = rotate_y(branch_theta2)*t2.rotation;

            return {points, {t1, t2}};
        }

        double pterm = P.pterm_k1*exp(P.pterm_k2*2.0*tip.p.r);
        if (U(g)<pterm || tip.somatic_distance>=P.max_extent) {
            return {points, {}};
        }
    }
}

struct segment_geometry {
    unsigned id;
    unsigned parent_id;
    std::vector<segment_point> points;
};

struct morphology {
    segment_point soma; // origin + spherical radius
    std::vector<segment_geometry> segments;
};

template <typename Gen>
morphology generate_morphology(const lsystem_param& P, Gen &g) {
    morphology morph;

    double soma_radius = 0.5*P.diam_soma(g);
    morph.soma = {0, 0, 0, soma_radius};

    struct segment_start {
        segment_tip tip;
        unsigned parent_id;
    };
    std::stack<segment_start> starts;
    unsigned next_id = 1u;

    int n = (int)std::round(P.n_tree(g));
    for (int i=0; i<n; ++i) {
        double phi = P.roll_initial(g);
        double theta = P.pitch_initial(g);
        double radius = 0.5*P.diam_initial(g);

        segment_tip tip;
        tip.rotation = rotate_y(theta)*rotate_x(phi);
        tip.somatic_distance = 0.0;

        auto p = tip.rotation*quaternion{0., soma_radius, 0., 0.}*tip.rotation.conjugate();
        tip.p = {p.x, p.y, p.z, radius};

        starts.push({tip, 0u});
    }

    while (!starts.empty()) {
        auto start = starts.top();
        starts.pop();

        auto branch = grow(start.tip, P, g);
        segment_geometry segment = {next_id++, start.parent_id, std::move(branch.points)};

        for (auto child: branch.children) {
            starts.push({child, segment.parent_id});
        }
        morph.segments.push_back(std::move(segment));
    }

    return morph;
}

// serialize morphology as SWC (soon)
std::ostream& operator<<(std::ostream& O, const morphology& m) {
    O << "soma radius: " << m.soma.r << "\n";
    O << "#segments: " << m.segments.size() << "\n";
    segment_point p = m.segments.back().points.back();
    O << "last segment tip: " << p.x << "," << p.y << "," << p.z << "; radius: " << p.r << "\n";
    return O;
}

int main() {
    lsystem_param P; // default
    std::minstd_rand g;

    for (int i=0; i<5; ++i) {
        auto morph = generate_morphology(P, g);
        std::cout << "#" << i << ":\n" << morph << "\n";
    }
}



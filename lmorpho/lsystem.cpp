#include <cmath>
#include <iostream>
#include <random>
#include <stack>
#include <vector>

#include "quaternion.h"
#include "lsystem.h"
#include "morphology.h"

// L-system implementation.

struct segment_tip {
    segment_point p = {0., 0., 0., 0.};
    quaternion rotation = {1., 0., 0., 0.};
    double somatic_distance = 0.;
};

struct grow_result {
    std::vector<segment_point> points;
    std::vector<segment_tip> children;
};

constexpr double deg_to_rad = 3.1415926535897932384626433832795l/180.0;

template <typename Gen>
grow_result grow(segment_tip tip, const lsys_param& P, Gen &g) {
    constexpr quaternion xaxis = {0, 1, 0, 0};
    static std::uniform_real_distribution<double> U;
    std::vector<segment_point> points;

    points.push_back(tip.p);
    for (;;) {
        quaternion step = xaxis^tip.rotation;
        double dl = P.length_step(g);
        tip.p.x += dl*step.x;
        tip.p.y += dl*step.y;
        tip.p.z += dl*step.z;
        tip.p.r += dl*0.5*P.taper(g);
        tip.somatic_distance += dl;

        if (tip.p.r<0) tip.p.r = 0;

        double phi = P.roll_segment(g);
        double theta = P.pitch_segment(g);
        tip.rotation *= rotation_x(deg_to_rad*theta);
        tip.rotation *= rotation_y(deg_to_rad*phi);

        points.push_back(tip.p);

        double pbranch = dl*std::min(
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

            tip.rotation *= rotation_x(deg_to_rad*branch_phi);

            segment_tip t1 = tip;
            t1.p.r = t1.p.r * (r1 + a*r2);
            t1.rotation *= rotation_y(deg_to_rad*branch_theta1);

            segment_tip t2 = tip;
            t2.p.r = t2.p.r * (r2 + a*r1);
            t2.rotation *= rotation_y(deg_to_rad*branch_theta2);

            return {points, {t1, t2}};
        }

        double pterm = dl*P.pterm_k1*exp(P.pterm_k2*2.0*tip.p.r);
        if (tip.p.r==0 || U(g)<pterm || tip.somatic_distance>=P.max_extent) {
            return {points, {}};
        }
    }
}

morphology generate_morphology(const lsys_param& P, lsys_generator &g) {
    constexpr quaternion xaxis = {0, 1, 0, 0};
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
        tip.rotation = rotation_x(deg_to_rad*phi)*rotation_y(deg_to_rad*theta);
        tip.somatic_distance = 0.0;

        auto p = (soma_radius*xaxis)^tip.rotation;
        tip.p = {p.x, p.y, p.z, radius};

        starts.push({tip, 0u});
        std::cerr << "initial tip: " << p.x << "," << p.y << "," << p.z << "; r=" << radius << "\n";
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


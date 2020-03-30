#include <algorithm>
#include <utility>

#include <arbor/cable_cell.hpp>
#include <arbor/math.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/util/optional.hpp>

#include "fvm_layout.hpp"
#include "util/span.hpp"

#include "common.hpp"
#include "common_morphologies.hpp"
#include "../common_cells.hpp"

using namespace arb;
using util::make_span;

TEST(cv_layout, empty) {
    using namespace common_morphology;

    cable_cell empty_cell{m_empty};
    fvm_cv_discretization D = fvm_cv_discretize(empty_cell, neuron_parameter_defaults);

    EXPECT_TRUE(D.empty());
    EXPECT_EQ(0u, D.size());
    EXPECT_EQ(1u, D.n_cell());

    EXPECT_EQ(0u, D.face_conductance.size());
    EXPECT_EQ(0u, D.cv_area.size());
    EXPECT_EQ(0u, D.cv_capacitance.size());
    EXPECT_EQ(0u, D.init_membrane_potential.size());
    EXPECT_EQ(0u, D.temperature_K.size());
    EXPECT_EQ(0u, D.diam_um.size());
}

TEST(cv_layout, trivial) {
    using namespace common_morphology;

    auto params = neuron_parameter_defaults;
    params.discretization = cv_policy_explicit(ls::nil());

    // For each cell, check size, confirm area is morphological area from
    // embedding, and that membrane-properties are equal to defaults.

    std::vector<cable_cell> cells;
    unsigned n_cv = 0;
    for (auto& p: test_morphologies) {
        cells.emplace_back(p.second);
        n_cv += !p.second.empty(); // one cv per non-empty cell
    }

    auto n_cells = cells.size();
    fvm_cv_discretization D = fvm_cv_discretize(cells, params);

    EXPECT_EQ(n_cv, D.size());
    for (unsigned i = 0; i<n_cells; ++i) {
        auto cv_indices = util::make_span(D.geometry.cell_cv_interval(i));
        if (test_morphologies[i].second.empty()) {
            ASSERT_TRUE(cv_indices.empty());
            continue;
        }
        else {
            ASSERT_EQ(1u, cv_indices.size());
        }

        auto cv = cv_indices.front();

        EXPECT_DOUBLE_EQ(params.temperature_K.value(), D.temperature_K[cv]);
        EXPECT_DOUBLE_EQ(params.init_membrane_potential.value(), D.init_membrane_potential[cv]);

        double total_area = 0;
        unsigned n_branch = cells[i].morphology().num_branches();
        const auto& embedding = cells[i].embedding();
        for (unsigned b = 0; b<n_branch; ++b) {
            total_area += embedding.integrate_area(mcable{b, 0., 1.});
        }

        EXPECT_DOUBLE_EQ(total_area, D.cv_area[cv]);
        EXPECT_DOUBLE_EQ(total_area*params.membrane_capacitance.value(), D.cv_capacitance[cv]);
    }
}

TEST(cv_layout, cable) {
    auto morph = common_morphology::m_reg_b1; // one branch, cable constant radius.

    auto params = neuron_parameter_defaults;
    params.init_membrane_potential = 0;

    cable_cell c(morph);
    c.paint(reg::cable(0, 0.0, 0.2), init_membrane_potential{10});
    c.paint(reg::cable(0, 0.2, 0.7), init_membrane_potential{20});
    c.paint(reg::cable(0, 0.7, 1.0), init_membrane_potential{30});

    params.discretization = cv_policy_explicit(ls::nil());
    fvm_cv_discretization D = fvm_cv_discretize(c, params);

    ASSERT_EQ(1u, D.size());
    EXPECT_DOUBLE_EQ(0.2*10+0.5*20+0.3*30, D.init_membrane_potential[0]);

    params.discretization = cv_policy_explicit(ls::location(0, 0.3));
    D = fvm_cv_discretize(c, params);

    ASSERT_EQ(2u, D.size());
    EXPECT_DOUBLE_EQ((0.2*10+0.1*20)/0.3, D.init_membrane_potential[0]);
    EXPECT_DOUBLE_EQ((0.4*20+0.3*30)/0.7, D.init_membrane_potential[1]);
}

TEST(cv_layout, cable_conductance) {
    auto morph = common_morphology::m_reg_b1; // one branch, cable constant radius.
    const double rho = 5.; // [Ω·cm]

    auto params = neuron_parameter_defaults;
    params.axial_resistivity = rho;

    cable_cell c(morph);
    double radius = c.embedding().radius(mlocation{0, 0.5});
    double length = c.embedding().branch_length(0);

    params.discretization = cv_policy_explicit(ls::location(0, 0.3));
    fvm_cv_discretization D = fvm_cv_discretize(c, params);

    ASSERT_EQ(2u, D.size());

    // Face conductance should be conductance between (relative) points 0.15 and 0.65.
    double xa = math::pi<double>*radius*radius; // [µm^2]
    double l = (0.65-0.15)*length; // [µm]
    double sigma = 100 * xa/(l*rho); // [µS]

    EXPECT_DOUBLE_EQ(0., D.face_conductance[0]);
    EXPECT_DOUBLE_EQ(sigma, D.face_conductance[1]);
}

TEST(cv_layout, zero_size_cv) {
    // Six branches; branches 0, 1 and 2 meet at (0, 1); branches
    // 2, 3, 4, and 5 meet at (2, 1). Terminal branches are 1, 3, 4, and 5.
    auto morph = common_morphology::m_reg_b6;
    cable_cell cell(morph);

    auto params = neuron_parameter_defaults;
    const double rho = 5.; // [Ω·cm]
    const double pi = math::pi<double>;
    params.axial_resistivity = rho;

    // With one CV per branch, expect reference points for face conductance
    // to be at (0, 0.5); (0, 1); (1, 0.5); (2, 1); (3, 0.5); (4, 0.5); (5, 0.5).
    // The first CV should be all of branch 0; the second CV should be the
    // zero-size CV covering the branch point (0, 1).
    params.discretization = cv_policy_fixed_per_branch(1);
    fvm_cv_discretization D = fvm_cv_discretize(cell, params);

    unsigned cv_a = 0, cv_x = 1;
    ASSERT_TRUE(util::equal(mcable_list{mcable{0, 0, 1}}, D.geometry.cables(cv_a)));
    ASSERT_TRUE(util::equal(mcable_list{mcable{0, 1, 1}, mcable{1, 0, 0}, mcable{2, 0, 0}},
                    D.geometry.cables(cv_x)));

    // Find the two CV children of CV x.
    unsigned cv_b = -1, cv_c = -1;
    for (unsigned i=2; i<D.size(); ++i) {
        if ((unsigned)D.geometry.cv_parent[i]==cv_x) {
            if (cv_b==(unsigned)-1) cv_b = i;
            else if (cv_c==(unsigned)-1) cv_c = i;
            else FAIL();
        }
    }

    ASSERT_EQ(1u, D.geometry.cables(cv_b).size());
    ASSERT_EQ(1u, D.geometry.cables(cv_c).size());
    if (D.geometry.cables(cv_b).front().branch>D.geometry.cables(cv_c).front().branch) {
        std::swap(cv_b, cv_c);
    }

    ASSERT_TRUE(util::equal(mcable_list{mcable{1, 0, 1}}, D.geometry.cables(cv_b)));
    ASSERT_TRUE(util::equal(mcable_list{mcable{2, 0, 1}}, D.geometry.cables(cv_c)));

    // All non-conductance values for zero-size cv_x should be zero.
    EXPECT_EQ(0., D.cv_area[cv_x]);
    EXPECT_EQ(0., D.cv_capacitance[cv_x]);
    EXPECT_EQ(0., D.init_membrane_potential[cv_x]);
    EXPECT_EQ(0., D.temperature_K[cv_x]);
    EXPECT_EQ(0., D.diam_um[cv_x]);

    // Face conductance for zero-size cv_x:
    double l_x = cell.embedding().branch_length(0);
    double r_x = cell.embedding().radius(mlocation{0, 0.5});
    double sigma_x = 100 * pi * r_x * r_x / (l_x/2 * rho); // [µS]
    EXPECT_DOUBLE_EQ(sigma_x, D.face_conductance[cv_x]);

    // Face conductance for child CV cv_b:
    double l_b = cell.embedding().branch_length(1);
    double r_b = cell.embedding().radius(mlocation{1, 0.5});
    double sigma_b = 100 * pi * r_b * r_b / (l_b/2 * rho); // [µS]
    EXPECT_DOUBLE_EQ(sigma_b, D.face_conductance[cv_b]);

    // Face conductance for child CV cv_c:
    // (Distal reference point is at end of branch, so l_c not l_c/2 below.)
    double l_c = cell.embedding().branch_length(1);
    double r_c = cell.embedding().radius(mlocation{1, 0.5});
    double sigma_c = 100 * pi * r_c * r_c / (l_c * rho); // [µS]
    EXPECT_DOUBLE_EQ(sigma_c, D.face_conductance[cv_c]);
}

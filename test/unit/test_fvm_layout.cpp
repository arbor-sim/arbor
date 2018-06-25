#include <vector>

#include <arbor/util/optional.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/mc_cell.hpp>

#include "fvm_layout.hpp"
#include "math.hpp"
#include "util/maputil.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

#include "common.hpp"
#include "../common_cells.hpp"

using namespace arb;
using namespace testing::string_literals;

using util::make_span;
using util::count_along;
using util::value_by_key;

namespace {
    double area(const mc_segment* s) {
        if (auto soma = s->as_soma()) {
            return math::area_sphere(soma->radius());
        }
        else if (auto cable = s->as_cable()) {
            unsigned nc = cable->num_sub_segments();
            double a = 0;
            for (unsigned i = 0; i<nc; ++i) {
                a += math::area_frustrum(cable->lengths()[i], cable->radii()[i], cable->radii()[i+1]);
            }
            return a;
        }
        else {
            return 0;
        }
    }

    double volume(const mc_segment* s) {
        if (auto soma = s->as_soma()) {
            return math::volume_sphere(soma->radius());
        }
        else if (auto cable = s->as_cable()) {
            unsigned nc = cable->num_sub_segments();
            double v = 0;
            for (unsigned i = 0; i<nc; ++i) {
                v += math::volume_frustrum(cable->lengths()[i], cable->radii()[i], cable->radii()[i+1]);
            }
            return v;
        }
        else {
            return 0;
        }
    }

    std::vector<mc_cell> two_cell_system() {
        std::vector<mc_cell> cells;

        // Cell 0: simple ball and stick (see common_cells.hpp)
        cells.push_back(make_cell_ball_and_stick());

        // Cell 1: ball and 3-stick, but with uneven dendrite
        // length and heterogeneous electrical properties:
        //
        // Bulk resistivity: 90 Ω·cm
        // capacitance:
        //    soma:       0.01  F/m² [default]
        //    segment 1:  0.017 F/m²
        //    segment 2:  0.013 F/m²
        //    segment 3:  0.018 F/m²
        //
        // Soma diameter: 14 µm
        // Some mechanisms: HH (default params)
        //
        // Segment 1 diameter: 1 µm
        // Segment 1 length:   200 µm
        //
        // Segment 2 diameter: 0.8 µm
        // Segment 2 length:   300 µm
        //
        // Segment 3 diameter: 0.7 µm
        // Segment 3 length:   180 µm
        //
        // Dendrite mechanisms: passive (default params).
        // Stimulus at end of segment 2, amplitude 0.45.
        // Stimulus at end of segment 3, amplitude -0.2.
        //
        // All dendrite segments with 4 compartments.

        mc_cell c2;
        mc_segment* s;

        s = c2.add_soma(14./2);
        s->add_mechanism("hh");

        s = c2.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200);
        s->cm = 0.017;

        s = c2.add_cable(1, section_kind::dendrite, 0.8/2, 0.8/2, 300);
        s->cm = 0.013;

        s = c2.add_cable(1, section_kind::dendrite, 0.7/2, 0.7/2, 180);
        s->cm = 0.018;

        c2.add_stimulus({2,1}, {5.,  80., 0.45});
        c2.add_stimulus({3,1}, {40., 10.,-0.2});

        for (auto& seg: c2.segments()) {
            seg->rL = 90.;
            if (seg->is_dendrite()) {
                seg->add_mechanism("pas");
                seg->set_compartments(4);
            }
        }
        cells.push_back(std::move(c2));
        return cells;
    }

    void check_two_cell_system(std::vector<mc_cell>& cells) {
        ASSERT_EQ(2u, cells[0].num_segments());
        ASSERT_EQ(cells[0].segment(1)->num_compartments(), 4u);
        ASSERT_EQ(cells[1].num_segments(), 4u);
        ASSERT_EQ(cells[1].segment(1)->num_compartments(), 4u);
        ASSERT_EQ(cells[1].segment(2)->num_compartments(), 4u);
        ASSERT_EQ(cells[1].segment(3)->num_compartments(), 4u);
    }
} // namespace

TEST(fvm_layout, topology) {
    std::vector<mc_cell> cells = two_cell_system();
    check_two_cell_system(cells);

    fvm_discretization D = fvm_discretize(cells);

    // Expected CV layouts for cells, segment indices in paren.
    //
    // Cell 0:
    //
    // CV: |  0        | 1 | 2 | 3 | 4|
    //     [soma (0)][  segment (1)   ]
    // 
    // Cell 1:
    //
    // CV: |  5        | 6 | 7 | 8 |  9  | 10 | 11 | 12 | 13|
    //     [soma (2)][  segment (3)   ][  segment (4)       ]
    //                                 [  segment (5)       ]
    //                                   | 14 | 15 | 16 | 17|

    EXPECT_EQ(2u, D.ncell);
    EXPECT_EQ(18u, D.ncomp);

    unsigned nseg = 6;
    EXPECT_EQ(nseg, D.segments.size());

    // General sanity checks:

    ASSERT_EQ(D.ncell, D.cell_segment_part().size());
    ASSERT_EQ(D.ncell, D.cell_cv_part().size());

    ASSERT_EQ(D.ncomp, D.parent_cv.size());
    ASSERT_EQ(D.ncomp, D.cv_to_cell.size());
    ASSERT_EQ(D.ncomp, D.face_conductance.size());
    ASSERT_EQ(D.ncomp, D.cv_area.size());
    ASSERT_EQ(D.ncomp, D.cv_capacitance.size());

    // Partitions of CVs and segments by cell:

    using spair = std::pair<fvm_size_type, fvm_size_type>;
    using ipair = std::pair<fvm_index_type, fvm_index_type>;

    EXPECT_EQ(spair(0, 2),    D.cell_segment_part()[0]);
    EXPECT_EQ(spair(2, nseg), D.cell_segment_part()[1]);

    EXPECT_EQ(ipair(0, 5),       D.cell_cv_part()[0]);
    EXPECT_EQ(ipair(5, D.ncomp), D.cell_cv_part()[1]);

    // Segment and CV parent relationships:

    using ivec = std::vector<fvm_index_type>;

    EXPECT_EQ(ivec({0,0,1,2,3,5,5,6,7,8,9,10,11,12,9,14,15,16}), D.parent_cv);

    EXPECT_FALSE(D.segments[0].has_parent());
    EXPECT_EQ(0, D.segments[1].parent_cv);

    EXPECT_FALSE(D.segments[2].has_parent());
    EXPECT_EQ(5, D.segments[3].parent_cv);
    EXPECT_EQ(9, D.segments[4].parent_cv);
    EXPECT_EQ(9, D.segments[5].parent_cv);

    // Segment CV ranges (half-open, exclusing parent):

    EXPECT_EQ(ipair(0,1), D.segments[0].cv_range());
    EXPECT_EQ(ipair(1,5), D.segments[1].cv_range());
    EXPECT_EQ(ipair(5,6), D.segments[2].cv_range());
    EXPECT_EQ(ipair(6,10), D.segments[3].cv_range());
    EXPECT_EQ(ipair(10,14), D.segments[4].cv_range());
    EXPECT_EQ(ipair(14,18), D.segments[5].cv_range());

    // CV to cell index:

    for (auto ci: make_span(D.ncell)) {
        for (auto cv: make_span(D.cell_cv_part()[ci])) {
            EXPECT_EQ(ci, (fvm_size_type)D.cv_to_cell[cv]);
        }
    }
}

TEST(fvm_layout, area) {
    std::vector<mc_cell> cells = two_cell_system();
    check_two_cell_system(cells);

    fvm_discretization D = fvm_discretize(cells);

    // Note: stick models have constant diameter segments.
    // Refer to comment above for CV vs. segment layout.

    std::vector<double> A;
    for (auto ci: make_span(D.ncell)) {
        for (auto si: make_span(cells[ci].num_segments())) {
            A.push_back(area(cells[ci].segment(si)));
        }
    }

    unsigned n = 4; // compartments per dendritic segment
    EXPECT_FLOAT_EQ(A[0]+A[1]/(2*n), D.cv_area[0]);
    EXPECT_FLOAT_EQ(A[1]/n,     D.cv_area[1]);
    EXPECT_FLOAT_EQ(A[1]/n,     D.cv_area[2]);
    EXPECT_FLOAT_EQ(A[1]/n,     D.cv_area[3]);
    EXPECT_FLOAT_EQ(A[1]/(2*n), D.cv_area[4]);

    EXPECT_FLOAT_EQ(A[2]+A[3]/(2*n), D.cv_area[5]);
    EXPECT_FLOAT_EQ(A[3]/n,     D.cv_area[6]);
    EXPECT_FLOAT_EQ(A[3]/n,     D.cv_area[7]);
    EXPECT_FLOAT_EQ(A[3]/n,     D.cv_area[8]);
    EXPECT_FLOAT_EQ((A[3]+A[4]+A[5])/(2*n), D.cv_area[9]);
    EXPECT_FLOAT_EQ(A[4]/n,     D.cv_area[10]);
    EXPECT_FLOAT_EQ(A[4]/n,     D.cv_area[11]);
    EXPECT_FLOAT_EQ(A[4]/n,     D.cv_area[12]);
    EXPECT_FLOAT_EQ(A[4]/(2*n), D.cv_area[13]);
    EXPECT_FLOAT_EQ(A[5]/n,     D.cv_area[14]);
    EXPECT_FLOAT_EQ(A[5]/n,     D.cv_area[15]);
    EXPECT_FLOAT_EQ(A[5]/n,     D.cv_area[16]);
    EXPECT_FLOAT_EQ(A[5]/(2*n), D.cv_area[17]);

    // Confirm proportional allocation of surface capacitance:

    // CV 9 should have area-weighted sum of the specific
    // capacitance from segments 3, 4 and 5 (cell 1 segments
    // 1, 2 and 3 respectively).

    double cm1 = cells[1].segment(1)->cm;
    double cm2 = cells[1].segment(2)->cm;
    double cm3 = cells[1].segment(3)->cm;

    double c = A[3]/(2*n)*cm1+A[4]/(2*n)*cm2+A[5]/(2*n)*cm3;
    EXPECT_FLOAT_EQ(c, D.cv_capacitance[9]);

    // CV 5 should be a weighted sum of soma and first segment
    // capacitcance from cell 1.

    double cm0 = cells[1].soma()->cm;
    c = A[2]*cm0+A[3]/(2*n)*cm1;
    EXPECT_FLOAT_EQ(c, D.cv_capacitance[5]);

    // Confirm face conductance within a constant diameter
    // equals a/h·1/rL where a is the cross sectional
    // area, and h is the compartment length (given the
    // regular discretization).

    cable_segment* cable = cells[1].segment(2)->as_cable();
    double a = volume(cable)/cable->length();
    EXPECT_FLOAT_EQ(math::pi<double>()*0.8*0.8/4, a);

    double h = cable->length()/4;
    double g = a/h/cable->rL; // [µm·S/cm]
    g *= 100; // [µS]

    EXPECT_FLOAT_EQ(g, D.face_conductance[11]);
}

TEST(fvm_layout, mech_index) {
    std::vector<mc_cell> cells = two_cell_system();
    check_two_cell_system(cells);

    // Add four synapses of two varieties across the cells.
    cells[0].add_synapse({1, 0.4}, "expsyn");
    cells[0].add_synapse({1, 0.4}, "expsyn");
    cells[1].add_synapse({2, 0.4}, "exp2syn");
    cells[1].add_synapse({3, 0.4}, "expsyn");

    fvm_discretization D = fvm_discretize(cells);
    fvm_mechanism_data M = fvm_build_mechanism_data(global_default_catalogue(), cells, D);

    auto& hh_config = M.mechanisms.at("hh");
    auto& expsyn_config = M.mechanisms.at("expsyn");
    auto& exp2syn_config = M.mechanisms.at("exp2syn");

    using ivec = std::vector<fvm_index_type>;
    using fvec = std::vector<fvm_value_type>;

    // HH on somas of two cells, with CVs 0 and 5.
    // Proportional area contrib: soma area/CV area.

    EXPECT_EQ(mechanismKind::density, hh_config.kind);
    EXPECT_EQ(ivec({0,5}), hh_config.cv);

    fvec norm_area({area(cells[0].soma())/D.cv_area[0], area(cells[1].soma())/D.cv_area[5]});
    EXPECT_TRUE(testing::seq_almost_eq<double>(norm_area, hh_config.norm_area));

    // Three expsyn synapses, two 0.4 along segment 1, and one 0.4 along segment 5.
    // 0.4 along => second (non-parent) CV for segment.

    EXPECT_EQ(ivec({2, 2, 15}), expsyn_config.cv);

    // One exp2syn synapse, 0.4 along segment 4.

    EXPECT_EQ(ivec({11}), exp2syn_config.cv);

    // There should be a K and Na ion channel associated with each
    // hh mechanism node.

    ASSERT_EQ(1u, M.ions.count(ionKind::na));
    ASSERT_EQ(1u, M.ions.count(ionKind::k));
    EXPECT_EQ(0u, M.ions.count(ionKind::ca));

    EXPECT_EQ(ivec({0,5}), M.ions.at(ionKind::na).cv);
    EXPECT_EQ(ivec({0,5}), M.ions.at(ionKind::k).cv);
}

TEST(fvm_layout, synapse_targets) {
    std::vector<mc_cell> cells = two_cell_system();

    // Add synapses with different parameter values so that we can
    // ensure: 1) CVs for each synapse mechanism are sorted while
    // 2) the target index for each synapse corresponds to the
    // original ordering.

    const unsigned nsyn = 7;
    std::vector<double> syn_e(nsyn);
    for (auto i: count_along(syn_e)) {
        syn_e[i] = 0.1*(1+i);
    }

    auto syn_desc = [&](const char* name, int idx) {
        return mechanism_desc(name).set("e", syn_e.at(idx));
    };

    cells[0].add_synapse({1, 0.9}, syn_desc("expsyn", 0));
    cells[0].add_synapse({0, 0.5}, syn_desc("expsyn", 1));
    cells[0].add_synapse({1, 0.4}, syn_desc("expsyn", 2));

    cells[1].add_synapse({2, 0.4}, syn_desc("exp2syn", 3));
    cells[1].add_synapse({1, 0.4}, syn_desc("exp2syn", 4));
    cells[1].add_synapse({3, 0.4}, syn_desc("expsyn", 5));
    cells[1].add_synapse({3, 0.7}, syn_desc("exp2syn", 6));

    fvm_discretization D = fvm_discretize(cells);
    fvm_mechanism_data M = fvm_build_mechanism_data(global_default_catalogue(), cells, D);

    ASSERT_EQ(1u, M.mechanisms.count("expsyn"));
    ASSERT_EQ(1u, M.mechanisms.count("exp2syn"));

    auto& expsyn_cv = M.mechanisms.at("expsyn").cv;
    auto& expsyn_target = M.mechanisms.at("expsyn").target;
    auto& expsyn_e = value_by_key(M.mechanisms.at("expsyn").param_values, "e"_s).value();

    auto& exp2syn_cv = M.mechanisms.at("exp2syn").cv;
    auto& exp2syn_target = M.mechanisms.at("exp2syn").target;
    auto& exp2syn_e = value_by_key(M.mechanisms.at("exp2syn").param_values, "e"_s).value();

    EXPECT_TRUE(util::is_sorted(expsyn_cv));
    EXPECT_TRUE(util::is_sorted(exp2syn_cv));

    using uvec = std::vector<fvm_size_type>;
    uvec all_target_indices;
    util::append(all_target_indices, expsyn_target);
    util::append(all_target_indices, exp2syn_target);
    util::sort(all_target_indices);

    uvec nsyn_iota;
    util::assign(nsyn_iota, make_span(nsyn));
    EXPECT_EQ(nsyn_iota, all_target_indices);

    for (auto i: count_along(expsyn_target)) {
        EXPECT_EQ(syn_e[expsyn_target[i]], expsyn_e[i]);
    }

    for (auto i: count_along(exp2syn_target)) {
        EXPECT_EQ(syn_e[exp2syn_target[i]], exp2syn_e[i]);
    }
}


// TODO: migrate tests for proportional parameter setting.


namespace {
    double wm_impl(double wa, double xa) {
        return wa? xa/wa: 0;
    }

    template <typename... R>
    double wm_impl(double wa, double xa, double w, double x, R... rest) {
        return wm_impl(wa+w, xa+w*x, rest...);
    }

    // Computed weighted mean (w*x + ...) / (w + ...).
    template <typename... R>
    double wmean(double w, double x, R... rest) {
        return wm_impl(w, w*x, rest...);
    }
}

TEST(fvm_layout, density_norm_area) {
    // Test area-weighted linear combination of density mechanism parameters.

    // Create a cell with 4 segments:
    //   - Soma (segment 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - HH mechanism on all segments.
    //   - Dendritic segments are given 3 compartments each.
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/6 of segment 1 and the initial 1/6 of segments 2 and 3.
    //
    // The HH mechanism current density parameters ('gnabar', 'gkbar' and 'gl') are set
    // differently for each segment:
    //
    //   soma:      all default values (gnabar = 0.12, gkbar = .036, gl = .0003)
    //   segment 1: gl = .0002
    //   segment 2: gkbar = .05
    //   segment 3: gkbar = .07, gl = .0004
    //
    // Geometry:
    //   segment 1: 100 µm long, 1 µm diameter cylinder.
    //   segment 2: 200 µm long, diameter linear taper from 1 µm to 0.2 µm.
    //   segment 3: 150 µm long, 0.8 µm diameter cylinder.
    //
    // Use divided compartment view on segments to compute area contributions.

    std::vector<mc_cell> cells(1);
    mc_cell& c = cells[0];
    auto soma = c.add_soma(12.6157/2.0);

    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.1, 200);
    c.add_cable(1, section_kind::dendrite, 0.4, 0.4, 150);

    auto& segs = c.segments();

    double dflt_gkbar = .036;
    double dflt_gl = 0.0003;

    double seg1_gl = .0002;
    double seg2_gkbar = .05;
    double seg3_gkbar = .0004;
    double seg3_gl = .0004;

    for (int i = 0; i<4; ++i) {
        mc_segment& seg = *segs[i];
        seg.set_compartments(3);

        mechanism_desc hh("hh");
        switch (i) {
        case 1:
            hh["gl"] = seg1_gl;
            break;
        case 2:
            hh["gkbar"] = seg2_gkbar;
            break;
        case 3:
            hh["gkbar"] = seg3_gkbar;
            hh["gl"] = seg3_gl;
            break;
        default: ;
        }
        seg.add_mechanism(hh);
    }

    int ncv = 10;
    std::vector<double> expected_gkbar(ncv, dflt_gkbar);
    std::vector<double> expected_gl(ncv, dflt_gl);

    auto div_by_ends = [](const cable_segment* cable) {
        return div_compartment_by_ends(cable->num_compartments(), cable->radii(), cable->lengths());
    };
    double soma_area = area(soma);
    auto seg1_divs = div_by_ends(segs[1]->as_cable());
    auto seg2_divs = div_by_ends(segs[2]->as_cable());
    auto seg3_divs = div_by_ends(segs[3]->as_cable());

    // CV 0: mix of soma and left of segment 1
    expected_gl[0] = wmean(soma_area, dflt_gl, seg1_divs(0).left.area, seg1_gl);

    expected_gl[1] = seg1_gl;
    expected_gl[2] = seg1_gl;

    // CV 3: mix of right of segment 1 and left of segments 2 and 3.
    expected_gkbar[3] = wmean(seg1_divs(2).right.area, dflt_gkbar, seg2_divs(0).left.area, seg2_gkbar, seg3_divs(0).left.area, seg3_gkbar);
    expected_gl[3] = wmean(seg1_divs(2).right.area, seg1_gl, seg2_divs(0).left.area, dflt_gl, seg3_divs(0).left.area, seg3_gl);

    // CV 4-6: just segment 2
    expected_gkbar[4] = seg2_gkbar;
    expected_gkbar[5] = seg2_gkbar;
    expected_gkbar[6] = seg2_gkbar;

    // CV 7-9: just segment 3
    expected_gkbar[7] = seg3_gkbar;
    expected_gkbar[8] = seg3_gkbar;
    expected_gkbar[9] = seg3_gkbar;
    expected_gl[7] = seg3_gl;
    expected_gl[8] = seg3_gl;
    expected_gl[9] = seg3_gl;

    fvm_discretization D = fvm_discretize(cells);
    fvm_mechanism_data M = fvm_build_mechanism_data(global_default_catalogue(), cells, D);

    // Check CV area assumptions.
    // Note: area integrator used here and in `fvm_multicell` may differ, and so areas computed may
    // differ some due to rounding area, even given that we're dealing with simple truncated cones
    // for segments. Check relative error within a tolerance of (say) 10 epsilon.

    double area_relerr = 10*std::numeric_limits<double>::epsilon();
    EXPECT_TRUE(testing::near_relative(D.cv_area[0],
        soma_area+seg1_divs(0).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(D.cv_area[1],
        seg1_divs(0).right.area+seg1_divs(1).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(D.cv_area[3],
        seg1_divs(2).right.area+seg2_divs(0).left.area+seg3_divs(0).left.area, area_relerr));
    EXPECT_TRUE(testing::near_relative(D.cv_area[6],
        seg2_divs(2).right.area, area_relerr));

    // Grab the HH parameters from the mechanism.

    EXPECT_EQ(1u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("hh"));
    auto& hh_params = M.mechanisms.at("hh").param_values;

    auto& gkbar = value_by_key(hh_params, "gkbar"_s).value();
    auto& gl = value_by_key(hh_params, "gl"_s).value();

    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gkbar, gkbar));
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gl, gl));
}

TEST(fvm_layout, ion_weights) {
    // Create a cell with 4 segments:
    //   - Soma (segment 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - Dendritic segments are given 1 compartments each.
    //
    //         /
    //        d2
    //       /
    //   s0-d1
    //       \.
    //        d3
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/2 of segment 1 and the initial 1/2 of segments 2 and 3.
    //
    // Geometry:
    //   soma 0: radius 5 µm
    //   dend 1: 100 µm long, 1 µm diameter cynlinder
    //   dend 2: 200 µm long, 1 µm diameter cynlinder
    //   dend 3: 100 µm long, 1 µm diameter cynlinder
    //
    // The radius of the soma is chosen such that the surface area of soma is
    // the same as a 100µm dendrite, which makes it easier to describe the
    // expected weights.

    auto construct_cell = [](mc_cell& c) {
        c.add_soma(5);

        c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
        c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 200);
        c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

        for (auto& s: c.segments()) s->set_compartments(1);
    };

    using uvec = std::vector<fvm_size_type>;
    using ivec = std::vector<fvm_index_type>;
    using fvec = std::vector<fvm_value_type>;

    uvec mech_segs[] = {
        {0}, {0,2}, {2, 3}, {0, 1, 2, 3}, {3}
    };

    ivec expected_ion_cv[] = {
        {0}, {0, 1, 2}, {1, 2, 3}, {0, 1, 2, 3}, {1, 3}
    };

    fvec expected_iconc_norm_area[] = {
        {1./3}, {1./3, 1./2, 0.}, {1./4, 0., 0.}, {0., 0., 0., 0.}, {3./4, 0.}
    };

    for (auto run: count_along(mech_segs)) {
        std::vector<mc_cell> cells(1);
        mc_cell& c = cells[0];
        construct_cell(c);

        for (auto i: mech_segs[run]) {
            c.segments()[i]->add_mechanism("test_ca");
        }

        fvm_discretization D = fvm_discretize(cells);
        fvm_mechanism_data M = fvm_build_mechanism_data(global_default_catalogue(), cells, D);

        ASSERT_EQ(1u, M.ions.count(ionKind::ca));
        auto& ca = M.ions.at(ionKind::ca);

        EXPECT_EQ(expected_ion_cv[run], ca.cv);
        EXPECT_TRUE(testing::seq_almost_eq<fvm_value_type>(expected_iconc_norm_area[run], ca.iconc_norm_area));

        EXPECT_TRUE(util::all_of(ca.econc_norm_area, [](fvm_value_type v) { return v==1.; }));
    }
}

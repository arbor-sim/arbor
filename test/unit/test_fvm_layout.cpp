#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/math.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>

#include <arborio/label_parse.hpp>

#include <arborenv/default_env.hpp>

#include "backends/multicore/fvm.hpp"
#include "fvm_lowered_cell.hpp"
#include "fvm_lowered_cell_impl.hpp"
#include "fvm_layout.hpp"
#include "util/maputil.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "io/sepval.hpp"

#include "common.hpp"
#include "common_morphologies.hpp"
#include "unit_test_catalogue.hpp"
#include "../common_cells.hpp"

using namespace std::string_literals;
using namespace arb;
using namespace arborio::literals;

using util::make_span;
using util::count_along;
using util::ptr_by_key;
using util::value_by_key;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

// instantiate template class
template class arb::fvm_lowered_cell_impl<arb::multicore::backend>;

namespace {
    struct system {
        std::vector<soma_cell_builder> builders;
        std::vector<cable_cell_description> descriptions;

        std::vector<arb::cable_cell> cells() const {
            std::vector<arb::cable_cell> C;
            C.reserve(descriptions.size());
            for (auto& d: descriptions) {
                C.emplace_back(d);
            }
            return C;
        }

    };

    system two_cell_system() {
        system s;
        auto& descriptions = s.descriptions;

        // Cell 0: simple ball and stick
        {
            soma_cell_builder builder(12.6157/2.0);
            builder.add_branch(0, 200, 1.0/2, 1.0/2, 4, "dend");

            auto description = builder.make_cell();
            description.decorations.paint("soma"_lab, density("hh"));
            description.decorations.paint("dend"_lab, density("pas"));
            description.decorations.place(builder.location({1,1}), i_clamp{5, 80, 0.3}, "clamp");

            s.builders.push_back(std::move(builder));
            descriptions.push_back(description);
        }

        // Cell 1: ball and 3-stick, but with uneven dendrite
        // length and heterogeneous electrical properties:
        //
        // Bulk resistivity: 90 Ω·cm
        // capacitance:
        //    soma:      0.01  F/m² [default]
        //    branch 1:  0.017 F/m²
        //    branch 2:  0.013 F/m²
        //    branch 3:  0.018 F/m²
        //
        // Soma diameter: 14 µm
        // Some mechanisms: HH (default params)
        //
        // Branch 1 diameter: 1 µm
        // Branch 1 length:   200 µm
        //
        // Branch 2 diameter: 0.8 µm
        // Branch 2 length:   300 µm
        //
        // Branch 3 diameter: 0.7 µm
        // Branch 3 length:   180 µm
        //
        // Dendrite mechanisms: passive (default params).
        // Stimulus at end of branch 2, amplitude 0.45.
        // Stimulus at end of branch 3, amplitude -0.2.
        //
        // All dendrite branches with 4 compartments.

        {
            soma_cell_builder b(7.);
            auto b1 = b.add_branch(0, 200, 0.5,  0.5, 4,  "dend");
            auto b2 = b.add_branch(1, 300, 0.4,  0.4, 4,  "dend");
            auto b3 = b.add_branch(1, 180, 0.35, 0.35, 4, "dend");
            auto desc = b.make_cell();

            desc.decorations.paint("soma"_lab, density("hh"));
            desc.decorations.paint("dend"_lab, density("pas"));

            using ::arb::reg::branch;
            auto c1 = reg::cable(b1-1, b.location({b1, 0}).pos, 1);
            auto c2 = reg::cable(b2-1, b.location({b2, 0}).pos, 1);
            auto c3 = reg::cable(b3-1, b.location({b3, 0}).pos, 1);
            desc.decorations.paint(c1, membrane_capacitance{0.017});
            desc.decorations.paint(c2, membrane_capacitance{0.013});
            desc.decorations.paint(c3, membrane_capacitance{0.018});

            desc.decorations.place(b.location({2,1}), i_clamp{5.,  80., 0.45}, "clamo0");
            desc.decorations.place(b.location({3,1}), i_clamp{40., 10.,-0.2}, "clamp1");

            desc.decorations.set_default(axial_resistivity{90});

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }

        return s;
    }

    void check_two_cell_system(std::vector<cable_cell>& cells) {
        ASSERT_EQ(2u, cells.size());
        ASSERT_EQ(1u, cells[0].morphology().num_branches());
        ASSERT_EQ(3u, cells[1].morphology().num_branches());
    }

    system six_cell_gj_system() {
        system s;
        auto& descriptions = s.descriptions;

        // Cell 0: simple soma cell, 1 CV.
        {
            soma_cell_builder builder(12.6157/2.0);

            auto desc = builder.make_cell();
            desc.decorations.paint("soma"_lab, density("hh"));

            s.builders.push_back(std::move(builder));
            descriptions.push_back(desc);
        }
        // Cell 1: ball and 3-stick, 2 CVs per dendrite, 1 CV for the soma.
        {
            soma_cell_builder b(7.);
            b.add_branch(0, 200, 0.5,  0.5, 2,  "dend");
            b.add_branch(1, 300, 0.4,  0.4, 2,  "dend");
            b.add_branch(1, 180, 0.35, 0.35, 2, "dend");
            auto desc = b.make_cell();

            desc.decorations.paint("soma"_lab, density("hh"));
            desc.decorations.paint("dend"_lab, density("pas"));

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }
        // Cell 2: ball and stick, 1 CV for the soma, 2 CVs for the dendrite.
        {
            soma_cell_builder b(7.);
            b.add_branch(0, 200, 0.5,  0.5, 2,  "dend");
            auto desc = b.make_cell();

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }
        // Cell 3: simple soma cell, 1 CV. 1 gap junction.
        {
            soma_cell_builder b(7.);
            auto desc = b.make_cell();

            desc.decorations.paint("soma"_lab, density("hh"));

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }
        // Cell 4: ball and stick, 1 CV for the soma, 3 CVs for the dendrite.
        {
            soma_cell_builder b(7.);
            b.add_branch(0, 200, 0.5,  0.5, 3,  "dend");
            auto desc = b.make_cell();

            desc.decorations.paint("soma"_lab, density("pas"));

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }
        // Cell 5: ball and stick, 1 CV for the soma, 1 CV for the dendrite.
        {
            soma_cell_builder b(7.);
            b.add_branch(0, 200, 0.5,  0.5, 1,  "dend");
            auto desc = b.make_cell();

            s.builders.push_back(std::move(b));
            descriptions.push_back(desc);
        }
        return s;
    }

} // namespace

template<typename P>
void check_compatible_mechanism_failure(cable_cell_global_properties gprop, P painter) {
    auto system = two_cell_system();

    painter(system);

    auto cells = system.cells();
    check_two_cell_system(cells);
    std::vector<cell_gid_type> gids = {0,1};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);

    EXPECT_THROW(fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D), arb::cable_cell_error);
}

TEST(fvm_layout, compatible_mechanisms) {
    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    check_compatible_mechanism_failure(gprop, [](auto& sys) {sys.descriptions[0].decorations.place(sys.builders[0].location({1, 0.4}), synapse("hh"), "syn0"); });
    check_compatible_mechanism_failure(gprop, [](auto& sys) {sys.descriptions[1].decorations.paint(sys.builders[1].cable(mcable{0}), density("expsyn")); });
    check_compatible_mechanism_failure(gprop, [](auto& sys) {sys.descriptions[0].decorations.set_default(ion_reversal_potential_method{"na", "expsyn"}); });

    gprop.default_parameters.reversal_potential_method["na"] = "pas";
    check_compatible_mechanism_failure(gprop, [](auto& sys) {});
}

TEST(fvm_layout, mech_index) {
    auto system = two_cell_system();
    auto& descriptions = system.descriptions;
    auto& builders = system.builders;

    // Add four synapses of two varieties across the cells.
    descriptions[0].decorations.place(builders[0].location({1, 0.4}), synapse("expsyn"), "syn0");
    descriptions[0].decorations.place(builders[0].location({1, 0.4}), synapse("expsyn"), "syn1");
    descriptions[1].decorations.place(builders[1].location({2, 0.4}), synapse("exp2syn"), "syn3");
    descriptions[1].decorations.place(builders[1].location({3, 0.4}), synapse("expsyn"), "syn4");

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    auto cells = system.cells();
    check_two_cell_system(cells);
    std::vector<cell_gid_type> gids = {0,1};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
    fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

    auto& hh_config = M.mechanisms.at("hh");
    auto& expsyn_config = M.mechanisms.at("expsyn");
    auto& exp2syn_config = M.mechanisms.at("exp2syn");

    using ivec = std::vector<arb_index_type>;

    // HH on somas of two cells, with CVs 0 and 5.
    // Proportional area contrib: soma area/CV area.

    EXPECT_EQ((unsigned)arb_mechanism_kind_density, hh_config.kind);
    EXPECT_EQ(ivec({0,6}), hh_config.cv);

    // Three expsyn synapses, two 0.4 along branch 1, and one 0.4 along branch 5.
    // These two synapses can be coalesced into 1 synapse
    // 0.4 along => second (non-parent) CV for branch.

    EXPECT_EQ(ivec({3, 17}), expsyn_config.cv);

    // One exp2syn synapse, 0.4 along branch 4.

    EXPECT_EQ(ivec({13}), exp2syn_config.cv);

    // There should be a K and Na ion channel associated with each
    // hh mechanism node.

    ASSERT_EQ(1u, M.ions.count("na"s));
    ASSERT_EQ(1u, M.ions.count("k"s));
    EXPECT_EQ(0u, M.ions.count("ca"s));

    EXPECT_EQ(ivec({0,6}), M.ions.at("na"s).cv);
    EXPECT_EQ(ivec({0,6}), M.ions.at("k"s).cv);
}

struct exp_instance {
    int cv;
    int multiplicity;
    std::vector<unsigned> targets;
    double e;
    double tau;

    template <typename Seq>
    exp_instance(int cv, const Seq& tgts, double e, double tau):
        cv(cv), multiplicity(std::size(tgts)), e(e), tau(tau)
    {
        targets.reserve(std::size(tgts));
        for (auto t: tgts) targets.push_back(t);
        util::sort(targets);
    }

    bool matches(const exp_instance& I) const {
        return I.cv==cv && I.e==e && I.tau==tau && I.targets==targets;
    }

    bool is_in(const arb::fvm_mechanism_config& C) const {
        std::vector<unsigned> _;
        auto part = util::make_partition(_, C.multiplicity);
        auto& evals = *ptr_by_key(C.param_values, "e");
        // Handle both expsyn and exp2syn by looking for "tau1" if "tau"
        // parameter is not found.
        auto& tauvals = *(value_by_key(C.param_values, "tau")?
            ptr_by_key(C.param_values, "tau"):
            ptr_by_key(C.param_values, "tau1"));

        for (auto i: make_span(C.multiplicity.size())) {
            exp_instance other(C.cv[i],
                               util::subrange_view(C.target, part[i]),
                               evals[i],
                               tauvals[i]);
            if (matches(other)) return true;
        }
        return false;
    }
};

TEST(fvm_layout, coalescing_synapses) {
    using ivec = std::vector<arb_index_type>;

    auto syn_desc = [&](const char* name, double val0, double val1) {
        mechanism_desc m(name);
        m.set("e", val0);
        m.set("tau", val1);
        return m;
    };

    auto syn_desc_2 = [&](const char* name, double val0, double val1) {
        mechanism_desc m(name);
        m.set("e", val0);
        m.set("tau1", val1);
        return m;
    };

    cable_cell_global_properties gprop_no_coalesce;
    gprop_no_coalesce.default_parameters = neuron_parameter_defaults;
    gprop_no_coalesce.coalesce_synapses = false;

    cable_cell_global_properties gprop_coalesce;
    gprop_coalesce.default_parameters = neuron_parameter_defaults;
    gprop_coalesce.coalesce_synapses = true;

    using L=std::initializer_list<unsigned>;

    soma_cell_builder builder(12.6157/2.0);
    builder.add_branch(0, 200, 1.0/2, 1.0/2, 4, "dend");

    {
        auto desc = builder.make_cell();

        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn0");
        desc.decorations.place(builder.location({1, 0.5}), synapse("expsyn"), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn2");
        desc.decorations.place(builder.location({1, 0.9}), synapse("expsyn"), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        auto &expsyn_config = M.mechanisms.at("expsyn");
        EXPECT_EQ(ivec({2, 3, 4, 5}), expsyn_config.cv);
        EXPECT_EQ(ivec({1, 1, 1, 1}), expsyn_config.multiplicity);
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn0");
        desc.decorations.place(builder.location({1, 0.5}), synapse("exp2syn"), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn2");
        desc.decorations.place(builder.location({1, 0.9}), synapse("exp2syn"), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        auto &expsyn_config = M.mechanisms.at("expsyn");
        EXPECT_EQ(ivec({2, 4}), expsyn_config.cv);
        EXPECT_EQ(ivec({1, 1}), expsyn_config.multiplicity);

        auto &exp2syn_config = M.mechanisms.at("exp2syn");
        EXPECT_EQ(ivec({3, 5}), exp2syn_config.cv);
        EXPECT_EQ(ivec({1, 1}), exp2syn_config.multiplicity);
    }
    {
        auto desc = builder.make_cell();

        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn0");
        desc.decorations.place(builder.location({1, 0.5}), synapse("expsyn"), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn2");
        desc.decorations.place(builder.location({1, 0.9}), synapse("expsyn"), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_no_coalesce, {cell}, {0}, {{0, {}}}, D);

        auto &expsyn_config = M.mechanisms.at("expsyn");
        EXPECT_EQ(ivec({2, 3, 4, 5}), expsyn_config.cv);
        EXPECT_TRUE(expsyn_config.multiplicity.empty());
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn0");
        desc.decorations.place(builder.location({1, 0.5}), synapse("exp2syn"), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn2");
        desc.decorations.place(builder.location({1, 0.9}), synapse("exp2syn"), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_no_coalesce, {cell}, {0}, {{0, {}}}, D);

        auto &expsyn_config = M.mechanisms.at("expsyn");
        EXPECT_EQ(ivec({2, 4}), expsyn_config.cv);
        EXPECT_TRUE(expsyn_config.multiplicity.empty());

        auto &exp2syn_config = M.mechanisms.at("exp2syn");
        EXPECT_EQ(ivec({3, 5}), exp2syn_config.cv);
        EXPECT_TRUE(exp2syn_config.multiplicity.empty());
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn0");
        desc.decorations.place(builder.location({1, 0.3}), synapse("expsyn"), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn2");
        desc.decorations.place(builder.location({1, 0.7}), synapse("expsyn"), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        auto &expsyn_config = M.mechanisms.at("expsyn");
        EXPECT_EQ(ivec({2, 4}), expsyn_config.cv);
        EXPECT_EQ(ivec({2, 2}), expsyn_config.multiplicity);
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 0, 0.2)), "syn0");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 0, 0.2)), "syn1");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 0.1, 0.2)), "syn2");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc("expsyn", 0.1, 0.2)), "syn3");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        std::vector<exp_instance> instances{
            exp_instance(2, L{0, 1}, 0., 0.2),
            exp_instance(2, L{2}, 0.1, 0.2),
            exp_instance(4, L{3}, 0.1, 0.2),
        };
        auto& config = M.mechanisms.at("expsyn");
        for (auto& instance: instances) {
            EXPECT_TRUE(instance.is_in(config));
        }
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc("expsyn", 0, 3)), "syn0");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc("expsyn", 1, 3)), "syn1");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc("expsyn", 0, 3)), "syn2");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc("expsyn", 1, 3)), "syn3");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 0, 2)), "syn4");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 1, 2)), "syn5");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 0, 2)), "syn6");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn", 1, 2)), "syn7");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        std::vector<exp_instance> instances{
            exp_instance(2, L{4, 6}, 0.0, 2.0),
            exp_instance(2, L{5, 7}, 1.0, 2.0),
            exp_instance(4, L{0, 2}, 0.0, 3.0),
            exp_instance(4, L{1, 3}, 1.0, 3.0),
        };
        auto& config = M.mechanisms.at("expsyn");
        for (auto& instance: instances) {
            EXPECT_TRUE(instance.is_in(config));
        }
    }
    {
        auto desc = builder.make_cell();

        // Add synapses of two varieties.
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn",  1, 2)), "syn0");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc_2("exp2syn", 4, 1)), "syn1");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn",  1, 2)), "syn2");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn",  5, 1)), "syn3");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc_2("exp2syn", 1, 3)), "syn4");
        desc.decorations.place(builder.location({1, 0.3}), synapse(syn_desc("expsyn",  1, 2)), "syn5");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc_2("exp2syn", 2, 2)), "syn6");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc_2("exp2syn", 2, 1)), "syn7");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc_2("exp2syn", 2, 1)), "syn8");
        desc.decorations.place(builder.location({1, 0.7}), synapse(syn_desc_2("exp2syn", 2, 2)), "syn9");

        cable_cell cell(desc);
        fvm_cv_discretization D = fvm_cv_discretize({cell}, neuron_parameter_defaults);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop_coalesce, {cell}, {0}, {{0, {}}}, D);

        for (auto &instance: {exp_instance(2, L{0,2,5}, 1, 2),
                              exp_instance(2, L{3},     5, 1)}) {
            EXPECT_TRUE(instance.is_in(M.mechanisms.at("expsyn")));
        }

        for (auto &instance: {exp_instance(2, L{4},   1, 3),
                              exp_instance(2, L{1},   4, 1),
                              exp_instance(4, L{7,8}, 2, 1),
                              exp_instance(4, L{6,9}, 2, 2)}) {
            EXPECT_TRUE(instance.is_in(M.mechanisms.at("exp2syn")));
        }
    }
}

TEST(fvm_layout, synapse_targets) {
    auto system = two_cell_system();
    auto& descriptions = system.descriptions;
    auto& builders = system.builders;

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

    descriptions[0].decorations.place(builders[0].location({1, 0.9}), synapse(syn_desc("expsyn", 0)), "syn0");
    descriptions[0].decorations.place(builders[0].location({0, 0.5}), synapse(syn_desc("expsyn", 1)), "syn1");
    descriptions[0].decorations.place(builders[0].location({1, 0.4}), synapse(syn_desc("expsyn", 2)), "syn2");

    descriptions[1].decorations.place(builders[1].location({2, 0.4}), synapse(syn_desc("exp2syn", 3)), "syn3");
    descriptions[1].decorations.place(builders[1].location({1, 0.4}), synapse(syn_desc("exp2syn", 4)), "syn4");
    descriptions[1].decorations.place(builders[1].location({3, 0.4}), synapse(syn_desc("expsyn", 5)), "syn5");
    descriptions[1].decorations.place(builders[1].location({3, 0.7}), synapse(syn_desc("exp2syn", 6)), "syn6");

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    auto cells = system.cells();
    std::vector<cell_gid_type> gids = {0,1};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
    fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

    ASSERT_EQ(1u, M.mechanisms.count("expsyn"));
    ASSERT_EQ(1u, M.mechanisms.count("exp2syn"));

    auto& expsyn_cv = M.mechanisms.at("expsyn").cv;
    auto& expsyn_target = M.mechanisms.at("expsyn").target;
    auto& expsyn_e = *ptr_by_key(M.mechanisms.at("expsyn").param_values, "e"s);

    auto& exp2syn_cv = M.mechanisms.at("exp2syn").cv;
    auto& exp2syn_target = M.mechanisms.at("exp2syn").target;
    auto& exp2syn_e = *ptr_by_key(M.mechanisms.at("exp2syn").param_values, "e"s);

    EXPECT_TRUE(util::is_sorted(expsyn_cv));
    EXPECT_TRUE(util::is_sorted(exp2syn_cv));

    using uvec = std::vector<arb_size_type>;
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

TEST(fvm_lowered, gj_example_0) {
    auto context = make_context({arbenv::default_concurrency(), -1});

    class gap_recipe: public recipe {
    public:
        gap_recipe(std::vector<cable_cell> cells, arb::cable_cell_global_properties gprop) : gprop_(gprop), cells_(cells) {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return cells_[gid];
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            conns.push_back(gap_junction_connection({(gid+1)%2, "gj", lid_selection_policy::assert_univalent}, {"gj", lid_selection_policy::assert_univalent}, 0.5));
            return conns;
        }
        std::any get_global_properties(cell_kind) const override {
            return gprop_;
        }
    protected:
        arb::cable_cell_global_properties gprop_;
        std::vector<cable_cell> cells_;
        cell_size_type n_ = 2;
    };

    std::vector<cable_cell> cells;
    soma_cell_builder b0(2.1);
    b0.add_branch(0, 10, 0.3, 0.2, 5, "dend");
    auto c0 = b0.make_cell();
    auto loc_0 = b0.location({1, 0.8});
    c0.decorations.place(loc_0, junction("gj"), "gj");
    cells.push_back(c0);

    soma_cell_builder b1(2.4);
    b1.add_branch(0, 10, 0.3, 0.2, 2, "dend");
    auto c1 = b1.make_cell();
    auto loc_1 = b1.location({1, 1});
    c1.decorations.place(loc_1, junction("gj", {{"g", 0.5}}), "gj");
    cells.push_back(c1);

    // Check the GJ CV map
    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    std::vector<cell_gid_type> gids = {0, 1};

    auto D = fvm_cv_discretize(cells, gprop.default_parameters, *context);
    auto gj_cvs = fvm_build_gap_junction_cv_map(cells, gids, D);

    auto cv_0 = D.geometry.location_cv(0, loc_0, cv_prefer::cv_nonempty);
    auto cv_1 = D.geometry.location_cv(1, loc_1, cv_prefer::cv_nonempty);

    EXPECT_EQ(2u, gj_cvs.size());

    EXPECT_EQ(cv_0, gj_cvs.at(cell_member_type{0, 0}));
    EXPECT_EQ(cv_1, gj_cvs.at(cell_member_type{1, 0}));

    // Check the resolved GJ connections
    fvm_cell fvcell(*context);
    gap_recipe rec(cells, gprop);

    auto fvm_info = fvcell.initialize(gids, rec);
    auto gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gj_cvs, rec);

    EXPECT_EQ(1u, gj_conns.at(0).size());
    EXPECT_EQ(1u, gj_conns.at(1).size());

    auto gj0 = fvm_gap_junction{0, cv_0, cv_1, 0.5};
    auto gj1 = fvm_gap_junction{0, cv_1, cv_0, 0.5};

    EXPECT_EQ(gj0, gj_conns.at(0).front());
    EXPECT_EQ(gj1, gj_conns.at(1).front());

    // Check the GJ mechanism data
    auto M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D, *context);

    EXPECT_EQ(1u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("gj"));

    const auto& gj_data = M.mechanisms["gj"];
    const auto& gj_g_param = *ptr_by_key(gj_data.param_values, "g"s);

    std::vector<std::tuple<int, int, double, double>> expected_gj_values = {
        {cv_0, cv_1, 0.5, 1.},
        {cv_1, cv_0, 0.5, 0.5},
    };
    util::sort(expected_gj_values);

    std::vector<int> expected_gj_cv, expected_gj_peer_cv;
    std::vector<double> expected_gj_weight, expected_gj_g;
    util::assign(expected_gj_cv,      util::transform_view(expected_gj_values, [](const auto& x){return std::get<0>(x);}));
    util::assign(expected_gj_peer_cv, util::transform_view(expected_gj_values, [](const auto& x){return std::get<1>(x);}));
    util::assign(expected_gj_weight,  util::transform_view(expected_gj_values, [](const auto& x){return std::get<2>(x);}));
    util::assign(expected_gj_g,       util::transform_view(expected_gj_values, [](const auto& x){return std::get<3>(x);}));

    EXPECT_EQ(2u, gj_data.cv.size());
    EXPECT_EQ(expected_gj_cv,      gj_data.cv);
    EXPECT_EQ(expected_gj_peer_cv, gj_data.peer_cv);
    EXPECT_EQ(expected_gj_weight,  gj_data.local_weight);
    EXPECT_EQ(expected_gj_g,       gj_g_param);
}

TEST(fvm_lowered, gj_example_1) {
    auto context = make_context({arbenv::default_concurrency(), -1});

    class gap_recipe: public recipe {
    public:
        gap_recipe(std::vector<cable_cell> cells, arb::cable_cell_global_properties gprop) : gprop_(gprop), cells_(cells) {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return cells_[gid];
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            switch (gid) {
            case 0:
                return {
                    gap_junction_connection({2, "gj0"}, {"gj1"}, 0.01),
                    gap_junction_connection({1, "gj0"}, {"gj0"}, 0.03),
                    gap_junction_connection({1, "gj1"}, {"gj0"}, 0.04)
                };
            case 1:
                return {
                    gap_junction_connection({0, "gj0"}, {"gj0"}, 0.03),
                    gap_junction_connection({0, "gj0"}, {"gj1"}, 0.04),
                    gap_junction_connection({2, "gj1"}, {"gj1"}, 0.02),
                    gap_junction_connection({2, "gj2"}, {"gj3"}, 0.01)
                };
            case 2:
                return {
                    gap_junction_connection({0, "gj1"}, {"gj0"}, 0.01),
                    gap_junction_connection({1, "gj1"}, {"gj1"}, 0.02),
                    gap_junction_connection({1, "gj3"}, {"gj2"}, 0.01)
                };
            default : return {};
            }
            return conns;
        }
        std::any get_global_properties(cell_kind) const override {
            return gprop_;
        }
    protected:
        arb::cable_cell_global_properties gprop_;
        std::vector<cable_cell> cells_;
        cell_size_type n_ = 3;
    };

    soma_cell_builder b0(2.1);
    b0.add_branch(0, 8, 0.3, 0.2, 4, "dend");

    auto c0 = b0.make_cell();
    mlocation c0_gj[2] = {b0.location({1, 1}), b0.location({1, 0.5})};

    c0.decorations.place(c0_gj[0], junction{"gj"}, "gj0");
    c0.decorations.place(c0_gj[1], junction{"gj"}, "gj1");

    soma_cell_builder b1(1.4);
    b1.add_branch(0, 12, 0.3, 0.5, 6, "dend");
    b1.add_branch(1,  9, 0.3, 0.2, 3, "dend");
    b1.add_branch(1,  5, 0.2, 0.2, 5, "dend");

    auto c1 = b1.make_cell();
    mlocation c1_gj[4] = {b1.location({2, 1}), b1.location({1, 1}), b1.location({1, 0.45}), b1.location({1, 0.1})};

    c1.decorations.place(c1_gj[0], junction{"gj"}, "gj0");
    c1.decorations.place(c1_gj[1], junction{"gj"}, "gj1");
    c1.decorations.place(c1_gj[2], junction{"gj"}, "gj2");
    c1.decorations.place(c1_gj[3], junction{"gj"}, "gj3");

    soma_cell_builder b2(2.9);
    b2.add_branch(0, 4, 0.3, 0.5, 2, "dend");
    b2.add_branch(1, 6, 0.4, 0.2, 2, "dend");
    b2.add_branch(1, 8, 0.1, 0.2, 2, "dend");
    b2.add_branch(2, 4, 0.2, 0.2, 2, "dend");
    b2.add_branch(2, 4, 0.2, 0.2, 2, "dend");

    auto c2 = b2.make_cell();
    mlocation c2_gj[3] = {b2.location({1, 0.5}), b2.location({4, 1}), b2.location({2, 1})};

    c2.decorations.place(c2_gj[0], junction{"gj"}, "gj0");
    c2.decorations.place(c2_gj[1], junction{"gj"}, "gj1");
    c2.decorations.place(c2_gj[2], junction{"gj"}, "gj2");

    // Check the GJ CV map
    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    std::vector<cable_cell> cells{c0, c1, c2};
    std::vector<cell_gid_type> gids = {0, 1, 2};

    auto D = fvm_cv_discretize(cells, neuron_parameter_defaults, *context);
    unsigned c0_gj_cv[2], c1_gj_cv[4], c2_gj_cv[3];
    for (int i = 0; i<2; ++i) c0_gj_cv[i] = D.geometry.location_cv(0, c0_gj[i], cv_prefer::cv_nonempty);
    for (int i = 0; i<4; ++i) c1_gj_cv[i] = D.geometry.location_cv(1, c1_gj[i], cv_prefer::cv_nonempty);
    for (int i = 0; i<3; ++i) c2_gj_cv[i] = D.geometry.location_cv(2, c2_gj[i], cv_prefer::cv_nonempty);

    auto gj_cvs = fvm_build_gap_junction_cv_map(cells, gids, D);

    EXPECT_EQ(9u, gj_cvs.size());
    EXPECT_EQ(c0_gj_cv[0], gj_cvs.at(cell_member_type{0, 0}));
    EXPECT_EQ(c0_gj_cv[1], gj_cvs.at(cell_member_type{0, 1}));
    EXPECT_EQ(c1_gj_cv[0], gj_cvs.at(cell_member_type{1, 0}));
    EXPECT_EQ(c1_gj_cv[1], gj_cvs.at(cell_member_type{1, 1}));
    EXPECT_EQ(c1_gj_cv[2], gj_cvs.at(cell_member_type{1, 2}));
    EXPECT_EQ(c1_gj_cv[3], gj_cvs.at(cell_member_type{1, 3}));
    EXPECT_EQ(c2_gj_cv[0], gj_cvs.at(cell_member_type{2, 0}));
    EXPECT_EQ(c2_gj_cv[1], gj_cvs.at(cell_member_type{2, 1}));
    EXPECT_EQ(c2_gj_cv[2], gj_cvs.at(cell_member_type{2, 2}));

    // Check the resolved GJ connections
    fvm_cell fvcell(*context);
    gap_recipe rec(cells, gprop);

    auto fvm_info = fvcell.initialize(gids, rec);
    auto gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gj_cvs, rec);

    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> expected;
    expected[0] = {
        {0, c0_gj_cv[0], c1_gj_cv[0], 0.03},
        {0, c0_gj_cv[0], c1_gj_cv[1], 0.04},
        {1, c0_gj_cv[1], c2_gj_cv[0], 0.01},
    };
    expected[1] = {
        {0, c1_gj_cv[0], c0_gj_cv[0], 0.03},
        {1, c1_gj_cv[1], c0_gj_cv[0], 0.04},
        {1, c1_gj_cv[1], c2_gj_cv[1], 0.02},
        {3, c1_gj_cv[3], c2_gj_cv[2], 0.01}
    };
    expected[2] = {
        {0, c2_gj_cv[0], c0_gj_cv[1], 0.01},
        {1, c2_gj_cv[1], c1_gj_cv[1], 0.02},
        {2, c2_gj_cv[2], c1_gj_cv[3], 0.01}
    };

    util::sort(expected.at(0));
    util::sort(expected.at(1));
    util::sort(expected.at(2));

    EXPECT_EQ(expected.at(0), gj_conns.at(0));
    EXPECT_EQ(expected.at(1), gj_conns.at(1));
    EXPECT_EQ(expected.at(2), gj_conns.at(2));

    // Check the GJ mechanism data
    auto M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D, *context);

    EXPECT_EQ(1u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("gj"));

    const auto& gj_data = M.mechanisms["gj"];
    const auto& gj_g_param = *ptr_by_key(gj_data.param_values, "g"s);

    std::vector<std::tuple<int, int, double, double>> expected_gj_values = {
        {c0_gj_cv[0], c1_gj_cv[0], 0.03, 1.},
        {c0_gj_cv[0], c1_gj_cv[1], 0.04, 1.},
        {c0_gj_cv[1], c2_gj_cv[0], 0.01, 1.},
        {c1_gj_cv[0], c0_gj_cv[0], 0.03, 1.},
        {c1_gj_cv[1], c0_gj_cv[0], 0.04, 1.},
        {c1_gj_cv[1], c2_gj_cv[1], 0.02, 1.},
        {c1_gj_cv[3], c2_gj_cv[2], 0.01, 1.},
        {c2_gj_cv[0], c0_gj_cv[1], 0.01, 1.},
        {c2_gj_cv[1], c1_gj_cv[1], 0.02, 1.},
        {c2_gj_cv[2], c1_gj_cv[3], 0.01, 1.}
    };
    util::sort(expected_gj_values);

    std::vector<int> expected_gj_cv, expected_gj_peer_cv;
    std::vector<double> expected_gj_weight, expected_gj_g;
    util::assign(expected_gj_cv,      util::transform_view(expected_gj_values, [](const auto& x){return std::get<0>(x);}));
    util::assign(expected_gj_peer_cv, util::transform_view(expected_gj_values, [](const auto& x){return std::get<1>(x);}));
    util::assign(expected_gj_weight,  util::transform_view(expected_gj_values, [](const auto& x){return std::get<2>(x);}));
    util::assign(expected_gj_g,       util::transform_view(expected_gj_values, [](const auto& x){return std::get<3>(x);}));

    EXPECT_EQ(10u, gj_data.cv.size());
    EXPECT_EQ(expected_gj_cv,      gj_data.cv);
    EXPECT_EQ(expected_gj_peer_cv, gj_data.peer_cv);
    EXPECT_EQ(expected_gj_weight,  gj_data.local_weight);
    EXPECT_EQ(expected_gj_g,       gj_g_param);
}

TEST(fvm_layout, gj_example_2) {
    auto context = make_context({arbenv::default_concurrency(), -1});

    class gap_recipe: public recipe {
    public:
        gap_recipe(std::vector<cable_cell> cells, arb::cable_cell_global_properties gprop) : gprop_(gprop), cells_(cells) {}

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return cells_[gid];
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            switch (gid) {
            case 0:
                return {};
            case 1:
                return {
                    gap_junction_connection({2, "j5"}, {"j2"}, 0.03),
                    gap_junction_connection({2, "j6"}, {"j3"}, 0.04),
                    gap_junction_connection({3, "j7"}, {"j0"}, 0.02),
                    gap_junction_connection({5, "j8"}, {"j1"}, 0.01),
                    gap_junction_connection({5, "j9"}, {"j4"}, 0.01)
                };
            case 2:
                return {
                    gap_junction_connection({1, "j2"}, {"j5"}, 0.03),
                    gap_junction_connection({1, "j3"}, {"j6"}, 0.04),
                    gap_junction_connection({3, "j7"}, {"j5"}, 0.02),
                    gap_junction_connection({5, "j9"}, {"j6"}, 0.01),
                };
            case 3:
                return {
                    gap_junction_connection({1, "j0"}, {"j7"}, 0.03),
                    gap_junction_connection({2, "j5"}, {"j7"}, 0.04),
                    gap_junction_connection({5, "j8"}, {"j7"}, 0.02),
                };
            case 4:
                return {};
            case 5:
                return {
                    gap_junction_connection({1, "j1"}, {"j8"}, 0.03),
                    gap_junction_connection({1, "j4"}, {"j9"}, 0.04),
                    gap_junction_connection({2, "j6"}, {"j9"}, 0.02),
                    gap_junction_connection({3, "j7"}, {"j8"}, 0.01),
                };
            default : return {};
            }
            return conns;
        }
        std::any get_global_properties(cell_kind) const override {
            return gprop_;
        }
    protected:
        arb::cable_cell_global_properties gprop_;
        std::vector<cable_cell> cells_;
        cell_size_type n_ = 6;
    };

    auto system = six_cell_gj_system();
    auto& desc = system.descriptions;
    auto& builders = system.builders;

    std::vector<mlocation> locs_1 = {
        builders[1].location({2, 0}),
        builders[1].location({3, 0.3}),
        builders[1].location({0, 0.5}),
        builders[1].location({0, 0.6}),
        builders[1].location({3, 0.8})};
    desc[1].decorations.place(locs_1[0], junction("gj1", {{"g", 0.3}, {"e", 1e-3}}), "j0");
    desc[1].decorations.place(locs_1[1], junction("gj0", {{"g", 0.2}}), "j1");
    desc[1].decorations.place(locs_1[2], junction("gj0", {{"g", 1.5}}), "j2");
    desc[1].decorations.place(locs_1[3], junction("gj0"), "j3");
    desc[1].decorations.place(locs_1[4], junction("gj0"), "j4");

    std::vector<mlocation> locs_2 = {
        builders[2].location({0, 0.5}),
        builders[2].location({1, 0})};
    desc[2].decorations.place(locs_2[0], junction("gj0", {{"g", 2.3}}), "j5");
    desc[2].decorations.place(locs_2[1], junction("gj1", {{"e", 5e-4}}), "j6");

    std::vector<mlocation> locs_3 = {
        builders[3].location({0, 0.5})};
    desc[3].decorations.place(locs_3[0], junction("gj1", {{"g", 0.4}}), "j7");

    std::vector<mlocation> locs_5 = {
        builders[5].location({0, 0.1}),
        builders[5].location({0, 0.2})};
    desc[5].decorations.place(locs_5[0], junction("gj1"), "j8");
    desc[5].decorations.place(locs_5[1], junction("gj0", {{"g", 1.6}}), "j9");

    // Check the GJ CV map
    cable_cell_global_properties gprop;
    gprop.catalogue = make_unit_test_catalogue();
    gprop.catalogue.import(arb::global_default_catalogue(), "");
    gprop.default_parameters = neuron_parameter_defaults;

    auto cells = system.cells();
    std::vector<cell_gid_type> gids = {0, 1, 2, 3, 4, 5};

    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);

    unsigned cvs_1[5], cvs_2[2], cvs_3[1], cvs_5[2];
    for (int i = 0; i<5; ++i) cvs_1[i] = D.geometry.location_cv(1, locs_1[i], cv_prefer::cv_nonempty);
    for (int i = 0; i<2; ++i) cvs_2[i] = D.geometry.location_cv(2, locs_2[i], cv_prefer::cv_nonempty);
    for (int i = 0; i<1; ++i) cvs_3[i] = D.geometry.location_cv(3, locs_3[i], cv_prefer::cv_nonempty);
    for (int i = 0; i<2; ++i) cvs_5[i] = D.geometry.location_cv(5, locs_5[i], cv_prefer::cv_nonempty);

    auto gj_cvs = fvm_build_gap_junction_cv_map(cells, gids, D);

    EXPECT_EQ(10u, gj_cvs.size());
    EXPECT_EQ(cvs_1[0], gj_cvs.at(cell_member_type{1, 0}));
    EXPECT_EQ(cvs_1[1], gj_cvs.at(cell_member_type{1, 1}));
    EXPECT_EQ(cvs_1[2], gj_cvs.at(cell_member_type{1, 2}));
    EXPECT_EQ(cvs_1[3], gj_cvs.at(cell_member_type{1, 3}));
    EXPECT_EQ(cvs_1[4], gj_cvs.at(cell_member_type{1, 4}));
    EXPECT_EQ(cvs_2[0], gj_cvs.at(cell_member_type{2, 0}));
    EXPECT_EQ(cvs_2[1], gj_cvs.at(cell_member_type{2, 1}));
    EXPECT_EQ(cvs_3[0], gj_cvs.at(cell_member_type{3, 0}));
    EXPECT_EQ(cvs_5[0], gj_cvs.at(cell_member_type{5, 0}));
    EXPECT_EQ(cvs_5[1], gj_cvs.at(cell_member_type{5, 1}));

    // Check the resolved GJ connections
    fvm_cell fvcell(*context);
    gap_recipe rec(cells, gprop);

    auto fvm_info = fvcell.initialize(gids, rec);
    auto gj_conns = fvm_resolve_gj_connections(gids, fvm_info.gap_junction_data, gj_cvs, rec);

    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> expected;
    expected[0] = {};
    expected[1] = {
        {2, cvs_1[2], cvs_2[0], 0.03},
        {3, cvs_1[3], cvs_2[1], 0.04},
        {0, cvs_1[0], cvs_3[0], 0.02},
        {1, cvs_1[1], cvs_5[0], 0.01},
        {4, cvs_1[4], cvs_5[1], 0.01},
    };
    expected[2] = {
        {0, cvs_2[0], cvs_1[2], 0.03},
        {1, cvs_2[1], cvs_1[3], 0.04},
        {0, cvs_2[0], cvs_3[0], 0.02},
        {1, cvs_2[1], cvs_5[1], 0.01}
    };
    expected[3] = {
        {0, cvs_3[0], cvs_1[0], 0.03},
        {0, cvs_3[0], cvs_2[0], 0.04},
        {0, cvs_3[0], cvs_5[0], 0.02}
    };
    expected[4] = {};
    expected[5] = {
        {0, cvs_5[0], cvs_1[1], 0.03},
        {1, cvs_5[1], cvs_1[4], 0.04},
        {1, cvs_5[1], cvs_2[1], 0.02},
        {0, cvs_5[0], cvs_3[0], 0.01}
    };

    util::sort(expected.at(1));
    util::sort(expected.at(2));
    util::sort(expected.at(3));
    util::sort(expected.at(5));

    EXPECT_EQ(expected.at(0), gj_conns.at(0));
    EXPECT_EQ(expected.at(1), gj_conns.at(1));
    EXPECT_EQ(expected.at(2), gj_conns.at(2));
    EXPECT_EQ(expected.at(3), gj_conns.at(3));
    EXPECT_EQ(expected.at(4), gj_conns.at(4));
    EXPECT_EQ(expected.at(5), gj_conns.at(5));

    // Check the GJ mechanism data
    auto M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D, *context);

    EXPECT_EQ(4u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("gj0"));
    ASSERT_EQ(1u, M.mechanisms.count("gj1"));

    const auto& gj0_data = M.mechanisms["gj0"];
    const auto& gj1_data = M.mechanisms["gj1"];

    const auto& gj0_params = gj0_data.param_values;
    const auto& gj1_params = gj1_data.param_values;

    const auto& gj0_g_param = *ptr_by_key(gj0_params, "g"s);
    const auto& gj1_g_param = *ptr_by_key(gj1_params, "g"s);
    const auto& gj1_e_param = *ptr_by_key(gj1_params, "e"s);

    std::vector<std::tuple<int, int, double, double>> expected_gj0_values = {
        {cvs_1[2], cvs_2[0], 0.03, 1.5},
        {cvs_1[3], cvs_2[1], 0.04, 1. },
        {cvs_1[1], cvs_5[0], 0.01, 0.2},
        {cvs_1[4], cvs_5[1], 0.01, 1.0},
        {cvs_2[0], cvs_1[2], 0.03, 2.3},
        {cvs_2[0], cvs_3[0], 0.02, 2.3},
        {cvs_5[1], cvs_1[4], 0.04, 1.6},
        {cvs_5[1], cvs_2[1], 0.02, 1.6}
    };
    util::sort(expected_gj0_values);

    std::vector<int> expected_gj0_cv, expected_gj0_peer_cv;
    std::vector<double> expected_gj0_weight, expected_gj0_g;
    util::assign(expected_gj0_cv,      util::transform_view(expected_gj0_values, [](const auto& x){return std::get<0>(x);}));
    util::assign(expected_gj0_peer_cv, util::transform_view(expected_gj0_values, [](const auto& x){return std::get<1>(x);}));
    util::assign(expected_gj0_weight,  util::transform_view(expected_gj0_values, [](const auto& x){return std::get<2>(x);}));
    util::assign(expected_gj0_g,       util::transform_view(expected_gj0_values, [](const auto& x){return std::get<3>(x);}));

    EXPECT_EQ(8u, gj0_data.cv.size());
    EXPECT_EQ(expected_gj0_cv,      gj0_data.cv);
    EXPECT_EQ(expected_gj0_peer_cv, gj0_data.peer_cv);
    EXPECT_EQ(expected_gj0_weight,  gj0_data.local_weight);
    EXPECT_EQ(expected_gj0_g,       gj0_g_param);

    std::vector<std::tuple<int, int, double, double, double>> expected_gj1_values = {
        {cvs_1[0], cvs_3[0], 0.02, 0.3, 1e-3},
        {cvs_2[1], cvs_1[3], 0.04, 1.0, 5e-4},
        {cvs_2[1], cvs_5[1], 0.01, 1.0, 5e-4},
        {cvs_3[0], cvs_1[0], 0.03, 0.4, 0.},
        {cvs_3[0], cvs_2[0], 0.04, 0.4, 0.},
        {cvs_3[0], cvs_5[0], 0.02, 0.4, 0.},
        {cvs_5[0], cvs_1[1], 0.03, 1.0, 0.},
        {cvs_5[0], cvs_3[0], 0.01, 1.0, 0.}
    };
    util::sort(expected_gj1_values);

    std::vector<int> expected_gj1_cv, expected_gj1_peer_cv;
    std::vector<double> expected_gj1_weight, expected_gj1_g, expected_gj1_e;
    util::assign(expected_gj1_cv,      util::transform_view(expected_gj1_values, [](const auto& x){return std::get<0>(x);}));
    util::assign(expected_gj1_peer_cv, util::transform_view(expected_gj1_values, [](const auto& x){return std::get<1>(x);}));
    util::assign(expected_gj1_weight,  util::transform_view(expected_gj1_values, [](const auto& x){return std::get<2>(x);}));
    util::assign(expected_gj1_g,       util::transform_view(expected_gj1_values, [](const auto& x){return std::get<3>(x);}));
    util::assign(expected_gj1_e,       util::transform_view(expected_gj1_values, [](const auto& x){return std::get<4>(x);}));

    EXPECT_EQ(8u, gj1_data.cv.size());
    EXPECT_EQ(expected_gj1_cv,      gj1_data.cv);
    EXPECT_EQ(expected_gj1_peer_cv, gj1_data.peer_cv);
    EXPECT_EQ(expected_gj1_weight,  gj1_data.local_weight);
    EXPECT_EQ(expected_gj1_g,       gj1_g_param);
    EXPECT_EQ(expected_gj1_e,       gj1_e_param);
}

TEST(fvm_lowered, cell_group_gj) {
    auto context = make_context({arbenv::default_concurrency(), -1});

    class gap_recipe: public recipe {
    public:
        gap_recipe(const std::vector<cable_cell>& cg0, const std::vector<cable_cell>& cg1) {
            cells_ = cg0;
            cells_.insert(cells_.end(), cg1.begin(), cg1.end());
            gprop_.default_parameters = neuron_parameter_defaults;
        }

        cell_size_type num_cells() const override { return n_; }
        cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::cable; }
        util::unique_any get_cell_description(cell_gid_type gid) const override {
            return cells_[gid];
        }
        std::vector<arb::gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override{
            std::vector<gap_junction_connection> conns;
            if (gid % 2 == 0) {
                // connect 5 of the first 10 cells in a ring; connect 5 of the second 10 cells in a ring
                auto next_cell = gid == 8 ? 0 : (gid == 18 ? 10 : gid + 2);
                auto prev_cell = gid == 0 ? 8 : (gid == 10 ? 18 : gid - 2);
                conns.push_back(gap_junction_connection({next_cell, "gj", lid_selection_policy::assert_univalent},
                                                        {"gj", lid_selection_policy::assert_univalent}, 0.03));
                conns.push_back(gap_junction_connection({prev_cell, "gj", lid_selection_policy::assert_univalent},
                                                        {"gj", lid_selection_policy::assert_univalent}, 0.03));
            }
            return conns;
        }
        std::any get_global_properties(cell_kind) const override {
            return gprop_;
        }

    protected:
        arb::cable_cell_global_properties gprop_;
        std::vector<cable_cell> cells_;
        cell_size_type n_ = 20;
    };

    std::vector<cable_cell> cell_group0;
    std::vector<cable_cell> cell_group1;

    // Make 20 cells
    for (unsigned i = 0; i < 20; i++) {
        cable_cell_description c = soma_cell_builder(2.1).make_cell();
        if (i % 2 == 0) {
            c.decorations.place(mlocation{0, 1}, junction{"gj"}, "gj");
        }
        if (i < 10) {
            cell_group0.push_back(c);
        }
        else {
            cell_group1.push_back(c);
        }
    }

    std::vector<cell_gid_type> gids_cg0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<cell_gid_type> gids_cg1 = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    gap_recipe rec(cell_group0, cell_group1);

    fvm_cell fvcell0(*context);
    fvm_cell fvcell1(*context);

    auto fvm_info_0 = fvcell0.initialize(gids_cg0, rec);
    auto fvm_info_1 = fvcell1.initialize(gids_cg1, rec);

    auto num_dom0 = fvcell0.fvm_intdom(rec, gids_cg0, fvm_info_0.cell_to_intdom);
    auto num_dom1 = fvcell1.fvm_intdom(rec, gids_cg1, fvm_info_1.cell_to_intdom);

    fvm_cv_discretization D0 = fvm_cv_discretize(cell_group0, neuron_parameter_defaults, *context);
    fvm_cv_discretization D1 = fvm_cv_discretize(cell_group1, neuron_parameter_defaults, *context);

    auto gj_cvs_0 = fvm_build_gap_junction_cv_map(cell_group0, gids_cg0, D0);
    auto gj_cvs_1 = fvm_build_gap_junction_cv_map(cell_group1, gids_cg1, D1);

    auto GJ0 = fvm_resolve_gj_connections(gids_cg0, fvm_info_0.gap_junction_data, gj_cvs_0, rec);
    auto GJ1 = fvm_resolve_gj_connections(gids_cg1, fvm_info_1.gap_junction_data, gj_cvs_1, rec);

    EXPECT_EQ(gids_cg0.size(), GJ0.size());
    EXPECT_EQ(gids_cg1.size(), GJ1.size());

    std::vector<std::vector<fvm_gap_junction>> expected = {
        {{0, 0, 2, 0.03}, {0, 0, 8, 0.03}}, {},
        {{0, 2, 0, 0.03}, {0, 2, 4, 0.03}}, {},
        {{0, 4, 2, 0.03} ,{0, 4, 6, 0.03}}, {},
        {{0, 6, 4, 0.03}, {0, 6, 8, 0.03}}, {},
        {{0, 8, 0, 0.03}, {0, 8, 6, 0.03}}, {}
    };

    for (unsigned i = 0; i < GJ0.size(); i++) {
        EXPECT_EQ(expected[i], GJ0[i]);
        EXPECT_EQ(expected[i], GJ1[i+10]);
    }

    std::vector<arb_index_type> expected_doms= {0u, 1u, 0u, 2u, 0u, 3u, 0u, 4u, 0u, 5u};
    EXPECT_EQ(6u, num_dom0);
    EXPECT_EQ(6u, num_dom1);

    EXPECT_EQ(expected_doms, fvm_info_0.cell_to_intdom);
    EXPECT_EQ(expected_doms, fvm_info_1.cell_to_intdom);
}

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

    // Create a cell with 4 branches:
    //   - Soma (branch 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - HH mechanism on all branches.
    //   - Discretize with 3 CVs per non-soma branch, centred on forks.
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/6 of branch 1 and the initial 1/6 of branches 2 and 3.
    //
    // The HH mechanism current density parameters ('gnabar', 'gkbar' and 'gl') are set
    // differently for each branch:
    //
    //   soma:      all default values (gnabar = 0.12, gkbar = .036, gl = .0003)
    //   branch 1: gl = .0002
    //   branch 2: gkbar = .05
    //   branch 3: gkbar = .07, gl = .0004
    //
    // Geometry:
    //   branch 1: 100 µm long, 1 µm diameter cylinder.
    //   branch 2: 200 µm long, diameter linear taper from 1 µm to 0.2 µm.
    //   branch 3: 150 µm long, 0.8 µm diameter cylinder.

    soma_cell_builder builder(12.6157/2.0);

    //                 p  len   r1   r2  ncomp tag
    builder.add_branch(0, 100, 0.5, 0.5,     3, "reg1");
    builder.add_branch(1, 200, 0.5, 0.1,     3, "reg2");
    builder.add_branch(1, 150, 0.4, 0.4,     3, "reg3");

    double dflt_gkbar = .036;
    double dflt_gl = 0.0003;

    double seg1_gl = .0002;
    double seg2_gkbar = .05;
    double seg3_gkbar = .0004;
    double seg3_gl = .0004;

    auto hh_0 = mechanism_desc("hh");

    auto hh_1 = mechanism_desc("hh");
    hh_1["gl"] = seg1_gl;

    auto hh_2 = mechanism_desc("hh");
    hh_2["gkbar"] = seg2_gkbar;

    auto hh_3 = mechanism_desc("hh");
    hh_3["gkbar"] = seg3_gkbar;
    hh_3["gl"] = seg3_gl;

    auto desc = builder.make_cell();
    desc.decorations.paint("soma"_lab, density(std::move(hh_0)));
    desc.decorations.paint("reg1"_lab, density(std::move(hh_1)));
    desc.decorations.paint("reg2"_lab, density(std::move(hh_2)));
    desc.decorations.paint("reg3"_lab, density(std::move(hh_3)));

    std::vector<cable_cell> cells{desc};

    int ncv = 11;
    std::vector<double> expected_gkbar(ncv, dflt_gkbar);
    std::vector<double> expected_gl(ncv, dflt_gl);

    // Last 1/6 of branch 1
    double seg1_area_right = cells[0].embedding().integrate_area(builder.cable({1, 5./6., 1.}));
    // First 1/6 of branch 2
    double seg2_area_left = cells[0].embedding().integrate_area(builder.cable({2, 0., 1./6.}));
    // First 1/6 of branch 3
    double seg3_area_left = cells[0].embedding().integrate_area(builder.cable({3, 0., 1./6.}));

    // CV 0: soma
    // CV1: left of branch 1
    expected_gl[0] = dflt_gl;
    expected_gl[1] = seg1_gl;

    expected_gl[2] = seg1_gl;
    expected_gl[3] = seg1_gl;

    // CV 4: mix of right of branch 1 and left of branches 2 and 3.
    expected_gkbar[4] = wmean(seg1_area_right, dflt_gkbar, seg2_area_left, seg2_gkbar, seg3_area_left, seg3_gkbar);
    expected_gl[4] = wmean(seg1_area_right, seg1_gl, seg2_area_left, dflt_gl, seg3_area_left, seg3_gl);

    // CV 5-7: just branch 2
    expected_gkbar[5] = seg2_gkbar;
    expected_gkbar[6] = seg2_gkbar;
    expected_gkbar[7] = seg2_gkbar;

    // CV 8-10: just branch 3
    expected_gkbar[8] = seg3_gkbar;
    expected_gkbar[9] = seg3_gkbar;
    expected_gkbar[10] = seg3_gkbar;
    expected_gl[8] = seg3_gl;
    expected_gl[9] = seg3_gl;
    expected_gl[10] = seg3_gl;

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    std::vector<cell_gid_type> gids = {0};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}};
    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
    fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

    // Grab the HH parameters from the mechanism.

    EXPECT_EQ(1u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("hh"));
    auto& hh_params = M.mechanisms.at("hh").param_values;

    auto& gkbar = *ptr_by_key(hh_params, "gkbar"s);
    auto& gl = *ptr_by_key(hh_params, "gl"s);

    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gkbar, gkbar));
    EXPECT_TRUE(testing::seq_almost_eq<double>(expected_gl, gl));
}

TEST(fvm_layout, density_norm_area_partial) {
    // Test area-weighted linear combination of density mechanism parameters,
    // when mechanism covers only part of CV.

    // Create a cell with 2 unbranched cables:
    //   - Soma (branch 0) plus one constant-diameter dendrite.
    //   - HH mechanism on part of the dendrite.
    //   - Discretize with 1 CV per branch.
    //
    // The HH mechanism is applied to the first 30% and last 60% of the dendrite:
    //
    //   first 30%:  all default values (gnabar = 0.12, gkbar = .036, gl = .0003)
    //   last 60%:   gl = .0002, gkbar = .05
    //
    // Geometry:
    //   dendrite: 200 µm long, diameter linear taper from 1 µm to 0.2 µm.

    soma_cell_builder builder(12.6157/2.0);

    //                 p  len   r1   r2  ncomp tag
    builder.add_branch(0, 200, 0.5, 0.1,     1, "dend");

    double dflt_gnabar = .12;
    double dflt_gkbar = .036;
    double dflt_gl = 0.0003;

    double end_gl = .0002;
    double end_gkbar = .05;

    auto hh_begin = mechanism_desc("hh");

    auto hh_end = mechanism_desc("hh");
    hh_end["gl"] = end_gl;
    hh_end["gkbar"] = end_gkbar;

    auto desc = builder.make_cell();
    desc.decorations.set_default(cv_policy_fixed_per_branch(1));

    desc.decorations.paint(builder.cable({1, 0., 0.3}), density(hh_begin));
    desc.decorations.paint(builder.cable({1, 0.4, 1.}), density(hh_end));

    std::vector<cable_cell> cells{desc};

    // Area of whole cell (which is area of the 1 branch)
    double area = cells[0].embedding().integrate_area({0, 0., 1});
    // First 30% of branch 1.
    double b1_area_begin = cells[0].embedding().integrate_area(builder.cable({1, 0., 0.3}));
    // Last 60% of branch 1.
    double b1_area_end = cells[0].embedding().integrate_area(builder.cable({1, 0.4, 1.}));

    double expected_norm_area = (b1_area_begin+b1_area_end)/area;
    double expected_gnabar = dflt_gnabar;
    double expected_gkbar = (dflt_gkbar*b1_area_begin + end_gkbar*b1_area_end)/(b1_area_begin + b1_area_end);
    double expected_gl = (dflt_gl*b1_area_begin + end_gl*b1_area_end)/(b1_area_begin + b1_area_end);

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    std::vector<cell_gid_type> gids = {0};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}};
    fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
    fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

    // Grab the HH parameters from the mechanism.

    EXPECT_EQ(1u, M.mechanisms.size());
    ASSERT_EQ(1u, M.mechanisms.count("hh"));

    auto& norm_area = M.mechanisms.at("hh").norm_area;
    ASSERT_EQ(1u, norm_area.size());
    EXPECT_DOUBLE_EQ(expected_norm_area, norm_area[0]);

    auto& hh_params = M.mechanisms.at("hh").param_values;

    auto& gkbar = *ptr_by_key(hh_params, "gkbar"s);
    auto& gnabar = *ptr_by_key(hh_params, "gnabar"s);
    auto& gl = *ptr_by_key(hh_params, "gl"s);

    ASSERT_EQ(1u, gkbar.size());
    ASSERT_EQ(1u, gnabar.size());
    ASSERT_EQ(1u, gl.size());

    EXPECT_DOUBLE_EQ(expected_gkbar, gkbar[0]);
    EXPECT_DOUBLE_EQ(expected_gnabar, gnabar[0]);
    EXPECT_DOUBLE_EQ(expected_gl, gl[0]);
}

TEST(fvm_layout, valence_verify) {
    auto desc = soma_cell_builder(6).make_cell();
    desc.decorations.paint("soma"_lab, density("test_cl_valence"));
    std::vector<cable_cell> cells{desc};
    std::vector<cell_gid_type> gids = {0};
    std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}};

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;

    fvm_cv_discretization D = fvm_cv_discretize(cells, neuron_parameter_defaults);

    gprop.catalogue = make_unit_test_catalogue();

    // Missing the 'cl' ion:
    EXPECT_THROW(fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D), cable_cell_error);

    // Adding ion, should be fine now:
    gprop.default_parameters.ion_data["cl"] = { 1., 1., 0., 0. };
    gprop.ion_species["cl"] = -1;
    EXPECT_NO_THROW(fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D));

    // 'cl' ion has wrong charge:
    gprop.ion_species["cl"] = -2;
    EXPECT_THROW(fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D), cable_cell_error);
}

TEST(fvm_layout, ion_weights) {
    // Create a cell with 4 branches:
    //   - Soma (branch 0) plus three dendrites (1, 2, 3) meeting at a branch point.
    //   - Dendritic branches are given 1 compartments each.
    //
    //         /
    //        d2
    //       /
    //   s0-d1
    //       \.
    //        d3
    //
    // The CV corresponding to the branch point should comprise the terminal
    // 1/2 of branch 1 and the initial 1/2 of branches 2 and 3.
    //
    // Geometry:
    //   soma 0: radius 5 µm, area 100π μm²
    //   dend 1: 100 µm long, 1 µm diameter cylinder, area 100π μm²
    //   dend 2: 200 µm long, 1 µm diameter cylinder, area 200π μm²
    //   dend 3: 100 µm long, 1 µm diameter cylinder, area 100π μm²
    //
    // The radius of the soma is chosen such that the surface area of soma is
    // the same as a 100µm dendrite, which makes it easier to describe the
    // expected weights.

    soma_cell_builder builder(5);
    builder.add_branch(0, 100, 0.5, 0.5, 1, "dend");
    builder.add_branch(1, 200, 0.5, 0.5, 1, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 1, "dend");

    using uvec = std::vector<arb_size_type>;
    using ivec = std::vector<arb_index_type>;
    using fvec = std::vector<arb_value_type>;

    //uvec mech_branches[] = {
        //{0}, {0,2}, {2, 3}, {0, 1, 2, 3}, {3}
    //};
    uvec mech_branches[] = {
        {0}, {0,2}
    };

    ivec expected_ion_cv[] = {
        {0}, {0, 2, 3}, {2, 3, 4}, {0, 1, 2, 3, 4}, {2, 4}
    };

    fvec expected_init_iconc[] = {
        {0.}, {0., 1./2, 0.}, {1./4, 0., 0.}, {0., 0., 0., 0., 0.}, {3./4, 0.}
    };

    cable_cell_global_properties gprop;
    gprop.catalogue = make_unit_test_catalogue();
    gprop.default_parameters = neuron_parameter_defaults;

    arb_value_type cai = gprop.default_parameters.ion_data["ca"].init_int_concentration.value();
    arb_value_type cao = gprop.default_parameters.ion_data["ca"].init_ext_concentration.value();

    for (auto& v: expected_init_iconc) {
        for (auto& iconc: v) {
            iconc *= cai;
        }
    }

    for (auto run: count_along(mech_branches)) {
        SCOPED_TRACE("run "+std::to_string(run));
        auto desc = builder.make_cell();

        for (auto i: mech_branches[run]) {
            auto cab = builder.cable({i, 0, 1});
            desc.decorations.paint(reg::cable(cab.branch, cab.prox_pos, cab.dist_pos), density("test_ca"));
        }

        std::vector<cable_cell> cells{desc};
        std::vector<cell_gid_type> gids = {0};
        std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}};
        fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

        ASSERT_EQ(1u, M.ions.count("ca"s));
        auto& ca = M.ions.at("ca"s);

        EXPECT_EQ(expected_ion_cv[run], ca.cv);

        EXPECT_TRUE(testing::seq_almost_eq<arb_value_type>(expected_init_iconc[run], ca.init_iconc));

        EXPECT_TRUE(util::all_of(ca.init_econc, [cao](arb_value_type v) { return v==cao; }));
    }
}

TEST(fvm_layout, revpot) {
    // Create two cells with three ions 'a', 'b' and 'c'.
    // Configure a reversal potential mechanism that writes to 'a' and
    // another that writes to 'b' and 'c'.
    //
    // Confirm:
    //     * Inconsistencies between revpot mech assignments are caught at discretization.
    //     * Reversal potential mechanisms are only extended where there exists another
    //       mechanism that reads them.

    soma_cell_builder builder(5);
    builder.add_branch(0, 100, 0.5, 0.5, 1, "dend");
    builder.add_branch(1, 200, 0.5, 0.5, 1, "dend");
    builder.add_branch(1, 100, 0.5, 0.5, 1, "dend");
    auto desc = builder.make_cell();
    desc.decorations.paint("soma"_lab, density("read_eX/c"));
    desc.decorations.paint("soma"_lab, density("read_eX/a"));
    desc.decorations.paint("dend"_lab, density("read_eX/a"));

    std::vector<cable_cell_description> descriptions{desc, desc};

    cable_cell_global_properties gprop;
    gprop.default_parameters = neuron_parameter_defaults;
    gprop.catalogue = make_unit_test_catalogue();

    gprop.ion_species = {{"a", 1}, {"b", 2}, {"c", 3}};
    gprop.add_ion("a", 1, 10., 0, 0);
    gprop.add_ion("b", 2, 30., 0, 0);
    gprop.add_ion("c", 3, 50., 0, 0);

    gprop.default_parameters.reversal_potential_method["a"] = "write_eX/a";
    mechanism_desc write_eb_ec = "write_multiple_eX/x=b,y=c";

    {
        // need to specify ion "c" as well.
        auto test_gprop = gprop;
        test_gprop.default_parameters.reversal_potential_method["b"] = write_eb_ec;

        std::vector<cable_cell> cells{descriptions[0], descriptions[1]};
        std::vector<cell_gid_type> gids = {0,1};
        std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
        fvm_cv_discretization D = fvm_cv_discretize(cells, test_gprop.default_parameters);
        EXPECT_THROW(fvm_build_mechanism_data(test_gprop, cells, gids, gj_conns, D), cable_cell_error);
    }

    {
        // conflict with ion "c" on second cell.
        auto test_gprop = gprop;
        test_gprop.default_parameters.reversal_potential_method["b"] = write_eb_ec;
        test_gprop.default_parameters.reversal_potential_method["c"] = write_eb_ec;
        descriptions[1].decorations.set_default(ion_reversal_potential_method{"c", "write_eX/c"});

        std::vector<cable_cell> cells{descriptions[0], descriptions[1]};
        std::vector<cell_gid_type> gids = {0,1};
        std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
        fvm_cv_discretization D = fvm_cv_discretize(cells, test_gprop.default_parameters);
        EXPECT_THROW(fvm_build_mechanism_data(test_gprop, cells, gids, gj_conns, D), cable_cell_error);
    }

    {
        auto& cell1_prop = const_cast<cable_cell_parameter_set&>(descriptions[1].decorations.defaults());
        cell1_prop.reversal_potential_method.clear();
        descriptions[1].decorations.set_default(ion_reversal_potential_method{"b", write_eb_ec});
        descriptions[1].decorations.set_default(ion_reversal_potential_method{"c", write_eb_ec});

        std::vector<cable_cell> cells{descriptions[0], descriptions[1]};
        std::vector<cell_gid_type> gids = {0,1};
        std::unordered_map<cell_gid_type, std::vector<fvm_gap_junction>> gj_conns = {{0, {}}, {1, {}}};
        fvm_cv_discretization D = fvm_cv_discretize(cells, gprop.default_parameters);
        fvm_mechanism_data M = fvm_build_mechanism_data(gprop, cells, gids, gj_conns, D);

        // Only CV which needs write_multiple_eX/x=b,y=c is the soma (first CV)
        // of the second cell.
        auto soma1_index = D.geometry.cell_cv_divs[1];
        ASSERT_EQ(1u, M.mechanisms.count(write_eb_ec.name()));
        EXPECT_EQ((std::vector<arb_index_type>(1, soma1_index)), M.mechanisms.at(write_eb_ec.name()).cv);
    }
}

TEST(fvm_layout, vinterp_cable) {
    // On a simple cable, expect CVs used forinterpolation to change at
    // the midpoints of interior CVs. Every site in the proximal CV should
    // interpolate between that and the next; every site in the distal CV
    // should interpolate between that and the parent.

    // Cable cell with just one branch, non-spherical root.
    arb::segment_tree tree;
    tree.append(mnpos, { 0,0,0,1}, {10,0,0,1}, 1);
    arb::morphology m(tree);
    decor d;

    // CV midpoints at branch pos 0.1, 0.3, 0.5, 0.7, 0.9.
    // Expect voltage reference locations to be CV modpoints.
    d.set_default(cv_policy_fixed_per_branch(5));
    cable_cell cell{m, d};
    fvm_cv_discretization D = fvm_cv_discretize(cell, neuron_parameter_defaults);

    // Test locations, either side of CV midpoints plus extrema, CV boundaries.
    double site_pos[] = { 0., 0.03, 0.11, 0.2, 0.28, 0.33, 0.4, 0.46, 0.55, 0.6, 0.75, 0.8, 0.83, 0.95, 1.};

    for (auto pos: site_pos) {
        mlocation site{0, pos};

        arb_index_type expected_distal;
        if (pos<0.3) {
            expected_distal = 1;
        }
        else if (pos<0.5) {
            expected_distal = 2;
        }
        else if (pos<0.7) {
            expected_distal = 3;
        }
        else {
            expected_distal = 4;
        }
        arb_index_type expected_proximal = expected_distal-1;

        fvm_voltage_interpolant I = fvm_interpolate_voltage(cell, D, 0, site);

        EXPECT_EQ(expected_proximal, I.proximal_cv);
        EXPECT_EQ(expected_distal, I.distal_cv);

        // Cable has constant diameter, so interpolation coefficients should
        // be simple linear functions of branch position.

        double prox_refpos = I.proximal_cv*0.2+0.1;
        double dist_refpos = I.distal_cv*0.2+0.1;

        // (Tortuous fp manipulation along the way makes the error greater than 4 ulp).
        const double relerr = 32*std::numeric_limits<double>::epsilon();

        EXPECT_TRUE(testing::near_relative((dist_refpos-pos)/0.2, I.proximal_coef, relerr));
        EXPECT_TRUE(testing::near_relative((pos-prox_refpos)/0.2, I.distal_coef, relerr));
    }
}

TEST(fvm_layout, vinterp_forked) {
    // If a CV contains points at both ends of a branch, there will be
    // no other adjacent CV on the same branch that we can use for
    // interpolation.

    // Cable cell with three branchses; branches 0 has child branches 1 and 2.
    segment_tree tree;
    tree.append(mnpos, {0., 0., 0., 1.}, {10., 0., 0., 1}, 1);
    tree.append(    0, {10., 20., 0., 1}, 1);
    tree.append(    0, {10.,-20., 0., 1}, 1);
    morphology m(tree);
    decor d;

    // CV 0 contains branch 0 and the fork point; CV 1 and CV 2 have CV 0 as parent,
    // and contain branches 1 and 2 respectively, excluding the fork point.
    mlocation_list cv_ends{{1, 0.}, {2, 0.}};
    d.set_default(cv_policy_explicit(cv_ends));
    cable_cell cell{m, d};
    fvm_cv_discretization D = fvm_cv_discretize(cell, neuron_parameter_defaults);

    // Points in branch 0 should only get CV 0 for interpolation.
    {
        fvm_voltage_interpolant I = fvm_interpolate_voltage(cell, D, 0, mlocation{0, 0.3});
        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(0, I.distal_cv);
        EXPECT_EQ(1, I.proximal_coef+I.distal_coef);
    }
    // Points in branches 1 and 2 should get CV 0 and CV 1 or 2 respectively.
    {
        fvm_voltage_interpolant I = fvm_interpolate_voltage(cell, D, 0, mlocation{1, 0});
        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(1., I.proximal_coef);
        EXPECT_EQ(1, I.distal_cv);
        EXPECT_EQ(0., I.distal_coef);

        // Past the midpoint, we're extrapolating.
        I = fvm_interpolate_voltage(cell, D, 0, mlocation{1, 0.7});
        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_LT(I.proximal_coef, 0.);
        EXPECT_EQ(1, I.distal_cv);
        EXPECT_GT(I.distal_coef, 1.);

        I = fvm_interpolate_voltage(cell, D, 0, mlocation{2, 0});
        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(1., I.proximal_coef);
        EXPECT_EQ(2, I.distal_cv);
        EXPECT_EQ(0., I.distal_coef);

        I = fvm_interpolate_voltage(cell, D, 0, mlocation{2, 0.7});
        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_LT(I.proximal_coef, 0.);
        EXPECT_EQ(2, I.distal_cv);
        EXPECT_GT(I.distal_coef, 1.);
    }
}

TEST(fvm_layout, iinterp) {
    // If we get two distinct interpolation points back, the coefficients
    // should match the face-conductance.

    // 1. Vertex-delimited and vertex-centred discretizations.
    using namespace common_morphology;

    std::vector<cable_cell> cells;
    std::vector<std::string> label;
    for (auto& p: test_morphologies) {
        if (p.second.empty()) continue;
        decor d;

        d.set_default(cv_policy_fixed_per_branch(3));
        cells.emplace_back(cable_cell{p.second, d});
        label.push_back(p.first+": forks-at-end"s);

        d.set_default(cv_policy_fixed_per_branch(3, cv_policy_flag::interior_forks));
        cells.emplace_back(cable_cell{p.second, d});
        label.push_back(p.first+": interior-forks"s);
    }

    fvm_cv_discretization D = fvm_cv_discretize(cells, neuron_parameter_defaults);
    for (unsigned cell_idx = 0; cell_idx<cells.size(); ++cell_idx) {
        SCOPED_TRACE(label[cell_idx]);
        unsigned n_branch = D.geometry.n_branch(cell_idx);
        for (msize_t bid = 0; bid<n_branch; ++bid) {
            for (double pos: {0., 0.3, 0.4, 0.7, 1.}) {
                mlocation x{bid, pos};
                SCOPED_TRACE(x);

                fvm_voltage_interpolant I = fvm_axial_current(cells[cell_idx], D, cell_idx, x);

                // With the given discretization policies, should only have no interpolation when
                // the cell has only the once CV.

                if (D.geometry.cell_cvs(cell_idx).size()==1) {
                    EXPECT_EQ(I.proximal_cv, I.distal_cv);
                    EXPECT_EQ(D.geometry.cell_cvs(cell_idx).front(), I.proximal_cv);
                }
                else {
                    EXPECT_EQ(D.geometry.cv_parent.at(I.distal_cv), I.proximal_cv);
                    EXPECT_TRUE(I.proximal_cv>=D.geometry.cell_cv_interval(cell_idx).first);

                    double fc = D.face_conductance.at(I.distal_cv);
                    EXPECT_DOUBLE_EQ(+fc, I.proximal_coef);
                    EXPECT_DOUBLE_EQ(-fc, I.distal_coef);
                }
            }
        }
    }

    // 2. Weird discretization: test points where the interpolated current has to be zero.
    // Use the same cell/discretiazation as in vinterp_forked test:

    // Cable cell with three branches; branch 0 has child branches 1 and 2.
    segment_tree tree;
    tree.append(mnpos, {0., 0., 0., 1.}, {10., 0., 0., 1}, 1);
    tree.append(    0, {10., 20., 0., 1}, 1);
    tree.append(    0, {10.,-20., 0., 1}, 1);
    morphology m(tree);
    decor d;

    // CV 0 contains branch 0 and the fork point; CV 1 and CV 2 have CV 0 as parent,
    // and contain branches 1 and 2 respectively, excluding the fork point.
    mlocation_list cv_ends{{1, 0.}, {2, 0.}};
    d.set_default(cv_policy_explicit(cv_ends));
    cable_cell cell{m, d};
    D = fvm_cv_discretize(cell, neuron_parameter_defaults);

    // Expect axial current interpolations on branches 1 and 2 to match CV 1 and 2
    // face-conductances; CV 0 contains the fork point, so there is nothing to
    // interpolate from on branch 0.

    // Branch 0:
    for (double pos: {0., 0.1, 0.8, 1.}) {
        mlocation x{0, pos};
        SCOPED_TRACE(x);

        fvm_voltage_interpolant I = fvm_axial_current(cell, D, 0, x);

        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(0, I.distal_cv);
        EXPECT_EQ(0., I.proximal_coef);
        EXPECT_EQ(0., I.distal_coef);
    }

    // Branch 1:
    double fc1 = D.face_conductance[1];
    for (double pos: {0., 0.1, 0.8, 1.}) {
        mlocation x{1, pos};
        SCOPED_TRACE(x);

        fvm_voltage_interpolant I = fvm_axial_current(cell, D, 0, x);

        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(1, I.distal_cv);
        EXPECT_EQ(+fc1, I.proximal_coef);
        EXPECT_EQ(-fc1, I.distal_coef);
    }

    // Branch 2:
    double fc2 = D.face_conductance[2];
    for (double pos: {0., 0.1, 0.8, 1.}) {
        mlocation x{2, pos};
        SCOPED_TRACE(x);

        fvm_voltage_interpolant I = fvm_axial_current(cell, D, 0, x);

        EXPECT_EQ(0, I.proximal_cv);
        EXPECT_EQ(2, I.distal_cv);
        EXPECT_EQ(+fc2, I.proximal_coef);
        EXPECT_EQ(-fc2, I.distal_coef);
    }
}

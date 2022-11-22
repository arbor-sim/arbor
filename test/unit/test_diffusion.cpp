#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <arborio/label_parse.hpp>

#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/math.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>
#include <arbor/schedule.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/util/any_ptr.hpp>

#include <arborenv/default_env.hpp>

using namespace std::string_literals;
using namespace arborio::literals;

using namespace arb;

constexpr double epsilon  = 1e-6;
#ifdef ARB_GPU_ENABLED
constexpr int    with_gpu = 0;
#else
constexpr int    with_gpu = -1;
#endif

struct linear: public recipe {
    linear(double x, double d, double c): extent{x}, diameter{d}, cv_length{c} {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_max_extent{cv_length};
        // Stick morphology
        // -----x-----
        segment_tree tree;
        auto p = mnpos;
        p = tree.append(p, { -extent, 0, 0, diameter}, {extent, 0, 0, diameter}, 1);
        morph = {tree};
    }

    arb::cell_size_type num_cells()                             const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type)            const override { return arb::cell_kind::cable; }
    std::any get_global_properties(arb::cell_kind)              const override { return gprop; }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type) const override { return {arb::cable_probe_ion_diff_concentration_cell{"na"}}; }
    util::unique_any get_cell_description(arb::cell_gid_type)   const override { return arb::cable_cell(morph, decor); }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override {
        std::vector<arb::event_generator> result;
        for (const auto& [t, w]: inject_at) {
            result.push_back(arb::explicit_generator({"Zap"}, w, std::vector<float>{t}));
        }
        return result;
    }

    arb::cable_cell_global_properties gprop;
    double extent = 1.0,
           diameter = 1.0,
           cv_length = 1.0;
    std::vector<std::tuple<float, float>> inject_at;
    morphology morph;
    arb::decor decor;

    linear& add_decay()  { decor.paint("(all)"_reg, arb::density("decay/x=na")); return *this; }
    linear& add_inject() { decor.place("(location 0 0.5)"_ls, arb::synapse("inject/x=na", {{"alpha", 200.0*cv_length}}), "Zap"); return *this; }
    linear& add_event(double t, float w) { inject_at.push_back({t, w}); return *this; }
    linear& set_diffusivity(double d, std::optional<region> rg = {}) {
        if (rg) decor.paint(*rg, ion_diffusivity{"na", d});
        else    decor.set_default(ion_diffusivity{"na", d});
        return *this;
    }
    linear& set_concentration(double d, std::optional<region> rg = {}) {
        if (rg) decor.paint(*rg, init_int_concentration{"na", d});
        else    decor.set_default(init_int_concentration{"na", d});
        return *this;
    }
};

using result_t = std::vector<std::tuple<double, double, double>>;

testing::AssertionResult all_near(const result_t& a, const result_t& b, double eps) {
    if (a.size() != b.size()) return testing::AssertionFailure() << "sequences differ in length";
    std::stringstream res;
    for (size_t ix = 0; ix < a.size(); ++ix) {
        const auto&[ax, ay, az] = a[ix];
        const auto&[bx, by, bz] = b[ix];
        if (fabs(ax - bx) > eps) res << " X elements " << ax << " and " << bx << " differ at index " << ix << ".";
        if (fabs(ay - by) > eps) res << " Y elements " << ay << " and " << by << " differ at index " << ix << ".";
        if (fabs(az - bz) > eps) res << " Z elements " << az << " and " << bz << " differ at index " << ix << ".";
    }
    std::string str = res.str();
    if (str.empty()) return testing::AssertionSuccess();
    else             return testing::AssertionFailure() << str;
}

testing::AssertionResult run(const linear& rec, const result_t exp) {
    result_t sample_values;
    auto sampler = [&sample_values](arb::probe_metadata pm, std::size_t n, const arb::sample_record* samples) {
        sample_values.clear();
        auto ptr = arb::util::any_cast<const arb::mcable_list*>(pm.meta);
        ASSERT_NE(ptr, nullptr);
        auto n_cable = ptr->size();
        for (std::size_t i = 0; i<n; ++i) {
            const auto& [val, _ig] = *arb::util::any_cast<const arb::cable_sample_range*>(samples[i].data);
            for (unsigned j = 0; j<n_cable; ++j) {
                arb::mcable loc = (*ptr)[j];
                sample_values.push_back({samples[i].time, loc.prox_pos, val[j]});
            }
        }
    };
    auto ctx = make_context({arbenv::default_concurrency(), with_gpu});
    auto sim = simulation{rec, ctx, partition_load_balance(rec, ctx)};
    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.1), sampler);
    sim.run(0.11, 0.01);
    return all_near(sample_values, exp, epsilon);
}

TEST(diffusion, errors) {
    {
        // Cannot R/W Xd w/o setting diffusivity
        auto rec = linear{30, 3, 1}.add_decay();
        ASSERT_THROW(run(rec, {}), illegal_diffusive_mechanism);
    }
    {
        // Cannot R/W Xd w/o setting diffusivity
        auto rec = linear{30, 3, 1}.add_inject();
        ASSERT_THROW(run(rec, {}), illegal_diffusive_mechanism);
    }
    {
        // No negative diffusivity
        auto rec = linear{30, 3, 1}.set_diffusivity(-42.0, "(all)"_reg);
        ASSERT_THROW(run(rec, {}), cable_cell_error);
    }
    {
        // No negative diffusivity
        auto rec = linear{30, 3, 1}.set_diffusivity(-42.0);
        ASSERT_THROW(run(rec, {}), cable_cell_error);
    }
}

TEST(diffusion, by_initial_concentration) {
    auto rec = linear{30, 3, 6}
        .set_diffusivity(0.005)
        .set_concentration(0.0)
        .set_concentration(0.1, "(cable 0 0.5 0.6)"_reg);
    result_t exp = {{0.000000, 0.000000, 0.000000},
                    {0.000000, 0.100000, 0.000000},
                    {0.000000, 0.200000, 0.000000},
                    {0.000000, 0.300000, 0.000000},
                    {0.000000, 0.400000, 0.000000},
                    {0.000000, 0.500000, 0.100000},
                    {0.000000, 0.600000, 0.000000},
                    {0.000000, 0.700000, 0.000000},
                    {0.000000, 0.800000, 0.000000},
                    {0.000000, 0.900000, 0.000000},
                    {0.100000, 0.000000, 0.000000},
                    {0.100000, 0.100000, 0.000000},
                    {0.100000, 0.200000, 0.000000},
                    {0.100000, 0.300000, 0.000023},
                    {0.100000, 0.400000, 0.001991},
                    {0.100000, 0.500000, 0.095973},
                    {0.100000, 0.600000, 0.001991},
                    {0.100000, 0.700000, 0.000023},
                    {0.100000, 0.800000, 0.000000},
                    {0.100000, 0.900000, 0.000000}};
    EXPECT_TRUE(run(rec, exp));
}

TEST(diffusion, by_event) {
    auto rec = linear{30, 3, 6}
        .set_diffusivity(0.005)
        .set_concentration(0.0)
        .add_inject()
        .add_event(0, 0.005);
    result_t exp = {{ 0.000000,  0.000000,  0.000000},
                    { 0.000000,  0.100000,  0.000000},
                    { 0.000000,  0.200000,  0.000000},
                    { 0.000000,  0.300000,  0.000000},
                    { 0.000000,  0.400000,  0.000000},
                    { 0.000000,  0.500000, 53.051647},
                    { 0.000000,  0.600000,  0.000000},
                    { 0.000000,  0.700000,  0.000000},
                    { 0.000000,  0.800000,  0.000000},
                    { 0.000000,  0.900000,  0.000000},
                    { 0.100000,  0.000000,  0.000000},
                    { 0.100000,  0.100000,  0.000001},
                    { 0.100000,  0.200000,  0.000100},
                    { 0.100000,  0.300000,  0.012051},
                    { 0.100000,  0.400000,  1.056130},
                    { 0.100000,  0.500000, 50.915085},
                    { 0.100000,  0.600000,  1.056130},
                    { 0.100000,  0.700000,  0.012051},
                    { 0.100000,  0.800000,  0.000100},
                    { 0.100000,  0.900000,  0.000001}};
    EXPECT_TRUE(run(rec, exp));
}

TEST(diffusion, decay) {
    auto rec = linear{30, 3, 6}
        .set_diffusivity(1e-300)
        .set_concentration(0.1)
        .add_decay();
    result_t exp = {{ 0.000000,  0.000000,  0.100000},
                    { 0.000000,  0.100000,  0.100000},
                    { 0.000000,  0.200000,  0.100000},
                    { 0.000000,  0.300000,  0.100000},
                    { 0.000000,  0.400000,  0.100000},
                    { 0.000000,  0.500000,  0.100000},
                    { 0.000000,  0.600000,  0.100000},
                    { 0.000000,  0.700000,  0.100000},
                    { 0.000000,  0.800000,  0.100000},
                    { 0.000000,  0.900000,  0.100000},
                    { 0.100000,  0.000000,  0.060647},
                    { 0.100000,  0.100000,  0.060647},
                    { 0.100000,  0.200000,  0.060647},
                    { 0.100000,  0.300000,  0.060647},
                    { 0.100000,  0.400000,  0.060647},
                    { 0.100000,  0.500000,  0.060647},
                    { 0.100000,  0.600000,  0.060647},
                    { 0.100000,  0.700000,  0.060647},
                    { 0.100000,  0.800000,  0.060647},
                    { 0.100000,  0.900000,  0.060647}};
    EXPECT_TRUE(run(rec, exp));
}

TEST(diffusion, decay_by_initial_concentration) {
    auto rec = linear{30, 3, 6}
        .set_diffusivity(0.005)
        .set_concentration(0.0)
        .set_concentration(0.1, "(cable 0 0.5 0.6)"_reg)
        .add_decay();
    result_t exp = {{ 0.000000,  0.000000,  0.000000},
                    { 0.000000,  0.100000,  0.000000},
                    { 0.000000,  0.200000,  0.000000},
                    { 0.000000,  0.300000,  0.000000},
                    { 0.000000,  0.400000,  0.000000},
                    { 0.000000,  0.500000,  0.100000},
                    { 0.000000,  0.600000,  0.000000},
                    { 0.000000,  0.700000,  0.000000},
                    { 0.000000,  0.800000,  0.000000},
                    { 0.000000,  0.900000,  0.000000},
                    { 0.100000,  0.000000,  0.000000},
                    { 0.100000,  0.100000,  0.000000},
                    { 0.100000,  0.200000,  0.000000},
                    { 0.100000,  0.300000,  0.000014},
                    { 0.100000,  0.400000,  0.001207},
                    { 0.100000,  0.500000,  0.058204},
                    { 0.100000,  0.600000,  0.001207},
                    { 0.100000,  0.700000,  0.000014},
                    { 0.100000,  0.800000,  0.000000},
                    { 0.100000,  0.900000,  0.000000}};
    EXPECT_TRUE(run(rec, exp));
}

TEST(diffusion, decay_by_event) {
    auto rec = linear{30, 3, 6}
        .set_diffusivity(0.005)
        .set_concentration(0.0)
        .add_decay()
        .add_inject()
        .add_event(0, 0.005);
    result_t exp = {{ 0.000000,  0.000000,  0.000000},
                    { 0.000000,  0.100000,  0.000000},
                    { 0.000000,  0.200000,  0.000000},
                    { 0.000000,  0.300000,  0.000000},
                    { 0.000000,  0.400000,  0.000000},
                    { 0.000000,  0.500000, 53.051647},
                    { 0.000000,  0.600000,  0.000000},
                    { 0.000000,  0.700000,  0.000000},
                    { 0.000000,  0.800000,  0.000000},
                    { 0.000000,  0.900000,  0.000000},
                    { 0.100000,  0.000000,  0.000000},
                    { 0.100000,  0.100000,  0.000000},
                    { 0.100000,  0.200000,  0.000061},
                    { 0.100000,  0.300000,  0.007308},
                    { 0.100000,  0.400000,  0.640508},
                    { 0.100000,  0.500000, 30.878342},
                    { 0.100000,  0.600000,  0.640508},
                    { 0.100000,  0.700000,  0.007308},
                    { 0.100000,  0.800000,  0.000061},
                    { 0.100000,  0.900000,  0.000000}};
    EXPECT_TRUE(run(rec, exp));
}

TEST(diffusion, setting_diffusivity) {
    // Skeleton recipe
    struct R: public recipe {
            R() {
                gprop.default_parameters = neuron_parameter_defaults;
                // make a two region tree
                tree.append(mnpos, { -1, 0, 0, 3}, {1, 0, 0, 3}, 1);
                tree.append(0, { -1, 0, 0, 3}, {1, 0, 0, 3}, 2);
                // Utilise diffusive ions
                dec.place("(location 0 0.5)"_ls, synapse("inject/x=bla", {{"alpha", 200.0}}), "Zap");
                dec.paint("(all)"_reg, density("decay/x=bla"));
            }

            cell_size_type num_cells()                                   const override { return 1; }
            cell_kind get_cell_kind(cell_gid_type)                       const override { return cell_kind::cable; }
            std::any get_global_properties(cell_kind)                    const override { return gprop; }
            util::unique_any get_cell_description(cell_gid_type)         const override { return cable_cell({tree}, dec); }

            cable_cell_global_properties gprop;
            segment_tree tree;
            decor dec;
    };

    // BAD: Trying to use a diffusive ion, but b=0.
    {
        R r;
        r.gprop.add_ion("bla", 1, 23, 42, 0, 0);
        EXPECT_THROW(simulation(r).run(1, 1), illegal_diffusive_mechanism);
    }
    // BAD: Trying to use a partially diffusive ion
    {
        R r;
        r.gprop.add_ion("bla", 1, 23, 42, 0, 0);
        r.dec.paint("(tag 1)"_reg, ion_diffusivity{"bla", 13});
        EXPECT_THROW(simulation(r).run(1, 1), cable_cell_error);
    }
    // OK: Using the global default
    {
        R r;
        r.gprop.add_ion("bla", 1, 23, 42, 0, 8);
        r.dec.paint("(tag 1)"_reg, ion_diffusivity{"bla", 13});
        EXPECT_NO_THROW(simulation(r).run(1, 1));
    }
    // OK: Using the cell default
    {
        R r;
        r.gprop.add_ion("bla", 1, 23, 42, 0, 0);
        r.dec.set_default(ion_diffusivity{"bla", 8});
        r.dec.paint("(tag 1)"_reg, ion_diffusivity{"bla", 13});
        EXPECT_NO_THROW(simulation(r).run(1, 1));
    }
    // BAD: Using an unknown species
    {
        R r;
        r.dec.set_default(ion_diffusivity{"bla", 8});
        r.dec.paint("(tag 1)"_reg, ion_diffusivity{"bla", 13});
        EXPECT_THROW(simulation(r).run(1, 1), cable_cell_error);
    }

}

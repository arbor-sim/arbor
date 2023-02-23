#include <gtest/gtest.h>

#include "common.hpp"

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/recipe.hpp>
#include <arbor/schedule.hpp>
#include <arbor/simulation.hpp>

#include <arborio/label_parse.hpp>
using namespace arborio::literals;

struct recipe: public arb::recipe {
    recipe(bool clamp, bool limit): limit{limit}, clamp{clamp} {
        gprop.default_parameters = arb::neuron_parameter_defaults;
        gprop.default_parameters.discretization = arb::cv_policy_max_extent(1.0);
    }

    arb::cell_size_type num_cells() const override { return 1; }
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override { return arb::cell_kind::cable; }
    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override { return {}; }
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override {
        auto tree = arb::segment_tree{};
        auto p = arb::mnpos;
        p = tree.append(p, {0, 0, 0, 1  }, {0, 0, 1, 1  }, 1); // soma 0-1
        p = tree.append(p, {0, 0, 1, 0.1}, {0, 0, 4, 0.1}, 2); // dend 1-4
        auto decor = arb::decor{}
            .paint("(tag 1)"_reg, arb::density("hh"))
            .paint("(tag 2)"_reg, arb::density("pas"))
            .place("(location 0 0.125)"_ls, arb::synapse("expsyn"), "tgt");

        if (clamp) decor.paint("(tag 1)"_reg, arb::voltage_process("v_clamp/v0=-42"));
        if (limit) decor.paint("(tag 1)"_reg, arb::voltage_process("v_limit/v_high=0,v_low=-60"));
        return arb::cable_cell(arb::morphology{tree}, decor);
    }
    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override {
        return { arb::cable_probe_membrane_voltage{"(location 0 0.125)"_ls},  // soma center: 0.25/2
                 arb::cable_probe_membrane_voltage{"(location 0 0.625)"_ls}}; // dend center: 0.75/2 + 0.25
    }
    std::vector<arb::event_generator> event_generators(arb::cell_gid_type) const override {
        return {arb::regular_generator({"tgt"}, 5.0, 0.2, 0.05)};
    }
    std::any get_global_properties(arb::cell_kind) const override { return gprop; }

    arb::cable_cell_global_properties gprop;
    bool limit = false;
    bool clamp = false;
};

struct Um_type {
    constexpr static double delta = 1e-6;

    double t;
    double u;

    friend std::ostream& operator<<(std::ostream& os, const Um_type& um) {
        os << "{ " << um.t << ", " << um.u << " }";
        return os;
    }

    friend bool operator==(const Um_type& lhs, const Um_type& rhs) {
        return (std::abs(lhs.t - rhs.t) <= delta)
            && (std::abs(lhs.u - rhs.u) <= delta);
    }
};

using um_s_type = std::vector<Um_type>;

TEST(v_process, clamp) {
    auto u_soma = um_s_type{};
    auto u_dend = um_s_type{};
    auto fun = [&u_soma, &u_dend](arb::probe_metadata pm,
                                  std::size_t n,
                                  const arb::sample_record* samples) {
        for (std::size_t ix = 0ul; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *arb::util::any_cast<const double*>(v);
            if (pm.id.index == 0) {
                u_soma.push_back({t, u});
            }
            else if (pm.id.index == 1) {
                u_dend.push_back({t, u});
            }
            else {
                throw std::runtime_error{"Unexpected probe id"};
            }
        }
    };
    auto sim = arb::simulation(recipe{true, false});
    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.05), fun);
    sim.run(1.0, 0.005);

    um_s_type exp_soma{{ 0, -65 },
                       { 0.05, -42 },
                       { 0.095, -42 },
                       { 0.145, -42 },
                       { 0.2, -42 },
                       { 0.25, -42 },
                       { 0.3, -42 },
                       { 0.35, -42 },
                       { 0.4, -42 },
                       { 0.45, -42 },
                       { 0.5, -42 },
                       { 0.55, -42 },
                       { 0.6, -42 },
                       { 0.65, -42 },
                       { 0.7, -42 },
                       { 0.75, -42 },
                       { 0.8, -42 },
                       { 0.85, -42 },
                       { 0.9, -42 },
                       { 0.95, -42 },};
    um_s_type exp_dend{{ 0, -65 },
                       { 0.05, -42.1167152 },
                       { 0.095, -42.0917893 },
                       { 0.145, -42.0464582 },
                       { 0.2, -41.9766685 },
                       { 0.25, -0.124773816 },
                       { 0.3, -0.0708536802 },
                       { 0.35, -0.0526604489 },
                       { 0.4, -0.043518448 },
                       { 0.45, -0.0380607556 },
                       { 0.5, -0.0344769712 },
                       { 0.55, -0.0319770579 },
                       { 0.6, -0.0301575597 },
                       { 0.65, -0.0287897554 },
                       { 0.7, -0.0277342043 },
                       { 0.75, -0.0269013197 },
                       { 0.8, -0.0262312483 },
                       { 0.85, -0.0256827719 },
                       { 0.9, -0.0252268065 },
                       { 0.95, -0.0248424087 }};
    ASSERT_TRUE(testing::seq_eq(u_soma, exp_soma));
    ASSERT_TRUE(testing::seq_eq(u_dend, exp_dend));
}

TEST(v_process, limit) {
    auto u_soma = um_s_type{};
    auto u_dend = um_s_type{};
    auto fun = [&u_soma, &u_dend](arb::probe_metadata pm,
                                  std::size_t n,
                                  const arb::sample_record* samples) {
        for (std::size_t ix = 0ul; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *arb::util::any_cast<const double*>(v);
            if (pm.id.index == 0) {
                u_soma.push_back({t, u});
            }
            else if (pm.id.index == 1) {
                u_dend.push_back({t, u});
            }
            else {
                throw std::runtime_error{"Unexpected probe id"};
            }
        }
    };
    auto sim = arb::simulation(recipe{false, true});
    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.05), fun);
    sim.run(1.0, 0.005);

    um_s_type exp_soma{{ 0, -65 },
                       { 0.05, -60 },
                       { 0.095, -60 },
                       { 0.145, -60 },
                       { 0.2, -60 },
                       { 0.25, -0.000425679283 },
                       { 0.3, 0 },
                       { 0.35, 0 },
                       { 0.4, 0 },
                       { 0.45, 0 },
                       { 0.5, 0 },
                       { 0.55, 0 },
                       { 0.6, 0 },
                       { 0.65, 0 },
                       { 0.7, 0 },
                       { 0.75, 0 },
                       { 0.8, 0 },
                       { 0.85, 0 },
                       { 0.9, 0 },
                       { 0.95, 0 }};
    um_s_type exp_dend{{ 0, -65 },
                       { 0.05, -60.032677 },
                       { 0.095, -60.0308171 },
                       { 0.145, -60.028791 },
                       { 0.2, -60.0266703 },
                       { 0.25, -0.0178456839 },
                       { 0.3, -0.0168966758 },
                       { 0.35, -0.0162935748 },
                       { 0.4, -0.0158862702 },
                       { 0.45, -0.0156348508 },
                       { 0.5, -0.0155055671 },
                       { 0.55, -0.0154673288 },
                       { 0.6, -0.0154936751 },
                       { 0.65, -0.0155635334 },
                       { 0.7, -0.0156609133 },
                       { 0.75, -0.015774124 },
                       { 0.8, -0.0158948866 },
                       { 0.85, -0.0160175185 },
                       { 0.9, -0.0161382535 },
                       { 0.95, -0.0162547064 },};
    ASSERT_TRUE(testing::seq_eq(u_soma, exp_soma));
    ASSERT_TRUE(testing::seq_eq(u_dend, exp_dend));
}

TEST(v_process, clamp_fine) {
    auto u_soma = um_s_type{};
    auto u_dend = um_s_type{};
    auto fun = [&u_soma, &u_dend](arb::probe_metadata pm,
                                  std::size_t n,
                                  const arb::sample_record* samples) {
        for (std::size_t ix = 0ul; ix < n; ++ix) {
            const auto& [t, v] = samples[ix];
            double u = *arb::util::any_cast<const double*>(v);
            if (pm.id.index == 0) {
                u_soma.push_back({t, u});
            }
            else if (pm.id.index == 1) {
                u_dend.push_back({t, u});
            }
            else {
                throw std::runtime_error{"Unexpected probe id"};
            }
        }
    };
    auto rec = recipe{true, false};
    rec.gprop.default_parameters.discretization = arb::cv_policy_max_extent(0.5);
    auto sim = arb::simulation(rec);
    sim.add_sampler(arb::all_probes, arb::regular_schedule(0.05), fun);
    sim.run(1.0, 0.005);

    um_s_type exp_soma{{ 0, -65 },
                       { 0.05, -42 },
                       { 0.095, -42 },
                       { 0.145, -42 },
                       { 0.2, -42 },
                       { 0.25, -42 },
                       { 0.3, -42 },
                       { 0.35, -42 },
                       { 0.4, -42 },
                       { 0.45, -42 },
                       { 0.5, -42 },
                       { 0.55, -42 },
                       { 0.6, -42 },
                       { 0.65, -42 },
                       { 0.7, -42 },
                       { 0.75, -42 },
                       { 0.8, -42 },
                       { 0.85, -42 },
                       { 0.9, -42 },
                       { 0.95, -42 },};
    um_s_type exp_dend{{ 0, -65 },
                       { 0.05, -42.1164544 },
                       { 0.095, -42.0915289 },
                       { 0.145, -42.0461939 },
                       { 0.2, -41.9764002 },
                       { 0.25, -0.124099713 },
                       { 0.3, -0.0701885048 },
                       { 0.35, -0.0519983288 },
                       { 0.4, -0.0428578593 },
                       { 0.45, -0.0374010627 },
                       { 0.5, -0.0338178467 },
                       { 0.55, -0.0313183138 },
                       { 0.6, -0.0294990809 },
                       { 0.65, -0.0281314686 },
                       { 0.7, -0.0270760611 },
                       { 0.75, -0.0262432877 },
                       { 0.8, -0.025573305 },
                       { 0.85, -0.0250249014 },
                       { 0.9, -0.0245689972 },
                       { 0.95, -0.0241846519 },};
    ASSERT_TRUE(testing::seq_eq(u_soma, exp_soma));
    ASSERT_TRUE(testing::seq_eq(u_dend, exp_dend));
}

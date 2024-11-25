// Test performance of vectorization for mechanism implementations.
//
// Start with pas (passive dendrite) mechanism

// NOTE: This targets an earlier version of the Arbor API and
// will need to be reworked in order to compile.

#include <any>
#include <fstream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/segment_tree.hpp>

#include "backends/multicore/fvm.hpp"
#include "benchmark/benchmark.h"
#include "execution_context.hpp"
#include "fvm_lowered_cell_impl.hpp"

using namespace arb;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

mechanism_ptr& find_mechanism(const std::string& name, fvm_cell& cell) {
    auto &mechs = cell.mechanisms();
    auto it = std::find_if(mechs.begin(),
                           mechs.end(),
                           [&](mechanism_ptr& m){return m->internal_name()==name;});
    if (it==mechs.end()) {
        std::cerr << "couldn't find mechanism with name " << name << "\n";
        exit(1);
    }
    return *it;
}

class recipe_expsyn_1_branch: public recipe {
    unsigned num_comp_;
    unsigned num_synapse_;
    arb::cable_cell_global_properties gprop_;

public:
    recipe_expsyn_1_branch(unsigned num_comp, unsigned num_synapse):
            num_comp_(num_comp), num_synapse_(num_synapse) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;

        double soma_radius = 12.6157/2.0;
        double dend_radius = 1.0/2;
        double dend_length = 200;

        // Add soma.
        auto s0 = tree.append(arb::mnpos, {0,0,-soma_radius,soma_radius}, {0,0,soma_radius,soma_radius}, 1);
        // Add dendrite
        auto s1 = tree.append(s0, {0,0,soma_radius,dend_radius}, 3);
        tree.append(s1, {0,0,soma_radius+dend_length,dend_radius}, 3);

        arb::decor decor;
        decor.paint(arb::reg::tagged(1), arb::density("pas"));
        decor.set_default(arb::cv_policy_max_extent((dend_length+soma_radius*2)/num_comp_));

        auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);
        for(unsigned i = 0; i < num_synapse_; i++) {
            auto gen = std::mt19937(i);
            decor.place(arb::mlocation{0, distribution(gen)}, arb::synapse("expsyn"), "syn");
        }

        return arb::cable_cell{arb::morphology(tree), decor};
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

class recipe_pas_1_branch: public recipe {
    unsigned num_comp_;
    arb::cable_cell_global_properties gprop_;
public:
    recipe_pas_1_branch(unsigned num_comp): num_comp_(num_comp) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;

        double soma_radius = 12.6157/2.0;
        double dend_radius = 1.0/2;
        double dend_length = 200;

        // Add soma.
        auto s0 = tree.append(arb::mnpos, {0,0,-soma_radius,soma_radius}, {0,0,soma_radius,soma_radius}, 1);
        // Add dendrite
        auto s1 = tree.append(s0, {0,0,soma_radius,dend_radius}, 3);
        tree.append(s1, {0,0,soma_radius+dend_length,dend_radius}, 3);

        arb::decor decor;
        decor.paint(arb::reg::all(), arb::density("pas"));
        decor.set_default(arb::cv_policy_max_extent((dend_length+soma_radius*2)/num_comp_));

        return arb::cable_cell {arb::morphology(tree), decor};
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

class recipe_pas_3_branches: public recipe {
    unsigned num_comp_;
    arb::cable_cell_global_properties gprop_;
public:
    recipe_pas_3_branches(unsigned num_comp): num_comp_(num_comp) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;

        double soma_radius = 12.6157/2.0;
        double dend_radius = 1.0/2;
        double dend_length = 200;

        // Add soma.
        auto s0 = tree.append(arb::mnpos, {0,0,-soma_radius,soma_radius}, {0,0,soma_radius,soma_radius}, 1);
        // Add dendrite
        auto s1 = tree.append(s0, {0          ,0          ,soma_radius,             dend_radius}, 3);
        auto s2 = tree.append(s1, {0          ,0          ,soma_radius+dend_length, dend_radius}, 3);
        tree.append(s2,           {0          ,dend_length,soma_radius+dend_length, dend_radius}, 3);
        tree.append(s2,           {dend_length,0          ,soma_radius+dend_length, dend_radius}, 3);

        arb::decor decor;
        decor.paint(arb::reg::all(), arb::density("pas"));
        decor.set_default(arb::cv_policy_max_extent((dend_length*3+soma_radius*2)/num_comp_));

        return arb::cable_cell{arb::morphology(tree), decor};
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

class recipe_hh_1_branch: public recipe {
    unsigned num_comp_;
    arb::cable_cell_global_properties gprop_;
public:
    recipe_hh_1_branch(unsigned num_comp): num_comp_(num_comp) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;

        double soma_radius = 12.6157/2.0;
        double dend_radius = 1.0/2;
        double dend_length = 200;

        // Add soma.
        auto s0 = tree.append(arb::mnpos, {0,0,-soma_radius,soma_radius}, {0,0,soma_radius,soma_radius}, 1);
        // Add dendrite
        auto s1 = tree.append(s0, {0          ,0          ,soma_radius,             dend_radius}, 3);
        tree.append(s1,           {0          ,0          ,soma_radius+dend_length, dend_radius}, 3);

        arb::decor decor;
        decor.paint(arb::reg::all(), arb::density("hh"));
        decor.set_default(arb::cv_policy_max_extent((dend_length+soma_radius*2)/num_comp_));

        return arb::cable_cell{arb::morphology(tree), decor};
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

class recipe_hh_3_branches: public recipe {
    unsigned num_comp_;
    arb::cable_cell_global_properties gprop_;
public:
    recipe_hh_3_branches(unsigned num_comp): num_comp_(num_comp) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;

        double soma_radius = 12.6157/2.0;
        double dend_radius = 1.0/2;
        double dend_length = 200;

        // Add soma.
        auto s0 = tree.append(arb::mnpos, {0,0,-soma_radius,soma_radius}, {0,0,soma_radius,soma_radius}, 1);
        // Add dendrite
        auto s1 = tree.append(s0, {0          ,0          ,soma_radius,             dend_radius}, 3);
        auto s2 = tree.append(s1, {0          ,0          ,soma_radius+dend_length, dend_radius}, 3);
        tree.append(          s2, {0          ,dend_length,soma_radius+dend_length, dend_radius}, 3);
        tree.append(          s2, {dend_length,0          ,soma_radius+dend_length, dend_radius}, 3);

        arb::decor decor;
        decor.paint(arb::reg::all(), arb::density("hh"));
        decor.set_default(arb::cv_policy_max_extent((dend_length*3+soma_radius*2)/num_comp_));

        return arb::cable_cell{arb::morphology(tree), decor};
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

    std::any get_global_properties(arb::cell_kind) const override {
        return gprop_;
    }
};

void expsyn_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    recipe_expsyn_1_branch rec_expsyn_1_branch(ncomp, nsynapse);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_expsyn_1_branch);

    auto& m = find_mechanism("expsyn", cell);

    while (state.KeepRunning()) {
        m->update_current();
    }
}

void expsyn_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    recipe_expsyn_1_branch rec_expsyn_1_branch(ncomp, nsynapse);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_expsyn_1_branch);

    auto& m = find_mechanism("expsyn", cell);

    while (state.KeepRunning()) {
        m->update_state();
    }
}

void pas_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_pas_1_branch rec_pas_1_branch(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_pas_1_branch);

    auto& m = find_mechanism("pas", cell);

    while (state.KeepRunning()) {
        m->update_current();
    }
}

void pas_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_pas_3_branches rec_pas_3_branches(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_pas_3_branches);

    auto& m = find_mechanism("pas", cell);

    while (state.KeepRunning()) {
        m->update_current();
    }
}

void hh_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_hh_1_branch);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->update_state();
    }
}

void hh_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_hh_1_branch);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->update_current();
    }
}

void hh_3_branches_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_hh_3_branches);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->update_state();
    }
}

void hh_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    fvm_cell cell((execution_context()));
    cell.initialize({0}, rec_hh_3_branches);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->update_current();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000}) {
        b->Args({ncomps});
    }
}
void run_exp_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000}) {
        b->Args({ncomps, ncomps*10});
    }
}
BENCHMARK(expsyn_1_branch_current)->Apply(run_exp_custom_arguments);
BENCHMARK(expsyn_1_branch_state)->Apply(run_exp_custom_arguments);
BENCHMARK(pas_1_branch_current)->Apply(run_custom_arguments);
BENCHMARK(hh_1_branch_current)->Apply(run_custom_arguments);
BENCHMARK(hh_1_branch_state)->Apply(run_custom_arguments);
BENCHMARK(pas_3_branches_current)->Apply(run_custom_arguments);
BENCHMARK(hh_3_branches_current)->Apply(run_custom_arguments);
BENCHMARK(hh_3_branches_state)->Apply(run_custom_arguments);
BENCHMARK_MAIN();

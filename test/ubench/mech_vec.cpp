// Test performance of vectorization for mechanism implementations.
//
// Start with pas (passive dendrite) mechanism

// NOTE: This targets an earlier version of the Arbor API and
// will need to be reworked in order to compile.

#include <fstream>

#include <arbor/cable_cell.hpp>

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
public:
    recipe_expsyn_1_branch(unsigned num_comp, unsigned num_synapse):
            num_comp_(num_comp), num_synapse_(num_synapse) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->set_compartments(num_comp_-1);
            }
        }

        auto distribution = std::uniform_real_distribution<float>(0.f, 1.0f);

        for(unsigned i = 0; i < num_synapse_; i++) {
            auto gen = std::mt19937(i);
            c.add_synapse({1, distribution(gen)}, "expsyn");
        }

        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

};

class recipe_pas_1_branch: public recipe {
    unsigned num_comp_;
public:
    recipe_pas_1_branch(unsigned num_comp): num_comp_(num_comp) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->add_mechanism("pas");
                seg->set_compartments(num_comp_-1);
            }
        }
        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

};

class recipe_pas_3_branches: public recipe {
    unsigned num_comp_;
public:
    recipe_pas_3_branches(unsigned num_comp): num_comp_(num_comp) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);
        c.add_cable(1, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);
        c.add_cable(1, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->add_mechanism("pas");
                seg->set_compartments(num_comp_-1);
            }
        }
        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

};

class recipe_hh_1_branch: public recipe {
    unsigned num_comp_;
public:
    recipe_hh_1_branch(unsigned num_comp): num_comp_(num_comp) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("hh");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->add_mechanism("hh");
                seg->set_compartments(num_comp_-1);
            }
        }
        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }

};

class recipe_hh_3_branches: public recipe {
    unsigned num_comp_;
public:
    recipe_hh_3_branches(unsigned num_comp): num_comp_(num_comp) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    virtual util::unique_any get_cell_description(cell_gid_type gid) const override {
        cable_cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);
        c.add_cable(1, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);
        c.add_cable(1, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->add_mechanism("hh");
                seg->set_compartments(num_comp_-1);
            }
        }
        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable;
    }
};

void expsyn_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    recipe_expsyn_1_branch rec_expsyn_1_branch(ncomp, nsynapse);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_expsyn_1_branch, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("expsyn", cell);

    while (state.KeepRunning()) {
        m->nrn_current();
    }
}

void expsyn_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    recipe_expsyn_1_branch rec_expsyn_1_branch(ncomp, nsynapse);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_expsyn_1_branch, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("expsyn", cell);

    while (state.KeepRunning()) {
        m->nrn_state();
    }
}

void pas_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_pas_1_branch rec_pas_1_branch(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_pas_1_branch, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("pas", cell);

    while (state.KeepRunning()) {
        m->nrn_current();
    }
}

void pas_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_pas_3_branches rec_pas_3_branches(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_pas_3_branches, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("pas", cell);

    while (state.KeepRunning()) {
        m->nrn_current();
    }
}

void hh_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_hh_1_branch, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->nrn_state();
    }
}

void hh_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_hh_1_branch, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->nrn_current();
    }
}

void hh_3_branches_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_hh_3_branches, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->nrn_state();
    }
}

void hh_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    std::vector<fvm_index_type> cell_to_intdom;
    probe_association_map<probe_handle> probe_handles;

    fvm_cell cell((execution_context()));
    cell.initialize(gids, rec_hh_3_branches, cell_to_intdom, target_handles, probe_handles);

    auto& m = find_mechanism("hh", cell);

    while (state.KeepRunning()) {
        m->nrn_current();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
        b->Args({ncomps});
    }
}
void run_exp_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000, 1000000}) {
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

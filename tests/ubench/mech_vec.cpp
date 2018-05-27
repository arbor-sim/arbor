// Test performance of vectorization for mechanism implementations.
//
// Start with pas (passive dendrite) mechanism

#include <backends/multicore/fvm.hpp>
#include <benchmark/benchmark.h>
#include <fvm_lowered_cell_impl.hpp>
#include <fstream>

using namespace arb;

using backend = arb::multicore::backend;
using fvm_cell = arb::fvm_lowered_cell_impl<backend>;

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
        cell c;

        auto soma = c.add_soma(12.6157/2.0);
        soma->add_mechanism("pas");

        c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

        for (auto& seg: c.segments()) {
            if (seg->is_dendrite()) {
                seg->set_compartments(num_comp_-1);
            }
        }

        for(unsigned i = 0; i < num_synapse_; i++) {
            float loc = ((float)i/(float)num_synapse_);
            c.add_synapse({1, loc}, "expsyn");
        }

        return std::move(c);
    }

    virtual cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_kind::cable1d_neuron;
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
        cell c;

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
        return cell_kind::cable1d_neuron;
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
        cell c;

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
        return cell_kind::cable1d_neuron;
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
        cell c;

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
        return cell_kind::cable1d_neuron;
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
        cell c;

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
        return cell_kind::cable1d_neuron;
    }

    //virtual cell_size_type num_targets(cell_gid_type) const { return 0; }
};

void expsyn_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_expsyn_1_branch recipe_expsyn_1_branch1(ncomp, nsynapse);

    fvm_cell cell;
    cell.initialize(gids, recipe_expsyn_1_branch1, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "expsyn") {
            idx = i;
            break;
        }
    }

    if (idx==-1) {
        std::cout << "ERROR: couldn't find pas\n";
        exit(1);
    }

    auto& m = mechs[idx];
    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void expsyn_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    const unsigned nsynapse = state.range(1);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_expsyn_1_branch recipe_expsyn_1_branch1(ncomp, nsynapse);

    fvm_cell cell;
    cell.initialize(gids, recipe_expsyn_1_branch1, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "expsyn") {
            idx = i;
            break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find pas\n";
        exit(1);
    }

    auto& m = mechs[idx];
    while (state.KeepRunning()) {
        // call nrn_state
        m->nrn_state();
    }
}

void pas_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_pas_1_branch rec_pas_1_branch(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_pas_1_branch, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "pas") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find pas\n";
        exit(1);
    }

    auto& m = mechs[idx];
    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void pas_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_pas_3_branches rec_pas_3_branches(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_pas_3_branches, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "pas") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find pas\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void hh_1_branch_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_hh_1_branch, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "hh") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find hh\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_state
        m->nrn_state();
    }
}

void hh_1_branch_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_hh_1_branch rec_hh_1_branch(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_hh_1_branch, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "hh") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find hh\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void hh_3_branches_state(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_hh_3_branches, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "hh") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find hh\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_state
        m->nrn_state();
    }
}

void hh_3_branches_current(benchmark::State& state) {
    const unsigned ncomp = state.range(0);
    std::vector<cell_gid_type> gids = {0};
    std::vector<target_handle> target_handles;
    probe_association_map<probe_handle> probe_handles;
    recipe_hh_3_branches rec_hh_3_branches(ncomp);

    fvm_cell cell;
    cell.initialize(gids, rec_hh_3_branches, target_handles, probe_handles);

    int idx = -1;
    auto &mechs = cell.mechanisms();
    for (unsigned i=0; i<mechs.size(); ++i) {
        if (mechs[i]->internal_name() == "hh") {
            idx = i;
			break;
        }
    }
    if (idx==-1) {
        std::cout << "ERROR: couldn't find hh\n";
        exit(1);
    }
    auto& m = mechs[idx];

    while (state.KeepRunning()) {
        // call nrn_current
        m->nrn_current();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
        b->Args({ncomps});
    }
}
void run_exp_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto ncomps: {10, 100, 1000, 10000, 100000, 1000000, 10000000}) {
        b->Args({ncomps, ncomps*10});
    }
}
BENCHMARK(expsyn_1_branch_current)->Apply(run_exp_custom_arguments);
BENCHMARK(expsyn_1_branch_state)->Apply(run_exp_custom_arguments);
/*BENCHMARK(pas_1_branch_current)->Apply(run_custom_arguments);
BENCHMARK(hh_1_branch_current)->Apply(run_custom_arguments);
BENCHMARK(hh_1_branch_state)->Apply(run_custom_arguments);
BENCHMARK(pas_3_branches_current)->Apply(run_custom_arguments);
BENCHMARK(hh_3_branches_current)->Apply(run_custom_arguments);
BENCHMARK(hh_3_branches_state)->Apply(run_custom_arguments);*/
BENCHMARK_MAIN();

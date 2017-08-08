#include "../gtest.h"

#include <cell_group_factory.hpp>
#include <fstream>
#include <pss_cell_description.hpp>
#include <pss_cell_group.hpp>

using namespace nest::mc;

// Produces a file of all produced spikes.
// Does not test anything.
// These spikes are supposed to be plotted later
// with provided python script "./tests/unit/plot_spikes_pss.py
TEST(pss_cell_group, cell_group_factory) {
    std::vector<util::unique_any> cells;
    cells.emplace_back(pss_cell_description());
    //cells.emplace_back(pss_cell_description());

    cell_group_ptr group = cell_group_factory(
         cell_kind::poisson_spike_source,
         0,
         cells,
         backend_policy::use_multicore);

    // Advance group to 500s by steps of 0.01.
    group->advance(500, 0.01);
    std::vector<spike> spikes = group->spikes();

    // Output spikes to a file.
    std::ofstream spikes_file;
    spikes_file.open("../../tests/unit/pps_spikes.txt");
    ASSERT_TRUE(spikes_file);

    for(auto& spike : spikes) {
        spikes_file << spike.time << std::endl;
    }

    spikes_file.close();
}

#pragma once

#include <algorithm>
#include <iostream>

#include <vector>
#include <memory>
#include <utility>
#include <string>

#include <spike.hpp>
#include <util.hpp>
#include <common_types.hpp>
#include "exporter_interface.hpp"
#include "exporter_spike_file.hpp"
#include "exporter_spike_single_file.hpp"

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy>
class export_manager {
public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    export_manager(bool spike_file_output, bool single_file_per_rank, bool over_write,
        std::string output_path, std::string file_name, std::string file_extention)
    {
        if (!spike_file_output)
        {
            return;
        }

        // single file per rank
        if (single_file_per_rank) { 
            rank_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_file<Time, CommunicationPolicy> >(
                        file_name, output_path, file_extention, over_write));
        }

        // single file per simulation
        if (!single_file_per_rank) { 
            single_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_file<Time, CommunicationPolicy> >(
                        file_name, output_path, file_extention, over_write));
        }
    }

    void do_export_local(const std::vector<spike_type>& spikes)
    {
        local_spikes_.insert(std::end(local_spikes_),
            std::begin(spikes), std::end(spikes));

        for (auto &exporter : rank_exporters_)
        {
            exporter->do_export(spikes);
        }

        local_spikes_.clear();
    }

    void do_export_global(const std::vector<spike_type>& spikes)
    {
        // We only output on a single rank
        if (!communication_policy_.id() == 0) {
            return;
        }

        global_spikes_.insert(std::end(global_spikes_),
            std::begin(spikes), std::end(spikes));


        for (auto &exporter : single_exporters_)
        {
            exporter->do_export(spikes);
        }

        global_spikes_.clear();
    }

private:
    // 
    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > rank_exporters_;

    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > single_exporters_;

    CommunicationPolicy communication_policy_;

    // local storage for sending spikes
    std::vector<spike_type> local_spikes_;
    std::vector<spike_type> global_spikes_;
};



} //communication
} // namespace mc
} // namespace nest

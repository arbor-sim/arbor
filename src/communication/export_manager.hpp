#pragma once

#include <algorithm>
#include <iostream>

#include <vector>
#include <memory>
#include <utility>

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
    export_manager() 
    {

        if (true) { // single file per rank

            rank_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_file<Time, CommunicationPolicy> >(
                        "rank", "./", "gdf"));


        }


        if (true) { // single file per simulation
            single_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_single_file<Time, CommunicationPolicy> >(
                    "single", "./", "gdf"));
        }
    }

    void do_export_rank(const std::vector<spike_type>& spikes)
    {
        // TODO: do the buffering of the spikes here and not in the 
        //      exporters itself!!!
        for (auto &exporter : rank_exporters_)
        {
            exporter->add_and_export(spikes);
        }
    }

    void do_export_single(const std::vector<spike_type>& spikes)
    {
        if (!communication_policy_.id() == 0) {
            return;
        }

        for (auto &exporter : single_exporters_)
        {
            exporter->add_and_export(spikes);
        }
    }

private:
    // 
    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > rank_exporters_;

    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > single_exporters_;


    CommunicationPolicy communication_policy_;
};



} //communication
} // namespace mc
} // namespace nest
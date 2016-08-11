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


// export_manager manages the export of simulation parameters to the outside 
// world. Examples could be output to disk, but also more advanced communication
// via mpi can be possible.
// The parameters to be exported are buffered locally.
// There are two export methods implemented: rank local where an export is
// done on each rank and a single export per simulation
// The output file is constructed from constructor arguments and will always
// contain a index also when a single file is written. THis allows automated
// parsing with a simple regex
// 
// TODO: The exporter currently only exports spikes to file. The constructor
// arguments reflect this. In future version a better way to configure this
// class is needed.
template <typename Time, typename CommunicationPolicy>
class export_manager {
public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;

    // Constructor
    // spike_file_output initiates spike file output. If false the object is 
    //          constructed in a valid state. No exporters are registed so no output
    //          is done
    // single_file_per_rank if true only rank zero performs output to disk
    // over_write if true will overwrite the specified output file
    // output_path  relative or absolute path
    // file_name    will be appended with "_x" with x the rank number
    // file_extention  a seperator will be added automatically
    export_manager(bool spike_file_output, bool single_file_per_rank, bool over_write,
        std::string output_path, std::string file_name, std::string file_extention)
        :
        spike_file_output_(spike_file_output) 
    {
        // simple switch to turn of the export, object will still be in a valid
        // state.
        if (!spike_file_output_) {
            return;
        }

        // single file per rank exporters
        if (single_file_per_rank) { 
            rank_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_file<Time, CommunicationPolicy> >(
                        file_name, output_path, file_extention, over_write));
        }

        // single file per simulation exporters
        if (!single_file_per_rank) { 
            single_exporters_.push_back(
                nest::mc::util::make_unique<
                    nest::mc::communication::exporter_spike_file<Time, CommunicationPolicy> >(
                        file_name, output_path, file_extention, over_write));
        }
    }

    // Perform a export of local spikes, typically used for exporting to multi-
    // ple files from each rank individually.
    // spikes are buffer before export
    void do_export_local(const std::vector<spike_type>& spikes)
    {
        // TODO: No export needed, so exit
        if (!spike_file_output_) {
            return;
        }

        local_spikes_.insert(std::end(local_spikes_),
            std::begin(spikes), std::end(spikes));

        // TODO: Do each exporter in a parallel thread?
        for (auto &exporter : rank_exporters_) {
            exporter->do_export(spikes);
        }

        local_spikes_.clear();
    }

    // Perform a export of global spikes, typically used for exporting spikes
    // from a single rank in a simulation
    // spikes are buffer before export
    void do_export_global(const std::vector<spike_type>& spikes)
    {
        if (!spike_file_output_) {
            return;
        }

        // We only output on a single rank
        if (!communication_policy_.id() == 0) {
            return;
        }

        global_spikes_.insert(std::end(global_spikes_),
            std::begin(spikes), std::end(spikes));

        // TODO: Do each exporter in a parallel thread?
        for (auto &exporter : single_exporters_) {
            exporter->do_export(spikes);
        }

        global_spikes_.clear();
    }

private:
    bool spike_file_output_;
    
    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > rank_exporters_;
    std::vector<std::unique_ptr<exporter_interface<Time, CommunicationPolicy> > > single_exporters_;

    CommunicationPolicy communication_policy_;

    // local buffer for spikes
    std::vector<spike_type> local_spikes_;
    std::vector<spike_type> global_spikes_;
};

} //communication
} // namespace mc
} // namespace nest

#pragma once

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <common_types.hpp>
#include <util.hpp>
#include <spike.hpp>


#include "exporter_interface.hpp"

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy> // TODO: Templating on data type, for now only spike_type
class exporter_spike_single_file : public exporter_interface<Time, CommunicationPolicy> {

public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    using communication_policy_type = CommunicationPolicy;

    // 
    exporter_spike_single_file(std::string file_name, std::string path,
        std::string file_extention, bool over_write=true)
        :
        ok_(false)
    {
        if (!communication_policy_.id() == 0) {
            ok_ = true;
            return;
        }

        std::string file_path(create_output_file_path(
            file_name, path, file_extention, communication_policy_.id()));

        //test if the file exist and depending on over_write throw or delete
        std::ifstream f(file_path);
        if (f.good()) {
            if (!over_write) {
                std::string error_string("Tried opening file for writing but it exists and over_write is false:\n" +
                    file_path);

                throw std::runtime_error(error_string);
            }

            std::remove(file_path.c_str());
        }

        file_handle_ = nest::mc::util::make_unique<std::ofstream>(file_path,
                                                       std::fstream::app);

        if (file_handle_->good()) {
            ok_ = true;
        }

        // Force output of the spike times with precision
        // TODO: We need to make this selectable
        file_handle_->precision(4);
        file_handle_->setf(std::ios::fixed, std::ios::floatfield);
    }

       
    // Performs the export of the data, 
    // Does not throw
    void do_export() override
    {        
        if (!communication_policy_.id() == 0) {
            return;
        }

        for (auto spike : spikes_) {
            *file_handle_ << spike.source.gid << " " << spike.time << std::endl;
        }
        if (!file_handle_->good()){
            ok_ = false;
        }

        spikes_.clear();
    }

    // Add data to the internal storage to be exported
    // Does not do the actual export  
    void add_data(std::vector<spike_type>spikes) override
    {
        if (!communication_policy_.id() == 0) {
            return;
        }

        spikes_.insert(std::end(spikes_), 
                       std::begin(spikes), std::end(spikes));       
    }

    // Add and export data to file in a single function
    void add_and_export(const std::vector<spike_type>& spikes) override
    {
        if (!communication_policy_.id() == 0) {
            return;
        }

        add_data(spikes);
        do_export();
    }

    // Internal state is ok
    // We are working with fstreams possibly on a seperate thread
    // We need a way to assertain the status of the stream
    bool ok() const override
    {
        if (!communication_policy_.id() == 0) {
            return true;
        }

        return ok_ && file_handle_->good();
    }

    // Creates an indexed filename
    static std::string create_output_file_path(std::string file_name, std::string path,
        std::string file_extention, unsigned index)
    {
        std::string file_path = path + file_name + "_" + std::to_string(index) +
                                "." + file_extention;
        // TODO: Nest does not produce the indexing for nrank == 0
        //       I have the feeling this disrupts consistent output. Id rather
        //       always put the zero in.
        return file_path;
    }

private:   
    // Are we in a valid state?
    bool ok_;

    // Handle to our owned opened file handle
    std::unique_ptr<std::ofstream>  file_handle_;
    
    // local storage for sending spikes
    std::vector<spike_type> spikes_;

    communication_policy_type communication_policy_;
};

} //communication
} // namespace mc
} // namespace nest

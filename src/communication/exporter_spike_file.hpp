#pragma once

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <cstring>

#include <common_types.hpp>
#include <util.hpp>
#include <spike.hpp>


#include "exporter_interface.hpp"

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy> // TODO: Templating on data type, for now only spike_type
class exporter_spike_file : public exporter_interface<Time, CommunicationPolicy> {

public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    using communication_policy_type = CommunicationPolicy;

    // 
    exporter_spike_file(std::string file_name, std::string path, 
        std::string file_extention, bool over_write=true, 
        bool single_file_per_rank=true)
        :
        ok_(false)
    {
        std::string file_path(create_output_file_path(
            file_name, path, file_extention, communication_policy_.id()));

        //test if the file exist and depending on over_write throw or delete
        std::ifstream f(file_path);
        if (f.good()) {
            if (!over_write) {
                std::string error_string("Tried opening file for writing but it exists and over_write is false: " +
                    file_path);

                throw std::runtime_error(error_string);
            }

            std::remove(file_path.c_str());
        }
        
        buffer = new char[length];
        file_handle_ = nest::mc::util::make_unique<std::ofstream>(file_path,
                                                       std::fstream::app);

        if (file_handle_->good()) {
            ok_ = true;
        }
    }

       
    // Performs the export of the data, 
    // Does not throw
    void do_export(const std::vector<spike_type>& spikes) override
    {
        unsigned current_loc_in_buffer = 0;
        unsigned nr_chars_written = 0;
        char single_value_buffer[20]; // Much to big 

        // Some constants needed for printing
        const char * space = " ";
        const char * endline = "\n";

        for (auto spike : spikes)
        {
            // First the id as output
            nr_chars_written = std::snprintf(single_value_buffer, 20, "%u", 
                                             spike.source.gid);
            std::memcpy(buffer + current_loc_in_buffer, single_value_buffer,
                        nr_chars_written);
            current_loc_in_buffer += nr_chars_written;

            // The a space
            std::memcpy(buffer + current_loc_in_buffer, space, 1);
            current_loc_in_buffer += 1;

            // Then the float
            nr_chars_written = std::snprintf(single_value_buffer, 20, "%.4f",
                spike.time);
            std::memcpy(buffer + current_loc_in_buffer, single_value_buffer,
                nr_chars_written);
            current_loc_in_buffer += nr_chars_written;

            // Then the endline
            std::memcpy(buffer + current_loc_in_buffer, endline, 2);
            current_loc_in_buffer += 1;  // Only a single char in the actual file!!

            // Check if we are nearing the end of our buffer
            if (current_loc_in_buffer > length - 45)
            {
                file_handle_->write(buffer, current_loc_in_buffer);
                current_loc_in_buffer = 0;
            }
        }
        
        // also write to buffer at end of the spikes processing
        if (current_loc_in_buffer != 0)
        {
            file_handle_->write(buffer, current_loc_in_buffer);
            current_loc_in_buffer = 0; // not needed
        }


        file_handle_->flush();


        if (!file_handle_->good()){
            ok_ = false;
        }

    }

    // Internal state is ok
    // We are working with fstreams possibly on a seperate thread
    // We need a way to assertain the status of the stream
    bool ok() const override
    {
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
    
    communication_policy_type communication_policy_;

    // Buffer (and size) for raw output of spikes
    char *buffer;
    const unsigned int length = 4096;
};

} //communication
} // namespace mc
} // namespace nest

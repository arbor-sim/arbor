#pragma once

#include <cstring>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <common_types.hpp>
#include <communication/exporter_interface.hpp>
#include <spike.hpp>
#include <util.hpp>

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy> // TODO: Templating on data type, for now only spike_type
class exporter_spike_file : public exporter_interface<Time, CommunicationPolicy> 
{
public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;
    using communication_policy_type = CommunicationPolicy;

    // Constructor
    // over_write if true will overwrite the specified output file (default = true)
    // output_path  relative or absolute path
    // file_name    will be appended with "_x" with x the rank number
    // file_extention  a seperator will be added automatically
    exporter_spike_file(std::string file_name, std::string path, 
        std::string file_extention, bool over_write=true)
        :
        file_path_(create_output_file_path(
            file_name, path, file_extention, communication_policy_.id()))
    {
        //test if the file exist and depending on over_write throw or delete
        std::ifstream f(file_path_);
        if (f.good()) {
            if (!over_write) {
                std::string error_string("Tried opening file for writing but it exists and over_write is false: " +
                    file_path_);

                throw std::runtime_error(error_string);
            }

            std::remove(file_path_.c_str());
        }
        
        buffer = new char[length];

        file_handle_ = nest::mc::util::make_unique<std::ofstream>(file_path_, std::fstream::app);

        if (!file_handle_->good()) {
            std::string error_string("Could not open file for writing: " + file_path_);
            throw std::runtime_error(error_string);
        }
    }
       
    // Performs the a export of the spikes to file
    // one id and spike time with 4 decimals after the comma on a line space separated
    void do_export(const std::vector<spike_type>& spikes) override
    {
        unsigned current_loc_in_buffer = 0;
        unsigned nr_chars_written = 0;
        char single_value_buffer[20]; // Much to big 

        // Some constants needed for printing
        const char * space = " ";
        const char * endline = "\n";

        for (auto spike : spikes) {
            // Manually convert the id and spike time to chars and use mem copy
            // to insert these in the buffer.

            // First the id as output
            nr_chars_written = std::snprintf(single_value_buffer, 20, "%u", 
                                             spike.source.gid);
            std::memcpy(buffer + current_loc_in_buffer, single_value_buffer,
                        nr_chars_written);
            current_loc_in_buffer += nr_chars_written;

            // Then a space
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
            // endl is only a single char in the actual file!!
            // TODO: WINDOWS? or should we asume newline seperated as the interface
            current_loc_in_buffer += 1;  

            // Check if we are nearing the end of our buffer
            // maximum size of the inserted character in the loop is 2 * 20 + 3
            // So if there are less then 45 chars left in the buffer, write to
            // file. and reset the buffer index to zero
            if (current_loc_in_buffer > length - 45) {
                file_handle_->write(buffer, current_loc_in_buffer);
                current_loc_in_buffer = 0;
            }
        }
        
        // write to buffer at end of the spikes processing
        if (current_loc_in_buffer != 0) {
            file_handle_->write(buffer, current_loc_in_buffer);
            current_loc_in_buffer = 0; // not needed
        }

        file_handle_->flush();

        if (!file_handle_->good()){
            std::string error_string("Error writing data file: " +
                file_path_);

            throw std::runtime_error(error_string);
        }
    }

    // Creates an indexed filename
    static std::string create_output_file_path(std::string file_name, std::string path,
        std::string file_extention, unsigned index)
    {
        std::string file_path = path + file_name + "_" + std::to_string(index) +
                                "." + file_extention;
        // Nest does not produce the indexing for nrank == 0
        // I have the feeling this disrupts consistent output. Id rather
        // always put the zero in. it allows a simpler regex when opening
        // files
        return file_path;
    }

private:   
    std::string file_path_;

    // Handle to opened file handle
    std::unique_ptr<std::ofstream>  file_handle_;
    
    communication_policy_type communication_policy_;

    // Buffer (and size) for raw output of spikes
    char *buffer;
    const unsigned int length = 4096;
};

} //communication
} // namespace mc
} // namespace nest

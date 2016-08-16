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
#include <io/exporter.hpp>
#include <spike.hpp>
#include <util.hpp>

namespace nest {
namespace mc {
namespace io {

template <typename Time, typename CommunicationPolicy>
class exporter_spike_file : public exporter<Time, CommunicationPolicy>
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
    exporter_spike_file(const std::string& file_name, const std::string& path,
        const std::string& file_extention, bool over_write=true)
    {
        auto file_path =
            create_output_file_path(file_name, path, file_extention,
                communication_policy_.id());

        //test if the file exist and depending on over_write throw or delete
        if (!over_write && file_exists(file_path))
        {
            throw std::runtime_error("Tried opening file for writing but it exists and over_write is false: " +
                file_path);
        }

        file_handle_.open(file_path);
    }

    // Performs the a export of the spikes to file
    // one id and spike time with 4 decimals after the comma on a
    // line space separated
    void output(const std::vector<spike_type>& spikes) override
    {
        for (auto spike : spikes) {
            char linebuf[45];
            auto n = std::snprintf(linebuf, sizeof(linebuf), "%u %.4f\n",
                unsigned{spike.source.gid}, float(spike.time));
            file_handle_.write(linebuf, n);
        }
    }

    bool good() const override
    {
        return file_handle_.good();
    }

    // Creates an indexed filename
    static std::string create_output_file_path(const std::string& file_name,
        const std::string& path, const std::string& file_extention,
        unsigned index)
    {
        // Nest does not produce the indexing for nrank == 0
        // I have the feeling this disrupts consistent output. Id rather
        // always put the zero in. it allows a simpler regex when opening
        // files
        return path + file_name + "_" +  std::to_string(index) + "." +
               file_extention;
    }

private:

    bool file_exists(const std::string& file_path)
    {
        std::ifstream fid(file_path);
        return fid.good();
    }

    // Handle to opened file handle
    std::ofstream file_handle_;

    communication_policy_type communication_policy_;

};

} //communication
} // namespace mc
} // namespace nest

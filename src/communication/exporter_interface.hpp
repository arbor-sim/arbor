#pragma once

#include <random>
#include <string>

#include <common_types.hpp>
#include <spike.hpp>

namespace nest {
namespace mc {
namespace communication {
    
// interface for exporters.
// Exposes one virtual functions: 
//    do_export(vector<type>) receiving a vector of parameters to export
template <typename Time, typename CommunicationPolicy>  // TODO: Templating on data type
class exporter_interface {

public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;

    // Performs the export of the data
    virtual void do_export(const std::vector<spike_type>&) = 0;

    // Returns the status of the exporter
    bool good() const;
};

} //communication
} // namespace mc
} // namespace nest

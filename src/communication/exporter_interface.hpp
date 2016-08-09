#pragma once

#include <random>
#include <string>

#include <spike.hpp>
#include <common_types.hpp>

namespace nest {
namespace mc {
namespace communication {

template <typename Time, typename CommunicationPolicy>  // TODO: Templating on data type, for now only spike_type
class exporter_interface {

public:
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;

    // Performs the export of the data, thread safe buffered
    virtual void do_export() = 0;

    // Add data to the internal storage to be exported
    // Does not do the actual export
    virtual void add_data(std::vector<spike_type>) = 0;

    // Internal state is ok
    // Export might encounter problems in a separate thread.
    virtual bool ok() const = 0;

    // TODO: Enum with status strings (might be added to the implemenation)
    // returns a textual explanation of the current error state
    // 
    // virtual string status_description() = 0;
    //virtual string status_id() = 0;
};



} //communication
} // namespace mc
} // namespace nest
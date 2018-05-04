#pragma once

#include <random>
#include <string>

#include <common_types.hpp>
#include <spike.hpp>

namespace arb {
namespace io {

// interface for exporters.
// Exposes one virtual functions:
//    do_export(vector<type>) receiving a vector of parameters to export

class exporter {
public:
    // Performs the export of the data
    virtual void output(const std::vector<spike>&) = 0;

    // Returns the status of the exporter
    virtual bool good() const = 0;
};

} //communication
} // namespace arb

#include <memory>
#include <vector>

#include <arbor/spike.hpp>
#include <arbor/simulation.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "strings.hpp"

namespace pyarb {

} // namespace pyarb

/*
    // Wrap the cell_kind enum type.
    pybind11::enum_<arb::cell_kind>(m, "cell_kind")
        .value("benchmark", arb::cell_kind::benchmark)
        .value("cable1d", arb::cell_kind::cable1d_neuron)
        .value("lif", arb::cell_kind::lif_neuron)
        .value("spike_souce", arb::cell_kind::spike_source);
*/

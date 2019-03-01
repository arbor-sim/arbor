#include <pybind11/pybind11.h>

#include "config.hpp"

namespace pyarb {

void register_config(pybind11::module &m) {

    m.def("config", &config, "Get Arbor's configuration.")
     .def("print_config", [](const pybind11::dict& d){return print_config(d);}, "Print Arbor's configuration.");
}
} // namespace pyarb

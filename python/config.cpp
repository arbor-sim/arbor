#include <iomanip>
#include <ios>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/version.hpp>

namespace pyarb {

// Returns a python dictionary that python users can use to look up
// which options the Arbor library was configured with at compile time.

pybind11::dict config() {
    pybind11::dict dict;
#ifdef ARB_MPI_ENABLED
    dict[pybind11::str("mpi")]     = pybind11::bool_(true);
#else
    dict[pybind11::str("mpi")]     = pybind11::bool_(false);
#endif
#ifdef ARB_WITH_MPI4PY
    dict[pybind11::str("mpi4py")]  = pybind11::bool_(true);
#else
    dict[pybind11::str("mpi4py")]  = pybind11::bool_(false);
#endif
#ifdef ARB_GPU_ENABLED
    dict[pybind11::str("gpu")]     = pybind11::bool_(true);
#else
    dict[pybind11::str("gpu")]     = pybind11::bool_(false);
#endif
    dict[pybind11::str("version")] = pybind11::str(ARB_VERSION);
    return dict;
}

void print_config(const pybind11::dict &d) {
    std::stringstream s;
    s << "Arbor's configuration:\n";

    for (auto x: d) {
        s << "     "
        << std::left << std::setw(7) << x.first << ": "
        << std::right << std::setw(10) << x.second << "\n";
    }

    pybind11::print(s.str());
}

// Register configuration
void register_config(pybind11::module &m) {

    m.def("config", &config, "Get Arbor's configuration.")
     .def("print_config", [](const pybind11::dict& d){return print_config(d);}, "Print Arbor's configuration.");
}
} // namespace pyarb

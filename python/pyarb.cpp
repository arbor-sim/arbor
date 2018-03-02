#include <iostream>

#include <pybind11/pybind11.h>

// Helpful string literals that reduce verbosity.
using namespace pybind11::literals;

PYBIND11_MODULE(pyarb, m) {
    m.attr("__version__") = "dev";

    // This is a placeholder.
    m.def("hello_world", [](){std::cout << "hello world from Pyarb!\n";});
}

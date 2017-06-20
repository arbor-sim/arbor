#include <pybind11/pybind11.h>

int add(int i, int j) {
        return i + j;
}

PYBIND11_MODULE(nmc, m) {
    m.doc() = "pybind11 plugin for NestMC"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers");
}

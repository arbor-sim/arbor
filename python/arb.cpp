#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

int add(int a, int b) {
    return a+b;
}

PYBIND11_MODULE(arb, m) {
    m.def("add", &add, "A function which adds two numbers");
}


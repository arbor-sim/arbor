#include <arbor/simulation.hpp>

#include "context.hpp"
#include "recipe.hpp"
#include "strings.hpp"

#include <pybind11/pybind11.h>

namespace pyarb {

void register_simulation(pybind11::module& m) {
    using namespace pybind11::literals;

    // models
    pybind11::class_<arb::simulation> simulation(m, "simulation", "An Arbor simulation.");

    simulation
        // A custom constructor that wraps a python recipe with
        // arb::py_recipe_shim before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
                [](std::shared_ptr<py_recipe>& r, const arb::domain_decomposition& d, const context_shim& ctx) {
                    return new arb::simulation(py_recipe_shim(r), d, ctx.context);
                }),
                // Release the python gil, so that callbacks into the python
                // recipe don't deadlock.
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Initialize the model described by a recipe, with cells and network distributed\n"
                "according to decomp, and computation resources described by ctx.",
                "recipe"_a, "dom_dec"_a, "context"_a)
        .def("reset", &arb::simulation::reset,
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Reset the model to its initial state to rerun the simulation again.")
        .def("run", &arb::simulation::run,
                pybind11::call_guard<pybind11::gil_scoped_release>(),
                "Advance the model state to time tfinal, in time steps of size dt.",
                "tfinal"_a, "dt"_a);
}

} // namespace pyarb

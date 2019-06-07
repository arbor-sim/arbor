#include <pybind11/pybind11.h>

#include <arbor/simulation.hpp>

#include "context.hpp"
#include "recipe.hpp"

namespace pyarb {

void register_simulation(pybind11::module& m) {
    using namespace pybind11::literals;

    // Simulation
    pybind11::class_<arb::simulation> simulation(m, "simulation",
        "The executable form of a model.\n"
        "A simulation is constructed from a recipe, and then used to update and monitor model state.");
    simulation
        // A custom constructor that wraps a python recipe with
        // arb::py_recipe_shim before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
            [](std::shared_ptr<py_recipe>& recipe, const arb::domain_decomposition& domain_decomp, const context_shim& ctx) {
                return new arb::simulation(py_recipe_shim(recipe), domain_decomp, ctx.context);
            }),
            // Release the python gil, so that callbacks into the python
            // recipe don't deadlock.
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Initialize the model described by a recipe, with cells and network distributed\n"
            "according to domain_decomp, and computational resources described by a context ctx.",
            "recipe"_a, "domain_decomp"_a, "ctx"_a)
        .def("reset", &arb::simulation::reset,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Reset the state of the simulation to its initial state.")
        .def("run", &arb::simulation::run,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Run the simulation from current simulation time to tfinal, with maximum time step size dt.",
            "tfinal"_a, "dt"_a)
        .def("__str__", [](const arb::simulation&){ return "<arbor.simulation>"; })
        .def("__repr__", [](const arb::simulation&){ return "<arbor.simulation>"; });
}

} // namespace pyarb

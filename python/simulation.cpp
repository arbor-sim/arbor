#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>

#include "context.hpp"
#include "error.hpp"
#include "recipe.hpp"

namespace pyarb {

void register_simulation(pybind11::module& m) {
    using namespace pybind11::literals;

    // Simulation
    pybind11::class_<arb::simulation> simulation(m, "simulation",
        "The executable form of a model.\n"
        "A simulation is constructed from a recipe, and then used to update and monitor model state.");
    simulation
        // A custom constructor that wraps a python recipe with arb::py_recipe_shim
        // before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
            [](std::shared_ptr<py_recipe>& rec, const arb::domain_decomposition& decomp, const context_shim& ctx) {
                try {
                    return new arb::simulation(py_recipe_shim(rec), decomp, ctx.context);
                }
                catch (...) {
                    py_reset_and_throw();
                    throw;
                }
            }),
            // Release the python gil, so that callbacks into the python recipe don't deadlock.
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Initialize the model described by a recipe, with cells and network distributed\n"
            "according to the domain decomposition and computational resources described by a context.",
            "recipe"_a, "domain_decomposition"_a, "context"_a)
        .def("reset", &arb::simulation::reset,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Reset the state of the simulation to its initial state.")
        .def("run", &arb::simulation::run,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].",
            "tfinal"_a, "dt"_a=0.025)
        .def("set_binning_policy", &arb::simulation::set_binning_policy,
            "Set the binning policy for event delivery, and the binning time interval if applicable [ms].",
            "policy"_a, "bin_interval"_a)
        .def("__str__",  [](const arb::simulation&){ return "<arbor.simulation>"; })
        .def("__repr__", [](const arb::simulation&){ return "<arbor.simulation>"; });
}

} // namespace pyarb

#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/spike_source_cell.hpp>

#include "event_generator.hpp"

namespace arb {
namespace py {

// py::recipe is the recipe interface that used by Python.
// Calls that return generic types return pybind11::object, to avoid
// having to wrap some C++ types used by the C++ interface (specifically
// util::unique_any, util::any, std::unique_ptr, etc.)
// For example, requests for cell description return pybind11::object, instead
// of util::unique_any used by the C++ recipe interface.
// The py_recipe_shim defined unwraps the python objects, and forwards them
// to the C++ back end.
class recipe {
public:
    recipe() = default;
    virtual ~recipe() {}

    virtual cell_size_type   num_cells() const = 0;
    virtual pybind11::object cell_description(cell_gid_type gid) const = 0;
    virtual cell_kind        kind(cell_gid_type gid) const = 0;
    virtual std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const {
        return {};
    };
    virtual cell_size_type num_sources(cell_gid_type) const {
        return 0;
    };
    virtual cell_size_type num_targets(cell_gid_type) const {
        return 0;
    };
    virtual std::vector<pybind11::object> event_generators(cell_gid_type gid) const {
        auto guard = pybind11::gil_scoped_acquire();
        return {};
    };
};

class recipe_trampoline: public recipe {
public:
    using recipe::recipe;

    cell_size_type num_cells() const override {
        PYBIND11_OVERLOAD_PURE(cell_size_type, recipe, num_cells);
    }

    pybind11::object cell_description(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD_PURE(pybind11::object, recipe, cell_description, gid);
    }

    cell_kind kind(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD_PURE(cell_kind, recipe, kind, gid);
    }

    std::vector<arb::cell_connection> connections_on(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD(std::vector<arb::cell_connection>, recipe, connections_on, gid);
    }

    cell_size_type num_sources(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD(cell_size_type, recipe, num_sources, gid);
    }

    cell_size_type num_targets(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD(cell_size_type, recipe, num_targets, gid);
    }

    std::vector<pybind11::object> event_generators(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD(std::vector<pybind11::object>, recipe, event_generators, gid);
    }
};

// A recipe shim that forwards calls to arb::recipe to a python-side
// arb::py::recipe implementation, and translates the output of the
// arb::py::recipe return values to those used by arb::recipe.
// For example, unwrap cell descriptions stored in PyObject, and rewrap
// in util::unique_any.
class py_recipe_shim: public arb::recipe {
    // pointer to the python recipe implementation
    std::shared_ptr<arb::py::recipe> impl_;

public:
    using recipe::recipe;

    py_recipe_shim(std::shared_ptr<py::recipe> r): impl_(std::move(r)) {}

    cell_size_type num_cells() const override {
        return impl_->num_cells();
    }

    // The py::recipe::cell_decription returns a pybind11::object, that is
    // unwrapped and copied into a util::unique_any.
    util::unique_any get_cell_description(cell_gid_type gid) const override {
        auto guard = pybind11::gil_scoped_acquire();
        pybind11::object o = impl_->cell_description(gid);

        if (pybind11::isinstance<lif_cell>(o)) {
            return util::unique_any(pybind11::cast<lif_cell>(o));
        }
        /*
       else if (pybind11::isinstance<mc_cell>(o)) {
            return util::unique_any(pybind11::cast<mc_cell>(o));
        }
        */

        throw std::runtime_error(
            "Python Arbor recipe.cell_description returned ("
            + std::string(pybind11::str(o))
            + "), which does not describe a known Arbor cell type.");
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return impl_->kind(gid);
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        return impl_->connections_on(gid);
    }

    cell_size_type num_sources(cell_gid_type gid) const override {
        return impl_->num_sources(gid);
    }

    cell_size_type num_targets(cell_gid_type gid) const override {
        return impl_->num_targets(gid);
    }

    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        using namespace std::string_literals;
        using pybind11::isinstance;
        using pybind11::cast;

        // Aquire the GIL because it must be held when calling isinstance and cast.
        auto guard = pybind11::gil_scoped_acquire();

        // Get the python list of arb::py::event_generator from the python front end
        auto pygens = impl_->event_generators(gid);

        std::vector<arb::event_generator> gens;
        gens.reserve(pygens.size());

        for (auto& g: pygens) {
            // check that a valid Python event_generator was passed
            if (!isinstance<arb::py::event_generator>(g)) {
                std::stringstream s;
                s << "recipe supplied an invalid event generator for gid "
                  << gid << ": " << pybind11::str(g);
                throw python_error(s.str());
            }
            // get a reference to the python event_generator
            auto& p = cast<const arb::py::event_generator&>(g);

            // convert the event_generator to an arb::event_generator
            gens.push_back(
                arb::schedule_generator(
                    {gid, p.lid}, p.weight, std::move(p.time_seq)));
        }

        return gens;
    }
};

} // namespace py

} // namespace arb


#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cell.hpp>
#include <recipe.hpp>
#include <rss_cell.hpp>
#include <lif_cell_description.hpp>

#include "event_generator.hpp"

namespace arb {

namespace py {

// py::recipe is the recipe interface that used by Python.
// Calls that return generic types return pybind11::object, to avoid
// having to wrap some C++ types used by the C++ interface (specifically
// util::unique_any, util::any, std::unique_ptr, etc.)
// For example, requests for cell description return pybind11::object, instead
// of util::unique_any used by the C++ recipe interface. The py_recipe_shim defined
// below can unwrap.
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

} // namespace arb::py

// A recipe shim that forwards calls to arb::recipe to a python-side
// arb::py::recipe implementation, and translates the output of the
// arb::py::recipe return values to those used by arb::recipe.
// For example, unwrap cell descriptions stored in PyObject, and rewrap
// in util::unique_any.
class py_recipe_shim: public arb::recipe {
    // pointer to the python recipe implementation
    std::shared_ptr<py::recipe> impl_;

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

        if (pybind11::isinstance<rss_cell>(o)) {
            return util::unique_any(pybind11::cast<rss_cell>(o));
        }
        else if (pybind11::isinstance<cell>(o)) {
            return util::unique_any(pybind11::cast<cell>(o));
        }
        else if (pybind11::isinstance<lif_cell_description>(o)) {
            return util::unique_any(pybind11::cast<lif_cell_description>(o));
        }

        throw std::runtime_error(
            "Python Arbor recipe.cell_description returned a value ("
            + std::string(pybind11::str(o))
            + ") that does not describe a known arbor cell type.");
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

    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        auto guard = pybind11::gil_scoped_acquire();
        using pybind11::isinstance;
        using pybind11::cast;
        auto pygens = impl_->event_generators(gid);
        std::vector<event_generator> gens;
        gens.reserve(pygens.size());

        for (auto& g: pygens) {
            // regular
            if (isinstance<py::regular_generator_desc>(g)) {
                gens.push_back(cast<py::regular_generator_desc>(g).make_cpp());
            }
            // poisson
            else if (isinstance<py::poisson_generator_desc>(g)) {
                gens.push_back(cast<py::poisson_generator_desc>(g).make_cpp());
            }
            // vector backed
            else if (isinstance<py::sequence_generator_desc>(g)) {
                gens.push_back(cast<py::sequence_generator_desc>(g).make_cpp());
            }
            else {
                std::stringstream s;
                s << "A recipe supplied an invalid event generator for gid "
                  << gid << ": " << pybind11::str(g);
                throw std::runtime_error(s.str());
            }
        }
        return gens;
    }
};

} // namespace arb


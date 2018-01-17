#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <recipe.hpp>
#include <rss_cell.hpp>

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
        auto o = impl_->cell_description(gid);
        if (pybind11::isinstance<rss_cell>(o)) {
            return util::unique_any(pybind11::cast<rss_cell>(o));
        }

        throw std::runtime_error(
            "Python Arbor recipe provided a cell_description ("
            + std::string(pybind11::str(o))
            + ") that does not describe a known arbor cell type.");
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return impl_->kind(gid);
    }
};

} // namespace arb


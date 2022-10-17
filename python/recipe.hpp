#pragma once

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/event_generator.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/recipe.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

// pyarb::py_recipe is the recipe interface used by Python.
// Calls that return generic types return pybind11::object, to avoid
// having to wrap some C++ types used by the C++ interface (specifically
// util::unique_any, std::any, std::unique_ptr, etc.)
// For example, requests for cell description return pybind11::object, instead
// of util::unique_any used by the C++ recipe interface.
// The py_recipe_shim unwraps the python objects, and forwards them
// to the C++ back end.

class py_recipe {
public:
    py_recipe() = default;
    virtual ~py_recipe() {}

    virtual arb::cell_size_type num_cells() const = 0;
    virtual pybind11::object cell_description(arb::cell_gid_type gid) const = 0;
    virtual arb::cell_kind cell_kind(arb::cell_gid_type gid) const = 0;

    virtual std::vector<pybind11::object> event_generators(arb::cell_gid_type gid) const {
        return {};
    }
    virtual std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const {
        return {};
    }
    virtual std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type) const {
        return {};
    }
    virtual std::vector<arb::probe_info> probes(arb::cell_gid_type gid) const {
        return {};
    }
    virtual pybind11::object global_properties(arb::cell_kind kind) const {
        return pybind11::none();
    };
};

class py_recipe_trampoline: public py_recipe {
public:
    arb::cell_size_type num_cells() const override {
        PYBIND11_OVERRIDE_PURE(arb::cell_size_type, py_recipe, num_cells);
    }

    pybind11::object cell_description(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE_PURE(pybind11::object, py_recipe, cell_description, gid);
    }

    arb::cell_kind cell_kind(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE_PURE(arb::cell_kind, py_recipe, cell_kind, gid);
    }

    std::vector<pybind11::object> event_generators(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE(std::vector<pybind11::object>, py_recipe, event_generators, gid);
    }

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE(std::vector<arb::cell_connection>, py_recipe, connections_on, gid);
    }

    std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE(std::vector<arb::gap_junction_connection>, py_recipe, gap_junctions_on, gid);
    }

    std::vector<arb::probe_info> probes(arb::cell_gid_type gid) const override {
        PYBIND11_OVERRIDE(std::vector<arb::probe_info>, py_recipe, probes, gid);
    }

    pybind11::object global_properties(arb::cell_kind kind) const override {
        PYBIND11_OVERRIDE(pybind11::object, py_recipe, global_properties, kind);
    }
};

// A recipe shim that holds a pyarb::py_recipe implementation.
// Unwraps/translates python-side output from pyarb::recipe and forwards
// to arb::recipe.
// For example, unwrap cell descriptions stored in PyObject, and rewrap
// in util::unique_any.

class py_recipe_shim: public arb::recipe {
    // pointer to the python recipe implementation
    std::shared_ptr<py_recipe> impl_;

public:
    using recipe::recipe;

    py_recipe_shim(std::shared_ptr<py_recipe> r): impl_(std::move(r)) {}

    const char* msg = "Python error already thrown";

    arb::cell_size_type num_cells() const override {
        return try_catch_pyexception([&](){ return impl_->num_cells(); }, msg);
    }

    // The pyarb::py_recipe::cell_decription method returns a pybind11::object, that is
    // unwrapped and copied into a util::unique_any.
    arb::util::unique_any get_cell_description(arb::cell_gid_type gid) const override;

    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override {
        return try_catch_pyexception([&](){ return impl_->cell_kind(gid); }, msg);
    }

    std::vector<arb::event_generator> event_generators(arb::cell_gid_type gid) const override;

    std::vector<arb::cell_connection> connections_on(arb::cell_gid_type gid) const override {
        return try_catch_pyexception([&](){ return impl_->connections_on(gid); }, msg);
    }

    std::vector<arb::gap_junction_connection> gap_junctions_on(arb::cell_gid_type gid) const override {
        return try_catch_pyexception([&](){ return impl_->gap_junctions_on(gid); }, msg);
    }

    std::vector<arb::probe_info> get_probes(arb::cell_gid_type gid) const override {
        return try_catch_pyexception([&](){ return impl_->probes(gid); }, msg);
    }

    std::any get_global_properties(arb::cell_kind kind) const override;
};

} // namespace pyarb

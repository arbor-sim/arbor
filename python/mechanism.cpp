#include <optional>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include "pybind11/pytypes.h"
#include <pybind11/stl.h>

#include <arbor/cable_cell_param.hpp>
#include <arbor/mechanism.hpp>
#include <arbor/mechcat.hpp>

#include "arbor/mechinfo.hpp"

#include "util.hpp"
#include "conversion.hpp"
#include "strprintf.hpp"

namespace pyarb {

void apply_derive(arb::mechanism_catalogue& m,
            const std::string& name,
            const std::string& parent,
            const std::unordered_map<std::string, double>& globals,
            const std::unordered_map<std::string, std::string>& ions)
{
    if (globals.empty() && ions.empty()) {
        m.derive(name, parent);
        return;
    }
    std::vector<std::pair<std::string, double>> G;
    for (auto& g: globals) {
        G.push_back({g.first, g.second});
    }
    std::vector<std::pair<std::string, std::string>> I;
    for (auto& i: ions) {
        I.push_back({i.first, i.second});
    }
    m.derive(name, parent, G, I);
}

void register_mechanisms(pybind11::module& m) {
    using std::optional;
    using namespace pybind11::literals;

    pybind11::class_<arb::mechanism_field_spec> field_spec(m, "mechanism_field",
        "Basic information about a mechanism field.");
    field_spec
        .def(pybind11::init<const arb::mechanism_field_spec&>())
        .def_readonly("units",   &arb::mechanism_field_spec::units)
        .def_readonly("default", &arb::mechanism_field_spec::default_value)
        .def_readonly("min",     &arb::mechanism_field_spec::lower_bound)
        .def_readonly("max",     &arb::mechanism_field_spec::upper_bound)
        .def("__repr__",
                [](const arb::mechanism_field_spec& spec) {
                    return util::pprintf("{units: '{}', default: {}, min: {}, max: {}}",
                                         (spec.units.size()? spec.units.c_str(): "1"), spec.default_value,
                                         spec.lower_bound, spec.upper_bound); })
        .def("__str__",
                [](const arb::mechanism_field_spec& spec) {
                    return util::pprintf("{units: '{}', default: {}, min: {}, max: {}}",
                                         (spec.units.size()? spec.units.c_str(): "1"), spec.default_value,
                                         spec.lower_bound, spec.upper_bound); });

    pybind11::class_<arb::ion_dependency> ion_dep(m, "ion_dependency",
        "Information about a mechanism's dependence on an ion species.");
    ion_dep
        .def(pybind11::init<const arb::ion_dependency&>())
        .def_readonly("write_int_con", &arb::ion_dependency::write_concentration_int)
        .def_readonly("write_ext_con", &arb::ion_dependency::write_concentration_ext)
        .def_readonly("write_rev_pot", &arb::ion_dependency::write_reversal_potential)
        .def_readonly("read_rev_pot",  &arb::ion_dependency::read_reversal_potential)
        .def("__repr__",
                [](const arb::ion_dependency& dep) {
                    auto tf = [](bool x) {return x? "True": "False";};
                    return util::pprintf("{write_int_con: {}, write_ext_con: {}, write_rev_pot: {}, read_rev_pot: {}}",
                                         tf(dep.write_concentration_int), tf(dep.write_concentration_ext),
                                         tf(dep.write_reversal_potential), tf(dep.read_reversal_potential)); })
        .def("__str__",
                [](const arb::ion_dependency& dep) {
                    auto tf = [](bool x) {return x? "True": "False";};
                    return util::pprintf("{write_int_con: {}, write_ext_con: {}, write_rev_pot: {}, read_rev_pot: {}}",
                                         tf(dep.write_concentration_int), tf(dep.write_concentration_ext),
                                         tf(dep.write_reversal_potential), tf(dep.read_reversal_potential)); })
        ;

    pybind11::class_<arb::mechanism_info> mech_inf(m, "mechanism_info",
        "Meta data about a mechanism's fields and ion dependendencies.");
    mech_inf
        .def(pybind11::init<const arb::mechanism_info&>())
        .def_readonly("globals", &arb::mechanism_info::globals,
            "Global fields have one value common to an instance of a mechanism, are constant in time and set at instantiation.")
        .def_readonly("parameters", &arb::mechanism_info::parameters,
            "Parameter fields may vary across the extent of a mechanism, but are constant in time and set at instantiation.")
        .def_readonly("state", &arb::mechanism_info::state,
            "State fields vary in time and across the extent of a mechanism, and potentially can be sampled at run-time.")
        .def_readonly("ions", &arb::mechanism_info::ions,
            "Ion dependencies.")
        .def_readonly("linear", &arb::mechanism_info::linear,
            "True if a synapse mechanism has linear current contributions so that multiple instances on the same compartment can be coalesced.")
        .def_readonly("post_events", &arb::mechanism_info::post_events,
            "True if a synapse mechanism has a `POST_EVENT` procedure defined.")
        .def_property_readonly("kind",
                [](const arb::mechanism_info& info) {
                    return arb_mechanism_kind_str(info.kind);
                }, "String representation of the kind of the mechanism.")
        .def("__repr__",
                [](const arb::mechanism_info& inf) {
                    return util::pprintf("(arbor.mechanism_info)"); })
        .def("__str__",
                [](const arb::mechanism_info& inf) {
                    return util::pprintf("(arbor.mechanism_info)"); });

    pybind11::class_<arb::mechanism_catalogue> cat(m, "catalogue");

    struct mech_cat_iter_state {
        mech_cat_iter_state(const arb::mechanism_catalogue &cat_, pybind11::object ref_): names(cat_.mechanism_names()), ref(ref_), cat(cat_) {
            std::sort(names.begin(), names.end());
        }
        std::vector<std::string> names;      // cache the names else these will be allocated multiple times
        pybind11::object ref;                // keep a reference to cat lest it dies while we iterate
        const arb::mechanism_catalogue& cat; // to query the C++ object
        size_t idx = 0;                      // where we are in the sequence
        std::string next() {
            if (idx == names.size()) throw pybind11::stop_iteration();
            return names[idx++];
        }
    };

    struct py_mech_cat_key_iterator {
        py_mech_cat_key_iterator(const arb::mechanism_catalogue &cat_, pybind11::object ref_): state{cat_, ref_} { }
        mech_cat_iter_state state;
        std::string next() { return state.next(); }
    };
    struct py_mech_cat_item_iterator {
        py_mech_cat_item_iterator(const arb::mechanism_catalogue &cat_, pybind11::object ref_): state{cat_, ref_} { }
        mech_cat_iter_state state;
        std::tuple<std::string, arb::mechanism_info> next() { auto name = state.next(); return {name, state.cat[name]}; }
    };
    struct py_mech_cat_value_iterator {
        py_mech_cat_value_iterator(const arb::mechanism_catalogue &cat_, pybind11::object ref_): state{cat_, ref_} { }
        mech_cat_iter_state state;
        arb::mechanism_info next() { return state.cat[state.next()]; }
    };

    pybind11::class_<py_mech_cat_key_iterator>(m, "MechCatKeyIterator")
        .def("__iter__", [](py_mech_cat_key_iterator &it) -> py_mech_cat_key_iterator& { return it; })
        .def("__next__", &py_mech_cat_key_iterator::next);

    pybind11::class_<py_mech_cat_value_iterator>(m, "MechCatValueIterator")
        .def("__iter__", [](py_mech_cat_value_iterator &it) -> py_mech_cat_value_iterator& { return it; })
        .def("__next__", &py_mech_cat_value_iterator::next);

    pybind11::class_<py_mech_cat_item_iterator>(m, "MechCatItemIterator")
        .def("__iter__", [](py_mech_cat_item_iterator &it) -> py_mech_cat_item_iterator& { return it; })
        .def("__next__", &py_mech_cat_item_iterator::next);

    cat
        .def(pybind11::init())
        .def(pybind11::init<const arb::mechanism_catalogue&>())
        .def("__contains__", &arb::mechanism_catalogue::has,
             "name"_a, "Is 'name' in the catalogue?")
        .def("__iter__",
             [](pybind11::object cat) { return py_mech_cat_key_iterator(cat.cast<const arb::mechanism_catalogue &>(), cat); },
             "Return an iterator over all mechanism names in this catalogues.")
        .def("keys",
             [](pybind11::object cat) { return py_mech_cat_key_iterator(cat.cast<const arb::mechanism_catalogue &>(), cat); },
             "Return an iterator over all mechanism names in this catalogues.")
        .def("values",
             [](pybind11::object cat) { return py_mech_cat_value_iterator(cat.cast<const arb::mechanism_catalogue &>(), cat); },
             "Return an iterator over all mechanism info values in this catalogues.")
        .def("items",
             [](pybind11::object cat) { return py_mech_cat_item_iterator(cat.cast<const arb::mechanism_catalogue &>(), cat); },
             "Return an iterator over all (name, mechanism) tuples  in this catalogues.")
        .def("is_derived", &arb::mechanism_catalogue::is_derived,
                "name"_a, "Is 'name' a derived mechanism or can it be implicitly derived?")
        .def("__getitem__",
            [](arb::mechanism_catalogue& c, const char* name) {
                try {
                    return c[name];
                }
                catch (...) {
                    throw pybind11::key_error(name);
                }
            })
        .def("extend", &arb::mechanism_catalogue::import,
             "other"_a, "Catalogue to import into self",
             "prefix"_a, "Prefix for names in other",
             "Import another catalogue, possibly with a prefix. Will overwrite in case of name collisions.")
        .def("derive", &apply_derive,
                "name"_a, "parent"_a,
                "globals"_a=std::unordered_map<std::string, double>{},
                "ions"_a=std::unordered_map<std::string, std::string>{})
        .def("__repr__",
                [](const arb::mechanism_catalogue& cat) {
                    return util::pprintf("<arbor.mechanism_catalogue>"); })
        .def("__str__",
                [](const arb::mechanism_catalogue& cat) {
                    return util::pprintf("<arbor.mechanism_catalogue>"); });

    m.def("default_catalogue", [](){return arb::global_default_catalogue();});
    m.def("allen_catalogue", [](){return arb::global_allen_catalogue();});
    m.def("bbp_catalogue", [](){return arb::global_bbp_catalogue();});
    m.def("stochastic_catalogue", [](){return arb::global_stochastic_catalogue();});
    m.def("load_catalogue", [](pybind11::object fn) { return arb::load_catalogue(util::to_string(fn)); });

    // arb::mechanism_desc
    // For specifying a mechanism in the cable_cell interface.

    pybind11::class_<arb::mechanism_desc> mechanism_desc(m, "mechanism");
    mechanism_desc
        .def(pybind11::init([](const char* name) {return arb::mechanism_desc{name};}),
            "name"_a, "The name of the mechanism"
        )
        // allow construction of a description with parameters provided in a dictionary:
        //      mech = arbor.mechanism('mech_name', {'param1': 1.2, 'param2': 3.14})
        .def(pybind11::init(
            [](const char* name, std::unordered_map<std::string, double> params) {
                arb::mechanism_desc md(name);
                for (const auto& p: params) md.set(p.first, p.second);
                return md;
            }),
            "name"_a, "The name of the mechanism",
            "params"_a, "A dictionary of parameter values, with parameter name as key.",
            "Example usage setting parameters:\n"
            "  m = arbor.mechanism('expsyn', {'tau': 1.4})\n"
            "will create parameters for the 'expsyn' mechanism, with the provided value\n"
            "for 'tau' overrides the default. If a parameter is not set, the default\n"
            "(as defined in NMODL) is used.\n\n"
            "Example overriding a global parameter:\n"
            "  m = arbor.mechanism('nernst/R=8.3145,F=96485')")
        .def("set",
            [](arb::mechanism_desc& md, std::string name, double value) {
                md.set(name, value);
            },
            "name"_a, "value"_a, "Set parameter value.")
        .def_property_readonly("name",
            [](const arb::mechanism_desc& md) {
                return md.name();
            },
            "The name of the mechanism.")
        .def_property_readonly("values",
            [](const arb::mechanism_desc& md) {
                return md.values();
            }, "A dictionary of parameter values with parameter name as key.")
        .def("__repr__",
                [](const arb::mechanism_desc& md) {
                    return util::pprintf("<arbor.mechanism: name '{}', parameters {}>", md.name(), util::dictionary_csv(md.values())); })
        .def("__str__",
                [](const arb::mechanism_desc& md) {
                    return util::pprintf("('{}' {})", md.name(), util::dictionary_csv(md.values())); });

}

} // namespace pyarb

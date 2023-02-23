#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arborio/label_parse.hpp>

#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/recipe.hpp>
#include <arbor/sampling.hpp>
#include <arbor/util/any_ptr.hpp>

#include "pyarb.hpp"
#include "strprintf.hpp"

using arb::util::any_cast;
using arb::util::any_ptr;
namespace py = pybind11;

namespace pyarb {

// Generic recorder classes for array-output sample data, corresponding
// to cable_cell scalar- and vector-valued probes.

template <typename Meta>
struct recorder_base: sample_recorder {
    // Return stride-column array: first column is time, remainder correspond to sample.

    py::object samples() const override {
        auto n_record = std::ptrdiff_t(sample_raw_.size()/stride_);
        return py::array_t<double>(
                    std::vector<std::ptrdiff_t>{n_record, stride_},
                    sample_raw_.data());
    }

    py::object meta() const override {
        return py::cast(meta_);
    }

    void reset() override {
        sample_raw_.clear();
    }

protected:
    Meta meta_;
    std::vector<double> sample_raw_;
    std::ptrdiff_t stride_;

    recorder_base(const Meta* meta_ptr, std::ptrdiff_t width):
        meta_(*meta_ptr), stride_(1+width)
    {}
};

template <typename Meta>
struct recorder_cable_scalar: recorder_base<Meta> {
    using recorder_base<Meta>::sample_raw_;

    void record(any_ptr, std::size_t n_sample, const arb::sample_record* records) override {
        for (std::size_t i = 0; i<n_sample; ++i) {
            if (auto* v_ptr =any_cast<const double*>(records[i].data)) {
                sample_raw_.push_back(records[i].time);
                sample_raw_.push_back(*v_ptr);
            }
            else {
                throw arb::arbor_internal_error("unexpected sample type");
            }
        }
    }

protected:
    recorder_cable_scalar(const Meta* meta_ptr): recorder_base<Meta>(meta_ptr, 1) {}
};

struct recorder_lif: recorder_base<arb::lif_probe_metadata> {
    using recorder_base<arb::lif_probe_metadata>::sample_raw_;

    void record(any_ptr, std::size_t n_sample, const arb::sample_record* records) override {
        for (std::size_t i = 0; i<n_sample; ++i) {
            if (auto* v_ptr = any_cast<double*>(records[i].data)) {
                sample_raw_.push_back(records[i].time);
                sample_raw_.push_back(*v_ptr);
            }
            else {
                std::string ty = records[i].data.type().name();
                throw arb::arbor_internal_error("LIF recorder: unexpected sample type " + ty);
            }
        }
    }

    recorder_lif(const arb::lif_probe_metadata* meta_ptr): recorder_base<arb::lif_probe_metadata>(meta_ptr, 1) {}
};


template <typename Meta>
struct recorder_cable_vector: recorder_base<Meta> {
    using recorder_base<Meta>::sample_raw_;

    void record(any_ptr, std::size_t n_sample, const arb::sample_record* records) override {
        for (std::size_t i = 0; i<n_sample; ++i) {
            if (auto* v_ptr = any_cast<const arb::cable_sample_range*>(records[i].data)) {
                sample_raw_.push_back(records[i].time);
                sample_raw_.insert(sample_raw_.end(), v_ptr->first, v_ptr->second);
            }
            else {
                throw arb::arbor_internal_error("unexpected sample type");
            }
        }
    }

protected:
    recorder_cable_vector(const Meta* meta_ptr, std::ptrdiff_t width):
        recorder_base<Meta>(meta_ptr, width) {}
};

// Specific recorder classes:

struct recorder_cable_scalar_mlocation: recorder_cable_scalar<arb::mlocation> {
    explicit recorder_cable_scalar_mlocation(const arb::mlocation* meta_ptr):
        recorder_cable_scalar(meta_ptr) {}
};

struct recorder_cable_scalar_point_info: recorder_cable_scalar<arb::cable_probe_point_info> {
    explicit recorder_cable_scalar_point_info(const arb::cable_probe_point_info* meta_ptr):
        recorder_cable_scalar(meta_ptr) {}
};

struct recorder_cable_vector_mcable: recorder_cable_vector<arb::mcable_list> {
    explicit recorder_cable_vector_mcable(const arb::mcable_list* meta_ptr):
        recorder_cable_vector(meta_ptr, std::ptrdiff_t(meta_ptr->size())) {}
};

struct recorder_cable_vector_point_info: recorder_cable_vector<std::vector<arb::cable_probe_point_info>> {
    explicit recorder_cable_vector_point_info(const std::vector<arb::cable_probe_point_info>* meta_ptr):
        recorder_cable_vector(meta_ptr, std::ptrdiff_t(meta_ptr->size())) {}
};

// Helper for registering sample recorder factories and (trivial) metadata conversions.

template <typename Meta, typename Recorder>
void register_probe_meta_maps(pyarb_global_ptr g) {
    g->recorder_factories.assign<Meta>(
        [](any_ptr meta_ptr) -> std::unique_ptr<sample_recorder> {
            return std::unique_ptr<Recorder>(new Recorder(any_cast<const Meta*>(meta_ptr)));
        });

    g->probe_meta_converters.assign<Meta>(
        [](any_ptr meta_ptr) -> py::object {
            return py::cast(*any_cast<const Meta*>(meta_ptr));
        });
}



// Wrapper functions around cable_cell probe types that return arb::probe_info values:
// (Probe tag value is implicitly left at zero.)

arb::probe_info cable_probe_membrane_voltage(const char* where) {
    return arb::cable_probe_membrane_voltage{arborio::parse_locset_expression(where).unwrap()};
}

arb::probe_info cable_probe_membrane_voltage_cell() {
    return arb::cable_probe_membrane_voltage_cell{};
}

arb::probe_info cable_probe_axial_current(const char* where) {
    return arb::cable_probe_axial_current{arborio::parse_locset_expression(where).unwrap()};
}

arb::probe_info cable_probe_total_ion_current_density(const char* where) {
    return arb::cable_probe_total_ion_current_density{arborio::parse_locset_expression(where).unwrap()};
}

arb::probe_info cable_probe_total_ion_current_cell() {
    return arb::cable_probe_total_ion_current_cell{};
}

arb::probe_info cable_probe_total_current_cell() {
    return arb::cable_probe_total_current_cell{};
}

arb::probe_info cable_probe_stimulus_current_cell() {
    return arb::cable_probe_stimulus_current_cell{};
}

arb::probe_info cable_probe_density_state(const char* where, const char* mechanism, const char* state) {
    return arb::cable_probe_density_state{arborio::parse_locset_expression(where).unwrap(), mechanism, state};
};

arb::probe_info cable_probe_density_state_cell(const char* mechanism, const char* state) {
    return arb::cable_probe_density_state_cell{mechanism, state};
};

arb::probe_info cable_probe_point_state(arb::cell_lid_type target, const char* mechanism, const char* state) {
    return arb::cable_probe_point_state{target, mechanism, state};
}

arb::probe_info cable_probe_point_state_cell(const char* mechanism, const char* state_var) {
    return arb::cable_probe_point_state_cell{mechanism, state_var};
}

arb::probe_info cable_probe_ion_current_density(const char* where, const char* ion) {
    return arb::cable_probe_ion_current_density{arborio::parse_locset_expression(where).unwrap(), ion};
}

arb::probe_info cable_probe_ion_current_cell(const char* ion) {
    return arb::cable_probe_ion_current_cell{ion};
}

arb::probe_info cable_probe_ion_int_concentration(const char* where, const char* ion) {
    return arb::cable_probe_ion_int_concentration{arborio::parse_locset_expression(where).unwrap(), ion};
}

arb::probe_info cable_probe_ion_int_concentration_cell(const char* ion) {
    return arb::cable_probe_ion_int_concentration_cell{ion};
}

arb::probe_info cable_probe_ion_diff_concentration(const char* where, const char* ion) {
    return arb::cable_probe_ion_diff_concentration{arborio::parse_locset_expression(where).unwrap(), ion};
}

arb::probe_info cable_probe_ion_diff_concentration_cell(const char* ion) {
    return arb::cable_probe_ion_diff_concentration_cell{ion};
}

arb::probe_info cable_probe_ion_ext_concentration(const char* where, const char* ion) {
    return arb::cable_probe_ion_ext_concentration{arborio::parse_locset_expression(where).unwrap(), ion};
}

arb::probe_info cable_probe_ion_ext_concentration_cell(const char* ion) {
    return arb::cable_probe_ion_ext_concentration_cell{ion};
}

// LIF cell probes
arb::probe_info lif_probe_voltage() {
    return arb::lif_probe_voltage{};
}


// Add wrappers to module, recorder factories to global data.

void register_cable_probes(pybind11::module& m, pyarb_global_ptr global_ptr) {
    using namespace pybind11::literals;
    using util::pprintf;

    // Probe metadata wrappers:

    py::class_<arb::lif_probe_metadata> lif_probe_metadata(m, "lif_probe_metadata",
        "Probe metadata associated with a LIF cell probe.");

    py::class_<arb::cable_probe_point_info> cable_probe_point_info(m, "cable_probe_point_info",
        "Probe metadata associated with a cable cell probe for point process state.");

    cable_probe_point_info
        .def_readwrite("target",   &arb::cable_probe_point_info::target,
            "The target index of the point process instance on the cell.")
        .def_readwrite("multiplicity", &arb::cable_probe_point_info::multiplicity,
            "Number of coalesced point processes (linear synapses) associated with this instance.")
        .def_readwrite("location", &arb::cable_probe_point_info::loc,
            "Location of point process instance on cell.")
        .def("__str__", [](arb::cable_probe_point_info m) {
            return pprintf("<arbor.cable_probe_point_info: target {}, multiplicity {}, location {}>", m.target, m.multiplicity, m.loc);})
        .def("__repr__",[](arb::cable_probe_point_info m) {
            return pprintf("<arbor.cable_probe_point_info: target {}, multiplicity {}, location {}>", m.target, m.multiplicity, m.loc);});

    // Probe address constructors:

    m.def("lif_probe_voltage", &lif_probe_voltage,
        "Probe specification for LIF cell membrane voltage.");

    m.def("cable_probe_membrane_voltage", &cable_probe_membrane_voltage,
        "Probe specification for cable cell membrane voltage interpolated at points in a location set.",
        "where"_a);

    m.def("cable_probe_membrane_voltage_cell", &cable_probe_membrane_voltage_cell,
        "Probe specification for cable cell membrane voltage associated with each cable in each CV.");

    m.def("cable_probe_axial_current", &cable_probe_axial_current,
        "Probe specification for cable cell axial current at points in a location set.",
        "where"_a);

    m.def("cable_probe_total_ion_current_density", &cable_probe_total_ion_current_density,
        "Probe specification for cable cell total transmembrane current density excluding capacitive currents at points in a location set.",
        "where"_a);

    m.def("cable_probe_total_ion_current_cell", &cable_probe_total_ion_current_cell,
        "Probe specification for cable cell total transmembrane current excluding capacitive currents for each cable in each CV.");

    m.def("cable_probe_total_current_cell", &cable_probe_total_current_cell,
        "Probe specification for cable cell total transmembrane current for each cable in each CV.");

    m.def("cable_probe_stimulus_current_cell", &cable_probe_stimulus_current_cell,
        "Probe specification for cable cell stimulus current across each cable in each CV.");

    m.def("cable_probe_density_state", &cable_probe_density_state,
        "Probe specification for a cable cell density mechanism state variable at points in a location set.",
        "where"_a, "mechanism"_a, "state"_a);

    m.def("cable_probe_density_state_cell", &cable_probe_density_state_cell,
        "Probe specification for a cable cell density mechanism state variable on each cable in each CV where defined.",
        "mechanism"_a, "state"_a);

    m.def("cable_probe_point_state", &cable_probe_point_state,
        "Probe specification for a cable cell point mechanism state variable value at a given target index.",
        "target"_a, "mechanism"_a, "state"_a);

    m.def("cable_probe_point_state_cell", &cable_probe_point_state_cell,
        "Probe specification for a cable cell point mechanism state variable value at every corresponding target.",
        "mechanism"_a, "state"_a);

    m.def("cable_probe_ion_current_density", &cable_probe_ion_current_density,
        "Probe specification for cable cell ionic current density at points in a location set.",
        "where"_a, "ion"_a);

    m.def("cable_probe_ion_current_cell", &cable_probe_ion_current_cell,
        "Probe specification for cable cell ionic current across each cable in each CV.",
        "ion"_a);

    m.def("cable_probe_ion_int_concentration", &cable_probe_ion_int_concentration,
        "Probe specification for cable cell internal ionic concentration at points in a location set.",
        "where"_a, "ion"_a);

    m.def("cable_probe_ion_int_concentration_cell", &cable_probe_ion_int_concentration_cell,
        "Probe specification for cable cell internal ionic concentration for each cable in each CV.",
        "ion"_a);

    m.def("cable_probe_ion_diff_concentration", &cable_probe_ion_diff_concentration,
        "Probe specification for cable cell diffusive ionic concentration at points in a location set.",
        "where"_a, "ion"_a);

    m.def("cable_probe_ion_diff_concentration_cell", &cable_probe_ion_diff_concentration_cell,
        "Probe specification for cable cell diffusive ionic concentration for each cable in each CV.",
        "ion"_a);

    m.def("cable_probe_ion_ext_concentration", &cable_probe_ion_ext_concentration,
        "Probe specification for cable cell external ionic concentration at points in a location set.",
        "where"_a, "ion"_a);

    m.def("cable_probe_ion_ext_concentration_cell", &cable_probe_ion_ext_concentration_cell,
        "Probe specification for cable cell external ionic concentration for each cable in each CV.",
        "ion"_a);

    // Add probe metadata to maps for converters and recorders.

    register_probe_meta_maps<arb::mlocation, recorder_cable_scalar_mlocation>(global_ptr);
    register_probe_meta_maps<arb::cable_probe_point_info, recorder_cable_scalar_point_info>(global_ptr);
    register_probe_meta_maps<arb::mcable_list, recorder_cable_vector_mcable>(global_ptr);
    register_probe_meta_maps<std::vector<arb::cable_probe_point_info>, recorder_cable_vector_point_info>(global_ptr);
    register_probe_meta_maps<arb::lif_probe_metadata, recorder_lif>(global_ptr);
}

} // namespace pyarb

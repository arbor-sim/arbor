#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/util/optional.hpp>

#include <memory>
#include <unordered_map>
#include <string>

namespace arb {

// Specialized arbor exception for errors in cell building.

struct cable_cell_error: arbor_exception {
    cable_cell_error(const std::string& what):
        arbor_exception("cable_cell: "+what) {}
};

// Ion inital concentration and reversal potential
// parameters, as used in cable_cell_parameter_set,
// and set locally via painting initial_ion_data
// (see below).

struct cable_cell_ion_data {
    double init_int_concentration = NAN;
    double init_ext_concentration = NAN;
    double init_reversal_potential = NAN;
};

// Current clamp description for stimulus specification.
struct i_clamp {
    using value_type = double;

    value_type delay = 0;      // [ms]
    value_type duration = 0;   // [ms]
    value_type amplitude = 0;  // [nA]

    i_clamp() = default;

    i_clamp(value_type delay, value_type duration, value_type amplitude):
        delay(delay), duration(duration), amplitude(amplitude)
    {}
};

// Threshold detector description.
struct threshold_detector {
    double threshold;
};

// Tag type for dispatching cable_cell::place() calls that add gap junction sites.
struct gap_junction_site {};

// Setter types for painting physical and ion parameters or setting
// cell-wide default:

struct init_membrane_potential {
    double value = NAN; // [mV]
};

struct temperature_K {
    double value = NAN; // [K]
};

struct axial_resistivity {
    double value = NAN; // [[Ω·cm]
};

struct membrane_capacitance {
    double value = NAN; // [F/m²]
};

// Mechanism description, viz. mechanism name and
// (non-global) parameter settings. Used to assign
// density and point mechanisms to segments and
// reversal potential computations to cells.

struct mechanism_desc {
    struct field_proxy {
        mechanism_desc* m;
        std::string key;

        field_proxy& operator=(double v) {
            m->set(key, v);
            return *this;
        }

        operator double() const {
            return m->get(key);
        }
    };

    // implicit
    mechanism_desc(std::string name): name_(std::move(name)) {}
    mechanism_desc(const char* name): name_(name) {}

    mechanism_desc() = default;
    mechanism_desc(const mechanism_desc&) = default;
    mechanism_desc(mechanism_desc&&) = default;

    mechanism_desc& operator=(const mechanism_desc&) = default;
    mechanism_desc& operator=(mechanism_desc&&) = default;

    mechanism_desc& set(const std::string& key, double value) {
        param_[key] = value;
        return *this;
    }

    double operator[](const std::string& key) const {
        return get(key);
    }

    field_proxy operator[](const std::string& key) {
        return {this, key};
    }

    double get(const std::string& key) const {
        auto i = param_.find(key);
        if (i==param_.end()) {
            throw std::out_of_range("no field "+key+" set");
        }
        return i->second;
    }

    const std::unordered_map<std::string, double>& values() const {
        return param_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::unordered_map<std::string, double> param_;
};

struct initial_ion_data {
    std::string ion;
    cable_cell_ion_data initial;
};

struct ion_reversal_potential_method {
    std::string ion;
    mechanism_desc method;
};

// Cable cell ion and electrical defaults.

// Parameters can be given as per-cell and global defaults via
// cable_cell::default_parameters and cable_cell_global_properties::default_parameters
// respectively.
//
// With the exception of `reversal_potential_method`, these properties can
// be set locally witihin a cell using the `cable_cell::paint()`, and the
// cell defaults can be individually set with `cable_cell:set_default()`.

struct cable_cell_parameter_set {
    util::optional<double> init_membrane_potential; // [mV]
    util::optional<double> temperature_K;           // [K]
    util::optional<double> axial_resistivity;       // [Ω·cm]
    util::optional<double> membrane_capacitance;    // [F/m²]

    std::unordered_map<std::string, cable_cell_ion_data> ion_data;
    std::unordered_map<std::string, mechanism_desc> reversal_potential_method;

    util::optional<cv_policy> discretization;
};

extern cable_cell_parameter_set neuron_parameter_defaults;

// Global cable cell data.

struct cable_cell_global_properties {
    const mechanism_catalogue* catalogue = &global_default_catalogue();

    // If >0, check membrane voltage magnitude is less than limit
    // during integration.
    double membrane_voltage_limit_mV = 0;

    // True => combine linear synapses for performance.
    bool coalesce_synapses = true;

    // Available ion species, together with charge.
    std::unordered_map<std::string, int> ion_species = {
        {"na", 1},
        {"k", 1},
        {"ca", 2}
    };

    cable_cell_parameter_set default_parameters;

    // Convenience methods for adding a new ion together with default ion values.
    void add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, double init_revpot) {
        ion_species[ion_name] = charge;

        auto &ion_data = default_parameters.ion_data[ion_name];
        ion_data.init_int_concentration = init_iconc;
        ion_data.init_ext_concentration = init_econc;
        ion_data.init_reversal_potential = init_revpot;
    }

    void add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, mechanism_desc revpot_mechanism) {
        add_ion(ion_name, charge, init_iconc, init_econc, 0);
        default_parameters.reversal_potential_method[ion_name] = std::move(revpot_mechanism);
    }
};

// Throw cable_cell_error if any default parameters are left unspecified,
// or if the supplied ion data is incomplete.
void check_global_properties(const cable_cell_global_properties&);

} // namespace arb

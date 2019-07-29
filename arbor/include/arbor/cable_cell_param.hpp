#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/optional.hpp>

#include <unordered_map>
#include <string>

namespace arb {

// Specialized arbor exception for errors in cell building.

struct cable_cell_error: arbor_exception {
    cable_cell_error(const std::string& what):
        arbor_exception("cable_cell: "+what) {}
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

// Cable cell ion and electrical defaults.
//
// Parameters can be overridden with `cable_cell_local_parameter_set`
// on unbranched segments within a cell; per-cell and global defaults
// use `cable_cell_parameter_set`, which extends the parameter set
// to supply per-cell or global ion reversal potential calculation
// mechanisms.

struct cable_cell_ion_data {
    double init_int_concentration = NAN;
    double init_ext_concentration = NAN;
    double init_reversal_potential = NAN;
};

struct cable_cell_local_parameter_set {
    std::unordered_map<std::string, cable_cell_ion_data> ion_data;
    util::optional<double> init_membrane_potential; // [mV]
    util::optional<double> temperature_K;           // [K]
    util::optional<double> axial_resistivity;       // [Ω·cm]
    util::optional<double> membrane_capacitance;    // [F/m²]
};

struct cable_cell_parameter_set: public cable_cell_local_parameter_set {
    std::unordered_map<std::string, mechanism_desc> reversal_potential_method;

    // We'll need something like this until C++17, for sane initialization syntax.
    cable_cell_parameter_set() = default;
    cable_cell_parameter_set(cable_cell_local_parameter_set p, std::unordered_map<std::string, mechanism_desc> m = {}):
        cable_cell_local_parameter_set(std::move(p)), reversal_potential_method(std::move(m))
    {}
};

extern cable_cell_local_parameter_set neuron_parameter_defaults;

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

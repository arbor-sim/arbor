#pragma once

#include <cmath>
#include <memory>
#include <optional>
#include <unordered_map>
#include <string>
#include <variant>

#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/locset.hpp>

namespace arb {

// Specialized arbor exception for errors in cell building.

struct cable_cell_error: arbor_exception {
    cable_cell_error(const std::string& what):
        arbor_exception("cable_cell: "+what) {}
};

// Ion inital concentration and reversal potential
// parameters, as used in cable_cell_parameter_set,
// and set locally via painting init_int_concentration,
// init_ext_concentration and init_reversal_potential
// separately (see below).

struct cable_cell_ion_data {
    std::optional<double> init_int_concentration;
    std::optional<double> init_ext_concentration;
    std::optional<double> init_reversal_potential;
};

// Clamp current is described by a sine wave with amplitude governed by a
// piecewise linear envelope. A frequency of zero indicates that the current is
// simply that given by the envelope.
//
// The envelope is given by a series of envelope_point values:
// * The time points must be monotonically increasing.
// * Onset and initial amplitude is given by the first point.
// * The amplitude for time after the last time point is that of the last
//   amplitude point; an explicit zero amplitude point must be provided if the
//   envelope is intended to have finite support.
//
// Periodic envelopes are not supported, but may well be a feature worth
// considering in the future.

struct i_clamp {
    struct envelope_point {
        double t;         // [ms]
        double amplitude; // [nA]
    };

    std::vector<envelope_point> envelope;
    double frequency = 0; // [kHz] 0 => constant
    double phase = 0;     // [rad]

    // A default constructed i_clamp, with empty envelope, describes
    // a trivial stimulus, providing no current at all.
    i_clamp() = default;

    // The simple constructor describes a constant amplitude stimulus starting from t=0.
    explicit i_clamp(double amplitude, double frequency = 0, double phase = 0):
        envelope({{0., amplitude}}),
        frequency(frequency),
        phase(phase)
    {}

    // Describe a stimulus by envelope and frequency.
    explicit i_clamp(std::vector<envelope_point> envelope, double frequency = 0, double phase = 0):
        envelope(std::move(envelope)),
        frequency(frequency),
        phase(phase)
    {}

    // A 'box' stimulus with fixed onset time, duration, and constant amplitude.
    static i_clamp box(double onset, double duration, double amplitude, double frequency = 0, double phase = 0) {
        return i_clamp({{onset, amplitude}, {onset+duration, amplitude}, {onset+duration, 0.}}, frequency, phase);
    }

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
    double value = NAN; // [Ω·cm]
};

struct membrane_capacitance {
    double value = NAN; // [F/m²]
};

struct init_int_concentration {
    std::string ion = "";
    double value = NAN; // [mM]
};

struct init_ext_concentration {
    std::string ion = "";
    double value = NAN; // [mM]
};

struct init_reversal_potential {
    std::string ion = "";
    double value = NAN; // [mV]
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

struct ion_reversal_potential_method {
    std::string ion;
    mechanism_desc method;
};

using paintable =
    std::variant<init_membrane_potential,
                 axial_resistivity,
                 temperature_K,
                 membrane_capacitance,
                 init_int_concentration,
                 init_ext_concentration,
                 init_reversal_potential,
                 mechanism_desc>;

using placeable =
    std::variant<mechanism_desc,
                 i_clamp,
                 threshold_detector,
                 gap_junction_site>;

using defaultable =
    std::variant<init_membrane_potential,
                 axial_resistivity,
                 temperature_K,
                 membrane_capacitance,
                 init_int_concentration,
                 init_ext_concentration,
                 init_reversal_potential,
                 ion_reversal_potential_method,
                 cv_policy>;

// Cable cell ion and electrical defaults.

// Parameters can be given as per-cell and global defaults via
// cable_cell::default_parameters and cable_cell_global_properties::default_parameters
// respectively.
//
// With the exception of `reversal_potential_method`, these properties can
// be set locally witihin a cell using the `cable_cell::paint()`, and the
// cell defaults can be individually set with `cable_cell:set_default()`.

struct cable_cell_parameter_set {
    std::optional<double> init_membrane_potential; // [mV]
    std::optional<double> temperature_K;           // [K]
    std::optional<double> axial_resistivity;       // [Ω·cm]
    std::optional<double> membrane_capacitance;    // [F/m²]

    std::unordered_map<std::string, cable_cell_ion_data> ion_data;
    std::unordered_map<std::string, mechanism_desc> reversal_potential_method;

    std::optional<cv_policy> discretization;

    std::vector<defaultable> serialize() const;
};

// A flat description of defaults, paintings and placings that
// are to be applied to a morphology in a cable_cell.
class decor {
    std::vector<std::pair<region, paintable>> paintings_;
    std::vector<std::tuple<locset, placeable, cell_tag_type>> placements_;
    cable_cell_parameter_set defaults_;

public:
    const auto& paintings()  const {return paintings_;  }
    const auto& placements() const {return placements_; }
    const auto& defaults()   const {return defaults_;   }

    void paint(region, paintable);
    void place(locset, placeable, cell_tag_type);
    void set_default(defaultable);
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

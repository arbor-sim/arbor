#pragma once

#include <cmath>
#include <optional>
#include <unordered_map>
#include <string>
#include <variant>

#include <arbor/export.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/iexpr.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/units.hpp>

namespace arb {

namespace U = arb::units;

// Specialized arbor exception for errors in cell building.

struct ARB_SYMBOL_VISIBLE cable_cell_error: arbor_exception {
    cable_cell_error(const std::string& what):
        arbor_exception("cable_cell: " + what) {}
};

// Ion inital concentration and reversal potential
// parameters, as used in cable_cell_parameter_set,
// and set locally via painting init_int_concentration,
// init_ext_concentration and init_reversal_potential
// separately (see below).

struct cable_cell_ion_data {
    std::optional<double> init_int_concentration;  // mM
    std::optional<double> init_ext_concentration;  // mM
    std::optional<double> init_reversal_potential; // mV
    std::optional<double> diffusivity;             // m²/s
};

/**
 * Current clamp; described by a sine wave with amplitude governed by a
 * piecewise linear envelope. A frequency of zero indicates that the current is
 * simply that given by the envelope.
 *
 * The envelope is given by a series of envelope_point values:
 * * The time points must be monotonically increasing.
 * * Onset and initial amplitude is given by the first point.
 * * The amplitude for time after the last time point is that of the last
 *   amplitude point; an explicit zero amplitude point must be provided if the
 *   envelope is intended to have finite support.
 *
 * Periodic envelopes are not supported, but may well be a feature worth
 * considering in the future.
 */
struct ARB_SYMBOL_VISIBLE i_clamp {
    struct envelope_point {
        /**
         * Current at point in time
         *
         * @param t, must be convertible to time
         * @param amplitude must be convertible to current
         */
        envelope_point(const U::quantity& time,
                       const U::quantity& current):
            t(time.value_as(U::ms)),
            amplitude(current.value_as(U::nA)) {

            if (std::isnan(t)) throw std::domain_error{"Time must be finite and convertible to ms."};
            if (std::isnan(amplitude)) throw std::domain_error{"Amplitude must be finite and convertible to nA."};
    }
        double t;         // [ms]
        double amplitude; // [nA]
    };

    std::vector<envelope_point> envelope;
    double frequency = 0; // [kHz] 0 => constant
    double phase = 0;     // [rad]

    // A default constructed i_clamp, with empty envelope, describes
    // a trivial stimulus, providing no current at all.
    i_clamp() = default;

    /**
     *  Constant amplitude stimulus starting at t = 0.
     *
     * @param amplitude must be convertible to current
     * @param frequency, must be convertible to frequency; gives a sine current if not zero
     * @param frequency, must be convertible to radians, phase shift of sine.
     */

    explicit i_clamp(const U::quantity& amplitude,
                     const U::quantity& frequency = 0*U::kHz,
                     const U::quantity& phase = 0*U::rad):
        i_clamp{{{0.0*U::ms, amplitude}}, frequency, phase}
    {}

    // Describe a stimulus by envelope and frequency.
    explicit i_clamp(std::vector<envelope_point> envelope,
                     const U::quantity& f = 0*U::kHz,
                     const U::quantity& phi = 0*U::rad):
        envelope(std::move(envelope)),
        frequency(f.value_as(U::kHz)),
        phase(phi.value_as(U::rad))
    {
        if (std::isnan(frequency)) throw std::domain_error{"Frequency must be finite and convertible to kHz."};
        if (std::isnan(phase)) throw std::domain_error{"Phase must be finite and convertible to rad."};
    }

    // A 'box' stimulus with fixed onset time, duration, and constant amplitude.
    static i_clamp box(const U::quantity& onset,
                       const U::quantity& duration,
                       const U::quantity& amplitude,
                       const U::quantity& frequency =  0*U::kHz,
                       const U::quantity& phase = 0*U::rad) {
        return i_clamp({{onset, amplitude}, {onset+duration, amplitude}, {onset+duration, 0.*U::nA}},
                       frequency,
                       phase);
    }
};

// Threshold detector description.
struct ARB_SYMBOL_VISIBLE threshold_detector {
    threshold_detector(const U::quantity& m): threshold(m.value_as(U::mV)) {
        if (std::isnan(threshold)) throw std::domain_error{"Threshold must be finite and in [mV]."};
    }
    static threshold_detector from_raw_millivolts(double v) { return {v*U::mV}; }
    double threshold; // [mV]
};

// Setter types for painting physical and ion parameters or setting
// cell-wide default:

struct ARB_SYMBOL_VISIBLE init_membrane_potential {
    double value = NAN;      // [mV]
    iexpr scale = 1;         // [1]

    init_membrane_potential() = default;
    init_membrane_potential(const U::quantity& m, iexpr scale=1):
      value(m.value_as(U::mV)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [mV]."};
    }
};


struct ARB_SYMBOL_VISIBLE temperature {
    double value = NAN;      // [K]
    iexpr scale = 1;         // [1]

    temperature() = default;
    temperature(const U::quantity& m, iexpr scale=1):
      value(m.value_as(U::Kelvin)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [K]."};
    }
};

struct ARB_SYMBOL_VISIBLE axial_resistivity {
    double value = NAN;      // [Ω·cm]
    iexpr scale = 1;         // [1]

    axial_resistivity() = default;
    axial_resistivity(const U::quantity& m, iexpr scale=1):
      value(m.value_as(U::cm*U::Ohm)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [Ω·cm]."};
    }
};

struct ARB_SYMBOL_VISIBLE membrane_capacitance {
    double value = NAN;      // [F/m²]
    iexpr scale = 1;         // [1]

    membrane_capacitance() = default;
    membrane_capacitance(const U::quantity& m, iexpr scale=1):
      value(m.value_as(U::F/U::m2)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [F/m²]."};
    }
};

struct ARB_SYMBOL_VISIBLE init_int_concentration {
    std::string ion = "";
    double value = NAN;      // [mM]
    iexpr scale = 1;         // [1]

    init_int_concentration() = default;
    init_int_concentration(const std::string& ion, const U::quantity& m, iexpr scale=1):
      ion{ion}, value(m.value_as(U::mM)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [mM]."};
    }
};

struct ARB_SYMBOL_VISIBLE ion_diffusivity {
    std::string ion = "";
    double value = NAN;      // [m²/s]
    iexpr scale = 1;         // [1]

    ion_diffusivity() = default;
    ion_diffusivity(const std::string& ion, const U::quantity& m, iexpr scale=1):
      ion{ion}, value(m.value_as(U::m2/U::s)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [m²/s]."};
    }
};

struct ARB_SYMBOL_VISIBLE init_ext_concentration {
    std::string ion = "";
    double value = NAN;      // [mM]
    iexpr scale = 1;         // [1]

    init_ext_concentration() = default;
    init_ext_concentration(const std::string& ion, const U::quantity& m, iexpr scale=1):
      ion{ion}, value(m.value_as(U::mM)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [mM]."};
    }
};

struct ARB_SYMBOL_VISIBLE init_reversal_potential {
    std::string ion = "";
    double value = NAN;      // [mV]
    iexpr scale = 1;         // [1]

    init_reversal_potential() = default;
    init_reversal_potential(const std::string& ion, const U::quantity& m, iexpr scale=1):
      ion{ion}, value(m.value_as(U::mV)), scale{scale} {
        if (std::isnan(value)) throw std::domain_error{"Value must be finite and in [mV]."};
    }
};

// Mechanism description, viz. mechanism name and
// (non-global) parameter settings. Used to assign
// density and point mechanisms to segments and
// reversal potential computations to cells.

struct ARB_SYMBOL_VISIBLE mechanism_desc {
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
    mechanism_desc(std::string name): name_(std::move(name)) {
        if (name_.empty()) throw cable_cell_error("mechanism_desc: null name");
    }
    mechanism_desc(const char* name): name_(name) {
        if (name_.empty()) throw cable_cell_error("mechanism_desc: null name");
    }

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


// Tagged mechanism types for dispatching decor::place() and decor::paint() calls
struct ARB_SYMBOL_VISIBLE junction {
    mechanism_desc mech;
    explicit junction(mechanism_desc m): mech(std::move(m)) {}
    junction(mechanism_desc m, const std::unordered_map<std::string, double>& params): mech(std::move(m)) {
        for (const auto& [param, value]: params) {
            mech.set(param, value);
        }
    }
};

struct ARB_SYMBOL_VISIBLE synapse {
    mechanism_desc mech;
    explicit synapse(mechanism_desc m): mech(std::move(m)) {}
    synapse(mechanism_desc m, const std::unordered_map<std::string, double>& params): mech(std::move(m)) {
        for (const auto& [param, value]: params) {
            mech.set(param, value);
        }
    }
};

struct ARB_SYMBOL_VISIBLE density {
    mechanism_desc mech;
    explicit density(mechanism_desc m): mech(std::move(m)) {}
    density(mechanism_desc m, const std::unordered_map<std::string, double>& params): mech(std::move(m)) {
        for (const auto& [param, value]: params) {
            mech.set(param, value);
        }
    }
};

struct ARB_SYMBOL_VISIBLE voltage_process {
    mechanism_desc mech;
    explicit voltage_process(mechanism_desc m): mech(std::move(m)) {}
    voltage_process(mechanism_desc m, const std::unordered_map<std::string, double>& params): mech(std::move(m)) {
        for (const auto& [param, value]: params) {
            mech.set(param, value);
        }
    }
};

struct ARB_SYMBOL_VISIBLE ion_reversal_potential_method {
    std::string ion;
    mechanism_desc method;
};

template <typename TaggedMech>
struct ARB_SYMBOL_VISIBLE scaled_mechanism {
    TaggedMech t_mech;
    std::unordered_map<std::string, iexpr> scale_expr;

    explicit scaled_mechanism(TaggedMech m) : t_mech(std::move(m)) {}

    scaled_mechanism& scale(std::string name, iexpr expr) {
        scale_expr.insert_or_assign(name, expr);
        return *this;
    }
};

using paintable =
    std::variant<init_membrane_potential,
                 axial_resistivity,
                 temperature,
                 membrane_capacitance,
                 ion_diffusivity,
                 init_int_concentration,
                 init_ext_concentration,
                 init_reversal_potential,
                 density,
                 voltage_process,
                 scaled_mechanism<density>>;

using placeable =
    std::variant<i_clamp,
                 threshold_detector,
                 synapse,
                 junction>;

using defaultable =
    std::variant<init_membrane_potential,
                 axial_resistivity,
                 temperature,
                 membrane_capacitance,
                 ion_diffusivity,
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

struct ARB_ARBOR_API cable_cell_parameter_set {
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
class ARB_ARBOR_API decor {
    std::vector<std::pair<region, paintable>> paintings_;
    std::vector<std::tuple<locset, placeable, hash_type>> placements_;
    cable_cell_parameter_set defaults_;
    std::unordered_map<hash_type, cell_tag_type> hashes_;

public:
    const auto& paintings()  const {return paintings_;  }
    const auto& placements() const {return placements_; }
    const auto& defaults()   const {return defaults_;   }

    decor& paint(region, paintable);
    decor& place(locset, placeable, cell_tag_type);
    decor& set_default(defaultable);

    cell_tag_type tag_of(hash_type) const;
};

ARB_ARBOR_API extern cable_cell_parameter_set neuron_parameter_defaults;

// Global cable cell data.

struct ARB_SYMBOL_VISIBLE cable_cell_global_properties {
    mechanism_catalogue catalogue = global_default_catalogue();

    // Optional check if membrane voltage magnitude is less than limit
    // during integration.
    std::optional<double> membrane_voltage_limit_mV;

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
    void add_ion(const std::string& ion_name,
                 int charge,
                 const U::quantity& init_iconc,
                 const U::quantity& init_econc,
                 const U::quantity& init_revpot,
                 const U::quantity& diffusivity=0.0*U::m2/U::s) {
        ion_species[ion_name] = charge;

        auto &ion_data = default_parameters.ion_data[ion_name];
        ion_data.init_int_concentration = init_iconc.value_as(U::mM);
        if (std::isnan(*ion_data.init_int_concentration)) throw std::domain_error("init_int_concentration must be finite and convertible to mM");
        ion_data.init_ext_concentration = init_econc.value_as(U::mM);
        if (std::isnan(*ion_data.init_ext_concentration)) throw std::domain_error("init_ext_concentration must be finite and convertible to mM");
        ion_data.init_reversal_potential = init_revpot.value_as(U::mV);
        if (std::isnan(*ion_data.init_reversal_potential)) throw std::domain_error("init_reversal_potential must be finite and convertible to mV");
        ion_data.diffusivity = diffusivity.value_as(U::m2/U::s);
        if (std::isnan(*ion_data.diffusivity) || *ion_data.diffusivity < 0) throw std::domain_error("diffusivity must be positive, finite, and convertible to m2/s");
    }

    void add_ion(const std::string& ion_name,
                 int charge,
                 const U::quantity& init_iconc,
                 const U::quantity& init_econc,
                 mechanism_desc revpot_mechanism,
                 const U::quantity& diffusivity=0.0*U::m2/U::s) {
        add_ion(ion_name, charge, init_iconc, init_econc, 0*U::mV, diffusivity);
        default_parameters.reversal_potential_method[ion_name] = std::move(revpot_mechanism);
    }
};

// Throw cable_cell_error if any default parameters are left unspecified,
// or if the supplied ion data is incomplete.
ARB_ARBOR_API void check_global_properties(const cable_cell_global_properties&);

} // namespace arb

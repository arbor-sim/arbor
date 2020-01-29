#pragma once

#include <arbor/arbexcept.hpp>
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

// FVM discretization policies/hints.
//
// CV polices, given a cable cell, provide a locset comprising
// CV boundary points to be used by the discretization. The
// discretization need not adopt the boundary points 100% faithfully;
// for example, it may elide empty CVs or perform other transformations
// for numeric fidelity or performance reasons.
//
// The cv_policy class is a value-like wrapper for actual
// policies that derive from `cv_policy_base`. At present,
// there are only three policies implemented, described below.
// The intent is to provide more sophisticated policies in the
// near future, specifically one based on the 'd-lambda' rule.
//
//   cv_policy_explicit:
//       Simply use the provided locset.
//
//   cv_policy_fixed_per_branch:
//       Use the same number of CVs for each branch.
//
//   cv_policy_max_extent:
//       Use as many CVs as required to ensure that no CV has
//       a length longer than a given value.
//
// Except for the explicit policy, CV policies may also take various
// flags (implemented as bitwise orable enums) to modify their
// behaviour. In general, future CV policies may choose to ignore
// flag values, but should respect them if their semantics are
// relevant.
//
//   cv_policy_flag::interior_forks:
//       Position CVs so as to include fork points, as opposed
//       to positioning them so that fork points are at the
//       boundaries of CVs.
//
//   cv_policy_flag::single_root_cv:
//       Always treat the root branch as a single CV regardless.

class cable_cell;

struct cv_policy_base {
    virtual locset cv_boundary_points(const cable_cell& cell) const = 0;
    virtual std::unique_ptr<cv_policy_base> clone() const = 0;
    virtual ~cv_policy_base() {}
};

using cv_policy_base_ptr = std::unique_ptr<cv_policy_base>;

struct cv_policy {
    cv_policy(const cv_policy_base& ref) { // implicit
        policy_ptr = ref.clone();
    }

    cv_policy(const cv_policy& other):
        policy_ptr(other.policy_ptr->clone()) {}

    cv_policy& operator=(const cv_policy& other) {
        policy_ptr = other.policy_ptr->clone();
        return *this;
    }

    cv_policy(cv_policy&&) = default;
    cv_policy& operator=(cv_policy&&) = default;

    locset cv_boundary_points(const cable_cell& cell) const {
        return policy_ptr->cv_boundary_points(cell);
    }

private:
    cv_policy_base_ptr policy_ptr;
};

// Common flags for CV policies; bitwise composable.
namespace cv_policy_flag {
    using value = unsigned;
    enum : unsigned {
        none = 0,
        interior_forks = 1<<0,
        single_root_cv = 1<<1
    };
}

struct cv_policy_explicit: cv_policy_base {
    explicit cv_policy_explicit(locset locs): locs_(std::move(locs)) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_explicit(*this));
    }

    locset cv_boundary_points(const cable_cell&) const override {
        return locs_;
    }

private:
    locset locs_;
};

struct cv_policy_max_extent: cv_policy_base {
    explicit cv_policy_max_extent(double max_extent, cv_policy_flag::value flags = cv_policy_flag::none):
         max_extent_(max_extent), flags_(flags) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_max_extent(*this));
    }

    locset cv_boundary_points(const cable_cell&) const override;

private:
    double max_extent_;
    cv_policy_flag::value flags_;
};

struct cv_policy_fixed_per_branch: cv_policy_base {
    explicit cv_policy_fixed_per_branch(unsigned cv_per_branch, cv_policy_flag::value flags = cv_policy_flag::none):
         cv_per_branch_(cv_per_branch), flags_(flags) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_fixed_per_branch(*this));
    }

    locset cv_boundary_points(const cable_cell&) const override;

private:
    unsigned cv_per_branch_;
    cv_policy_flag::value flags_;
};

struct cv_policy_every_sample: cv_policy_base {
    explicit cv_policy_every_sample(cv_policy_flag::value flags = cv_policy_flag::none):
         flags_(flags) {}

    cv_policy_base_ptr clone() const override {
        return cv_policy_base_ptr(new cv_policy_every_sample(*this));
    }

    locset cv_boundary_points(const cable_cell&) const override;

private:
    cv_policy_flag::value flags_;
};

inline cv_policy default_cv_policy() {
    return cv_policy_fixed_per_branch(1);
}

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

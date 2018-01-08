#pragma once

/* Classes for representing a mechanism schema, including those
 * generated automatically by modcc.
 */

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ion.hpp>

namespace arb {

struct mechanism_field_spec {
    enum field_kind {
        parameter,
        global,
        state,
    };
    enum field_kind kind = parameter;

    std::string units;

    double default_value = 0;
    double lower_bound = std::numeric_limits<double>::lowest();
    double upper_bound = std::numeric_limits<double>::max();

    bool valid(double x) const { return x>=lower_bound && x<=upper_bound; }

    // TODO: C++14 - no need for ctor below, as aggregate initialization
    // will work with default member initializers.

    mechanism_field_spec(
        enum field_kind kind = parameter,
        std::string units = "",
        double default_value = 0.,
        double lower_bound = std::numeric_limits<double>::lowest(),
        double upper_bound = std::numeric_limits<double>::max()
     ):
        kind(kind), units(units), default_value(default_value), lower_bound(lower_bound), upper_bound(upper_bound)
    {}
};

struct ion_dependency {
    bool write_concentration_int;
    bool write_concentration_ext;
};

// A hash of the mechanism dynamics description is used to ensure that offline-compiled
// mechanism implementations are correctly associated with their corresponding generated
// mechanism information.
// 
// Use a textual representation to ease readability.
using mechanism_fingerprint = std::string;

struct mechanism_info {
    // Global fields have one value common to an instance of a mechanism, are
    // constant in time and set at instantiation.
    std::unordered_map<std::string, mechanism_field_spec> globals;

    // Parameter fields may vary across the extent of a mechanism, but are
    // constant in time and set at instantiation.
    std::unordered_map<std::string, mechanism_field_spec> parameters;

    // State fields vary in time and across the extent of a mechanism, and
    // potentially can be sampled at run-time.
    std::unordered_map<std::string, mechanism_field_spec> state;

    // Ion dependencies.
    std::unordered_map<ionKind, ion_dependency> ions;

    mechanism_fingerprint fingerprint;
};

} // namespace arb

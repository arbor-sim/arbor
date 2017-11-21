#pragma once

#include <algorithm>
#include <memory>
#include <string>

#include <backends/fvm_types.hpp>
#include <backends/event.hpp>
#include <backends/multi_event_stream_state.hpp>
#include <ion.hpp>
#include <util/indirect.hpp>
#include <util/meta.hpp>
#include <util/make_unique.hpp>

namespace arb {

struct field_spec {
    enum field_kind {
        parameter, // defined in 'PARAMETER' block and a 'RANGE' variable.
        global,    // defined in 'PARAMETER' block and a 'GLOBAL' variable.
        state,     // defined in 'STATE' block; run-time, read only values.
    };
    enum field_kind kind = parameter;

    std::string units;

    fvm_value_type default_value = 0;
    fvm_value_type lower_bound = std::numeric_limits<fvm_value_type>::lowest();
    fvm_value_type upper_bound = std::numeric_limits<fvm_value_type>::max();

    // Until C++14, we need a ctor to provide default values instead of using
    // default member initializers and aggregate initialization.
    field_spec(
        enum field_kind kind = parameter,
        std::string units = "",
        fvm_value_type default_value = 0.,
        fvm_value_type lower_bound = std::numeric_limits<fvm_value_type>::lowest(),
        fvm_value_type upper_bound = std::numeric_limits<fvm_value_type>::max()
     ):
        kind(kind), units(units), default_value(default_value), lower_bound(lower_bound), upper_bound(upper_bound)
    {}
};


enum class mechanismKind {point, density};

/// The mechanism type is templated on a memory policy type.
/// The only difference between the abstract definition of a mechanism on host
/// or gpu is the information is stored, and how it is accessed.
template <typename Backend>
class mechanism {
public:
    struct ion_spec {
        bool uses;
        bool write_concentration_in;
        bool write_concentration_out;
    };

    using backend = Backend;

    using value_type = typename backend::value_type;
    using size_type = typename backend::size_type;

    // define storage types
    using array = typename backend::array;
    using iarray = typename backend::iarray;

    using view = typename backend::view;
    using iview = typename backend::iview;

    using const_view = typename backend::const_view;
    using const_iview = typename backend::const_iview;

    using ion_type = ion<backend>;

    using deliverable_event_stream_state = multi_event_stream_state<deliverable_event_data>;

    mechanism(size_type mech_id, const_iview vec_ci, const_view vec_t, const_view vec_t_to, const_view vec_dt, view vec_v, view vec_i, iarray&& node_index):
        mech_id_(mech_id),
        vec_ci_(vec_ci),
        vec_t_(vec_t),
        vec_t_to_(vec_t_to),
        vec_dt_(vec_dt),
        vec_v_(vec_v),
        vec_i_(vec_i),
        node_index_(std::move(node_index))
    {}

    std::size_t size() const {
        return node_index_.size();
    }

    const_iview node_index() const {
        return node_index_;
    }

    // Save pointers to data for use with GPU-side mechanisms;
    // TODO: might be able to remove this method if we separate instantiation
    // from initialization.
    virtual void set_params() {}
    virtual void set_weights(array&& weights) {} // override for density mechanisms

    virtual std::string name() const = 0;
    virtual std::size_t memory() const = 0;
    virtual void nrn_init()     = 0;
    virtual void nrn_state()    = 0;
    virtual void nrn_current()  = 0;
    virtual void deliver_events(const deliverable_event_stream_state& events) {};
    virtual ion_spec uses_ion(ionKind) const = 0;
    virtual void set_ion(ionKind k, ion_type& i, const std::vector<size_type>& index) = 0;
    virtual mechanismKind kind() const = 0;

    // Used by mechanisms that update ion concentrations.
    // Calling will copy the concentration, stored as internal state of the
    // mechanism, to the "global" copy of ion species state.
    virtual void write_back() {};

    // Mechanism instances with different global parameter settings can be distinguished by alias.
    std::string alias() const {
        return alias_.empty()? name(): alias_;
    }

    void set_alias(std::string alias) {
        alias_ = std::move(alias);
    }

    // For non-global fields:
    virtual view mechanism::* field_view_ptr(const char* id) const { return nullptr; }
    // For global fields:
    virtual value_type mechanism::* field_value_ptr(const char* id) const { return nullptr; }

    // Convenience wrappers for field access methods with string parameter.
    view mechanism::* field_view_ptr(const std::string& id) const { return field_view_ptr(id.c_str()); }
    value_type mechanism::* field_value_ptr(const std::string& id) const { return field_value_ptr(id.c_str()); }

    // net_receive() is used internally by deliver_events(), but
    // is exposed primarily for unit testing.
    virtual void net_receive(int, value_type) {};

    virtual ~mechanism() = default;

    // Mechanism identifier: index into list of mechanisms on cell group.
    size_type mech_id_;

    // Maps compartment index to cell index.
    const_iview vec_ci_;
    // Maps cell index to integration start time.
    const_view vec_t_;
    // Maps cell index to integration stop time.
    const_view vec_t_to_;
    // Maps compartment index to (stop time) - (start time).
    const_view vec_dt_;
    // Maps compartment index to voltage.
    view vec_v_;
    // Maps compartment index to current.
    view vec_i_;
    // Maps mechanism instance index to compartment index.
    iarray node_index_;

    std::string alias_;
};

template <class Backend>
using mechanism_ptr = std::unique_ptr<mechanism<Backend>>;

template <typename M>
auto make_mechanism(
    typename M::size_type mech_id,
    typename M::const_iview vec_ci,
    typename M::const_view vec_t,
    typename M::const_view vec_t_to,
    typename M::const_view vec_dt,
    typename M::view vec_v,
    typename M::view vec_i,
    typename M::array&& weights,
    typename M::iarray&& node_indices
)
DEDUCED_RETURN_TYPE(util::make_unique<M>(mech_id, vec_ci, vec_t, vec_t_to, vec_dt, vec_v, vec_i, std::move(weights), std::move(node_indices)))

} // namespace arb

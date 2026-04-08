#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <arbor/mechanism_abi.h>
#include <arbor/math.hpp>

#include <iostream>

namespace arb {
namespace efields_catalogue {
namespace kernel_reader {

using ::arb::math::exprelr;
using ::arb::math::safeinv;
using ::std::abs;
using ::std::cos;
using ::std::exp;
using ::std::max;
using ::std::min;
using ::std::pow;
using ::std::sin;
using ::std::sqrt;
using ::std::tanh;

static constexpr unsigned simd_width_ = 1;
static constexpr unsigned min_align_ = std::max(alignof(arb_value_type), alignof(arb_index_type));
using ::std::log;

#define PPACK_IFACE_BLOCK \
[[maybe_unused]] auto _pp_var_width                                                 = pp->width;\
[[maybe_unused]] auto _pp_var_n_detectors                                           = pp->n_detectors;\
[[maybe_unused]] auto _pp_var_dt                                                    = pp->dt;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_vec_ci                       = pp->vec_ci;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_v                        = pp->vec_v;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_i                        = pp->vec_i;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_vec_g                        = pp->vec_g;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_temperature_degC             = pp->temperature_degC;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_diam_um                      = pp->diam_um;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_area_um2                     = pp->area_um2;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_time_since_spike             = pp->time_since_spike;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_node_index                   = pp->node_index;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_peer_index                   = pp->peer_index;\
[[maybe_unused]] arb_index_type * __restrict__ _pp_var_multiplicity                 = pp->multiplicity;\
[[maybe_unused]] arb_value_type * __restrict__ _pp_var_weight                       = pp->weight;\
[[maybe_unused]] auto& _pp_var_events                                               = pp->events;\
[[maybe_unused]] auto _pp_var_mechanism_id                                          = pp->mechanism_id;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_contiguous               = pp->index_constraints.n_contiguous;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_constant                 = pp->index_constraints.n_constant;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_independent              = pp->index_constraints.n_independent;\
[[maybe_unused]] arb_size_type _pp_var_index_constraints_n_none                     = pp->index_constraints.n_none;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_contiguous  = pp->index_constraints.contiguous;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_constant    = pp->index_constraints.constant;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_independent = pp->index_constraints.independent;\
[[maybe_unused]] arb_index_type* __restrict__ _pp_var_index_constraints_none        = pp->index_constraints.none;\
[[maybe_unused]] auto _pp_var_field = pp->globals[0];\
[[maybe_unused]] auto const * const * _pp_var_random_numbers = pp->random_numbers;\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_da = pp->state_vars[0];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_xp = pp->parameters[0];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_yp = pp->parameters[1];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_zp = pp->parameters[2];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_xd = pp->parameters[3];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_yd = pp->parameters[4];\
[[maybe_unused]] arb_value_type* __restrict__ _pp_var_zd = pp->parameters[5];\
//End of IFACEBLOCK


// interface methods
static void init(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        arb_value_type dy, dz, dx;
        dx = _pp_var_xd[i_]-_pp_var_xp[i_];
        dy = _pp_var_yd[i_]-_pp_var_yp[i_];
        dz = _pp_var_zd[i_]-_pp_var_zp[i_];
        _pp_var_da[i_] =  1.0/sqrt(dx*dx+dy*dy+dz*dz);
    }
    if (!_pp_var_multiplicity) return;
    for (arb_size_type ix = 0; ix < 0; ++ix) {
        for (arb_size_type iy = 0; iy < _pp_var_width; ++iy) {
            pp->state_vars[ix][iy] *= _pp_var_multiplicity[iy];
        }
    }
}

static void advance_state(arb_mechanism_ppack* pp) {
}

static void compute_currents(arb_mechanism_ppack* pp) {
    PPACK_IFACE_BLOCK;
    auto e_field_ptr = (double*)((uint64_t) _pp_var_field);
    for (arb_size_type i_ = 0; i_ < _pp_var_width; ++i_) {
        auto node_indexi_ = _pp_var_node_index[i_];
        arb_value_type e = e_field_ptr[0];
        arb_value_type i = e*(_pp_var_xd[i_]-_pp_var_xp[i_])*_pp_var_da[i_];
        _pp_var_vec_i[node_indexi_] = fma(_pp_var_weight[i_], i, _pp_var_vec_i[node_indexi_]);
    }
}

static void write_ions(arb_mechanism_ppack* pp) {
}

static void apply_events(arb_mechanism_ppack* pp, arb_deliverable_event_stream* stream_ptr) {
    PPACK_IFACE_BLOCK;
    auto [begin_, end_] = *stream_ptr;
    for (; begin_<end_; ++begin_) {
        [[maybe_unused]] auto [i_, weight] = *begin_;
    }
}

static void post_event(arb_mechanism_ppack*) {}
#undef PPACK_IFACE_BLOCK
} // namespace kernel_reader
} // namespace efields_catalogue
} // namespace arb

extern "C" {
  arb_mechanism_interface* make_arb_efields_catalogue_reader_interface_multicore() {
    static arb_mechanism_interface result;
    result.partition_width = arb::efields_catalogue::kernel_reader::simd_width_;
    result.backend = arb_backend_kind_cpu;
    result.alignment = arb::efields_catalogue::kernel_reader::min_align_;
    result.init_mechanism = arb::efields_catalogue::kernel_reader::init;
    result.compute_currents = arb::efields_catalogue::kernel_reader::compute_currents;
    result.apply_events = arb::efields_catalogue::kernel_reader::apply_events;
    result.advance_state = arb::efields_catalogue::kernel_reader::advance_state;
    result.write_ions = arb::efields_catalogue::kernel_reader::write_ions;
    result.post_event = arb::efields_catalogue::kernel_reader::post_event;
    return &result;
  }
}


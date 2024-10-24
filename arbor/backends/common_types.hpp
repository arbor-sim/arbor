#pragma once

#include <arbor/common_types.hpp>

#include "fvm_layout.hpp"
#include "util/range.hpp"
#include "backends/threshold_crossing.hpp"
#include "execution_context.hpp"

namespace arb {

struct fvm_integration_result {
    util::range<const threshold_crossing*> crossings;
    util::range<const arb_value_type*> sample_time;
    util::range<const arb_value_type*> sample_value;
};

struct fvm_detector_info {
    arb_size_type count = 0;
    std::vector<arb_index_type> cv;
    std::vector<arb_value_type> threshold;
    execution_context ctx;
};

struct ion_data_flags {
    // flags for resetting ion states after W access
    bool write_eX_:1 = false; // is eX written?
    bool write_Xo_:1 = false; // is Xo written?
    bool write_Xi_:1 = false; // is Xi written?
    bool write_Xd_:1 = false; // is Xd written?
    bool read_eX_:1  = false; // is eX read?
    bool read_Xo_:1  = false; // is Xo read?
    bool read_Xi_:1  = false; // is Xi read?
    bool read_Xd_:1  = false; // is Xd read?

    ion_data_flags(const fvm_ion_config& config):
        write_eX_(config.revpot_written),
        write_Xo_(config.econc_written),
        write_Xi_(config.iconc_written),
        write_Xd_(config.is_diffusive),
        read_eX_(config.revpot_read),
        read_Xo_(config.econc_read),
        read_Xi_(config.iconc_read),
        read_Xd_(config.is_diffusive)
    {}

    ion_data_flags() = default;
    ion_data_flags(const ion_data_flags&) = default;
    ion_data_flags& operator=(const ion_data_flags&) = default;

    bool xi() const { return read_Xi_ || write_Xi_; }
    bool reset_xi() const { return write_Xi_; }

    bool xo() const { return read_Xo_ || write_Xo_; }
    bool reset_xo() const { return write_Xo_; }

    bool xd() const { return read_Xd_ || write_Xd_; }
    bool reset_xd() const { return write_Xd_; }

    bool ex() const { return read_eX_ || write_eX_; }
    bool reset_ex() const { return write_eX_; }
};


}

#include "fvm.hpp"

#include <mechanisms/gpu/hh_gpu.hpp>
#include <mechanisms/gpu/pas_gpu.hpp>
#include <mechanisms/gpu/expsyn_gpu.hpp>
#include <mechanisms/gpu/exp2syn_gpu.hpp>
#include <mechanisms/gpu/test_kin1_gpu.hpp>
#include <mechanisms/gpu/test_kinlva_gpu.hpp>
#include <mechanisms/gpu/test_ca_gpu.hpp>

namespace arb {
namespace gpu {

std::map<std::string, backend::maker_type>
backend::mech_map_ = {
    { "pas",         maker<mechanism_pas> },
    { "hh",          maker<mechanism_hh> },
    { "expsyn",      maker<mechanism_expsyn> },
    { "exp2syn",     maker<mechanism_exp2syn> },
    { "test_kin1",   maker<mechanism_test_kin1> },
    { "test_kinlva", maker<mechanism_test_kinlva> },
    { "test_ca",     maker<mechanism_test_ca> }
};

} // namespace gpu
} // namespace arb

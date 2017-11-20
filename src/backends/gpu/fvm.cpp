#include "fvm.hpp"

#include <mechanisms/gpu/hh.hpp>
#include <mechanisms/gpu/pas.hpp>
#include <mechanisms/gpu/expsyn.hpp>
#include <mechanisms/gpu/exp2syn.hpp>
#include <mechanisms/gpu/test_kin1.hpp>
#include <mechanisms/gpu/test_kinlva.hpp>
#include <mechanisms/gpu/test_ca.hpp>

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

} // namespace multicore
} // namespace arb

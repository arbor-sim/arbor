#include "fvm.hpp"

#include <mechanisms/gpu/hh.hpp>
#include <mechanisms/gpu/pas.hpp>
#include <mechanisms/gpu/expsyn.hpp>
#include <mechanisms/gpu/exp2syn.hpp>
#include <mechanisms/gpu/test_kin1.hpp>
#include <mechanisms/gpu/test_kinlva.hpp>

namespace nest {
namespace mc {
namespace gpu {

std::map<std::string, backend::maker_type>
backend::mech_map_ = {
    { "pas",     maker<mechanisms::gpu::pas::mechanism_pas> },
    { "hh",      maker<mechanisms::gpu::hh::mechanism_hh> },
    { "expsyn",  maker<mechanisms::gpu::expsyn::mechanism_expsyn> },
    { "exp2syn", maker<mechanisms::gpu::exp2syn::mechanism_exp2syn> },
    { "test_kin1", maker<mechanisms::gpu::test_kin1::mechanism_test_kin1> },
    { "test_kinlva", maker<mechanisms::gpu::test_kinlva::mechanism_test_kinlva> }
};

} // namespace multicore
} // namespace mc
} // namespace nest

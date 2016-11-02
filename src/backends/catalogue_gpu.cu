#include "catalogue_gpu.hpp"

#include <mechanisms/gpu/hh.hpp>
#include <mechanisms/gpu/pas.hpp>
#include <mechanisms/gpu/expsyn.hpp>
#include <mechanisms/gpu/exp2syn.hpp>

namespace nest {
namespace mc {
namespace gpu {

const std::map<std::string, catalogue::maker_type>
catalogue::mech_map_ = {
    { "pas",     maker<mechanisms::gpu::pas::mechanism_pas> },
    { "hh",      maker<mechanisms::gpu::hh::mechanism_hh> },
    { "expsyn",  maker<mechanisms::gpu::expsyn::mechanism_expsyn> },
    { "exp2syn", maker<mechanisms::gpu::exp2syn::mechanism_exp2syn> }
};

} // namespace multicore
} // namespace mc
} // namespace nest


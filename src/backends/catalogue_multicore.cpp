#include "catalogue_multicore.hpp"

#include <mechanisms/multicore/hh.hpp>
#include <mechanisms/multicore/pas.hpp>
#include <mechanisms/multicore/expsyn.hpp>
#include <mechanisms/multicore/exp2syn.hpp>

namespace nest {
namespace mc {
namespace multicore {

const std::map<std::string, catalogue::maker_type>
catalogue::mech_map_ = {
    { "pas",     maker<mechanisms::pas::mechanism_pas> },
    { "hh",      maker<mechanisms::hh::mechanism_hh> },
    { "expsyn",  maker<mechanisms::expsyn::mechanism_expsyn> },
    { "exp2syn", maker<mechanisms::exp2syn::mechanism_exp2syn> }
};

} // namespace multicore
} // namespace mc
} // namespace nest

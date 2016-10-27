#include "catalogue_multicore.hpp"

#include <mechanisms/hh.hpp>
#include <mechanisms/pas.hpp>
#include <mechanisms/expsyn.hpp>
#include <mechanisms/exp2syn.hpp>

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

#include "fvm_multicore.hpp"

#include <mechanisms/multicore/hh.hpp>
#include <mechanisms/multicore/pas.hpp>
#include <mechanisms/multicore/expsyn.hpp>
#include <mechanisms/multicore/exp2syn.hpp>

namespace nest {
namespace mc {
namespace multicore {

std::map<std::string, backend::maker_type>
backend::mech_map_ = {
    { std::string("pas"),     maker<mechanisms::pas::mechanism_pas> },
    { std::string("hh"),      maker<mechanisms::hh::mechanism_hh> },
    { std::string("expsyn"),  maker<mechanisms::expsyn::mechanism_expsyn> },
    { std::string("exp2syn"), maker<mechanisms::exp2syn::mechanism_exp2syn> }
};

} // namespace multicore
} // namespace mc
} // namespace nest

#include "fvm.hpp"

#include <mechanisms/multicore/hh.hpp>
#include <mechanisms/multicore/pas.hpp>
#include <mechanisms/multicore/expsyn.hpp>
#include <mechanisms/multicore/exp2syn.hpp>
#include <mechanisms/multicore/test_kin1.hpp>
#include <mechanisms/multicore/test_kinlva.hpp>

namespace nest {
namespace mc {
namespace multicore {

std::map<std::string, backend::maker_type>
backend::mech_map_ = {
    { std::string("pas"),       maker<mechanism_pas> },
    { std::string("hh"),        maker<mechanism_hh> },
    { std::string("expsyn"),    maker<mechanism_expsyn> },
    { std::string("exp2syn"),   maker<mechanism_exp2syn> },
    { std::string("test_kin1"), maker<mechanism_test_kin1> },
    { std::string("test_kinlva"), maker<mechanism_test_kinlva> }
};

} // namespace multicore
} // namespace mc
} // namespace nest

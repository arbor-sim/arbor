#include "fvm.hpp"

#include <mechanisms/multicore/hh_cpu.hpp>
#include <mechanisms/multicore/pas_cpu.hpp>
#include <mechanisms/multicore/expsyn_cpu.hpp>
#include <mechanisms/multicore/exp2syn_cpu.hpp>
#include <mechanisms/multicore/test_kin1_cpu.hpp>
#include <mechanisms/multicore/test_kinlva_cpu.hpp>
#include <mechanisms/multicore/test_ca_cpu.hpp>

namespace arb {
namespace multicore {

std::map<std::string, backend::maker_type>
backend::mech_map_ = {
    { std::string("pas"),       maker<mechanism_pas> },
    { std::string("hh"),        maker<mechanism_hh> },
    { std::string("expsyn"),    maker<mechanism_expsyn> },
    { std::string("exp2syn"),   maker<mechanism_exp2syn> },
    { std::string("test_kin1"), maker<mechanism_test_kin1> },
    { std::string("test_kinlva"), maker<mechanism_test_kinlva> },
    { std::string("test_ca"),   maker<mechanism_test_ca> }
};

} // namespace multicore
} // namespace arb

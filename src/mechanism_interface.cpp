#include "mechanism_interface.hpp"

/*  include the mechanisms
#include <mechanisms/hh.hpp>
#include <mechanisms/pas.hpp>
*/

namespace nest {
namespace mc {
namespace mechanisms {

std::map<std::string, mechanism_helper<double, int>> mechanism_map;

void setup_mechanism_helpers() {
    /*  manually insert
    mechanism_map["hh"]  = mechanisms::hh;
    mechanism_map["pas"] = mechanisms::pas;
    */
}

} // namespace mechanisms
} // namespace nest
} // namespace mc


#include "mechanism_interface.hpp"

//
//  include the mechanisms
//

#include <mechanisms/hh.hpp>
#include <mechanisms/pas.hpp>
#include <mechanisms/expsyn.hpp>
#include <mechanisms/exp2syn.hpp>


namespace nest {
namespace mc {
namespace mechanisms {

std::map<std::string, mechanism_helper_ptr<value_type, index_type>> mechanism_helpers;

void setup_mechanism_helpers() {
    mechanism_helpers["pas"] =
        make_mechanism_helper<
            mechanisms::pas::helper<value_type, index_type>
        >();

    mechanism_helpers["hh"] =
        make_mechanism_helper<
            mechanisms::hh::helper<value_type, index_type>
        >();

    mechanism_helpers["expsyn"] =
        make_mechanism_helper<
            mechanisms::expsyn::helper<value_type, index_type>
        >();

    mechanism_helpers["exp2syn"] =
        make_mechanism_helper<
            mechanisms::exp2syn::helper<value_type, index_type>
        >();
}

mechanism_helper_ptr<value_type, index_type>&
get_mechanism_helper(const std::string& name)
{
    auto helper = mechanism_helpers.find(name);
    if(helper==mechanism_helpers.end()) {
        throw std::out_of_range(
            nest::mc::util::pprintf("there is no mechanism named \'%\'", name)
        );
    }

    return helper->second;
}

} // namespace mechanisms
} // namespace nest
} // namespace mc


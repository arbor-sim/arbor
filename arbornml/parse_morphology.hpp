#pragma once

#include <arbornml/arbornml.hpp>
#include "xmlwrap.hpp"

namespace arbnml {

morphology_data parse_morphology_element(xml_xpathctx ctx, xml_node morph);

} // namespace arbnml

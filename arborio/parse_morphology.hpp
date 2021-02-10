#pragma once

#include <arborio/arbornml.hpp>
#include "xmlwrap.hpp"

namespace arborio {

morphology_data parse_morphology_element(xml_xpathctx ctx, xml_node morph);

} // namespace arborio

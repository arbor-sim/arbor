#pragma once

#include <arborio/neuroml.hpp>
#include "xmlwrap.hpp"

namespace arborio {

nml_morphology_data nml_parse_morphology_element(xmlwrap::xml_xpathctx ctx, xmlwrap::xml_node morph, enum neuroml_options::values);

} // namespace arborio

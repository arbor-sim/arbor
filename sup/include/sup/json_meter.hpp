#include <arbor/profile/meter_manager.hpp>
#include <sup/export.hpp>
#include <nlohmann/json.hpp>

namespace sup {

ARB_SUP_API nlohmann::json to_json(const arb::profile::meter_report&);

} // namespace sup

#include <arbor/profile/meter_manager.hpp>
#include <nlohmann/json.hpp>

namespace anc {

nlohmann::json to_json(const arb::profile::meter_report&);

}

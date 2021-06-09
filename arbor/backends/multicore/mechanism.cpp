#include <algorithm>
#include <cstddef>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <arbor/fvm_types.hpp>
#include <arbor/common_types.hpp>
#include <arbor/math.hpp>
#include <arbor/mechanism.hpp>

#include "util/index_into.hpp"
#include "util/strprintf.hpp"
#include "util/maputil.hpp"
#include "util/padded_alloc.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

#include "backends/multicore/mechanism.hpp"
#include "backends/multicore/fvm.hpp"
#include "backends/multicore/multicore_common.hpp"
#include "backends/multicore/partition_by_constraint.hpp"

namespace arb {
namespace multicore {

void mechanism::set_parameter(const std::string& key, const std::vector<fvm_value_type>& values) {
    if (values.size()!=ppack_.width) throw arbor_internal_error("multicore/mechanism: mechanism parameter size mismatch");
    auto field_ptr = field_data(key);
    if (!field_ptr) throw arbor_internal_error(util::pprintf("multicore/mechanism: no such mechanism parameter '{}'", key));
    if (!ppack_.width) return;
    auto field = util::range_n(field_ptr, width_padded_);
    copy_extend(values, field, values.back());
}

} // namespace multicore
} // namespace arb

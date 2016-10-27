#pragma once

#include <map>
#include <string>

#include <ion.hpp>
#include <mechanism.hpp>

#include "memory_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

class catalogue : public memory_traits {
public:
    using base = memory_traits;
    using base::view;
    using base::const_iview;
    using mechanism_type = mechanisms::mechanism<base>;
    using mechanism_ptr_type = mechanisms::mechanism_ptr<base>;

    using ion_type = mechanisms::ion<base>;

    template <typename Indices>
    static mechanism_ptr_type make(
        const std::string& name,
        view vec_v, view vec_i,
        const Indices& node_indices)
    {
        auto entry = mech_map_.find(name);
        if (entry==mech_map_.end()) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        return entry->second(vec_v, vec_i,  memory::make_const_view(node_indices));
    }

    static bool has(const std::string& name) {
        return mech_map_.count(name)>0;
    }

private:
    using maker_type = mechanism_ptr_type (*)(view, view, const_iview);
    static const std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism_ptr_type maker(view vec_v, view vec_i, const_iview node_indices) {
        return mechanisms::make_mechanism<Mech<memory_traits>>(vec_v, vec_i, node_indices);
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

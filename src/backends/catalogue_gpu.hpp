#pragma once

#include <map>
#include <string>

#include <ion.hpp>
#include <mechanism.hpp>

#include "memory_gpu.hpp"

namespace nest {
namespace mc {
namespace gpu {

class catalogue : public memory_traits {
public:
    using mechanism_type = mechanisms::mechanism<memory_traits>;
    using mechanism_ptr_type = mechanisms::mechanism_ptr<memory_traits>;

    using ion_type = mechanisms::ion<memory_traits>;

    static mechanism_ptr_type make(
        const std::string& name,
        view vec_v, view vec_i,
        const std::vector<size_type>& node_indices)
    {
        auto entry = mech_map_.find(name);
        if (entry==mech_map_.end()) {
            throw std::out_of_range("no mechanism in database : " + name);
        }

        //return entry->second(vec_v, vec_i, memory::on_gpu(node_indices));
        iarray tmp(memory::make_const_view(node_indices));
        return entry->second(vec_v, vec_i, std::move(tmp));
    }

    static bool has(const std::string& name) {
        return mech_map_.count(name)>0;
    }

private:
    using maker_type = mechanism_ptr_type (*)(view, view, iarray&&);
    static const std::map<std::string, maker_type> mech_map_;

    template <template <typename> class Mech>
    static mechanism_ptr_type maker(view vec_v, view vec_i, iarray&& node_indices) {
        return mechanisms::make_mechanism<Mech<memory_traits>>(vec_v, vec_i, std::move(node_indices));
    }
};

} // namespace gpu
} // namespace mc
} // namespace nest


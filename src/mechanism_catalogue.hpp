#pragma once

#include <map>
#include <stdexcept>
#include <string>

#include <mechanism.hpp>
#include <mechanisms/hh.hpp>
#include <mechanisms/pas.hpp>
#include <mechanisms/expsyn.hpp>
#include <mechanisms/exp2syn.hpp>

namespace nest {
namespace mc {
namespace mechanisms {

template <typename T, typename I>
struct catalogue {
    using view_type = typename mechanism<T, I>::view_type;
    using index_view = typename mechanism<T, I>::index_view;

    template <typename Indices>
    static mechanism_ptr<T, I> make(
        const std::string& name,
        view_type vec_v,
        view_type vec_i,
        Indices& node_indices)
    {
        auto entry = mech_map.find(name);
        if (entry==mech_map.end()) {
            throw std::out_of_range("no such mechanism");
        }

        auto node_view = index_view{node_indices};
        return entry->second(vec_v, vec_i, node_view);
    }

    static bool has(const std::string& name) {
        return mech_map.count(name)>0;
    }

private:
    using maker_type = mechanism_ptr<T, I> (*)(view_type, view_type, index_view);
    static const std::map<std::string, maker_type> mech_map;

    template <template <typename, typename> class mech>
    static mechanism_ptr<T, I> maker(view_type vec_v, view_type vec_i, index_view node_indices) {
        return make_mechanism<mech<T, I>>(vec_v, vec_i, node_indices);
    }
};

template <typename T, typename I>
const std::map<std::string, typename catalogue<T, I>::maker_type> catalogue<T, I>::mech_map = {
    { "pas",     maker<pas::mechanism_pas> },
    { "hh",      maker<hh::mechanism_hh> },
    { "expsyn",  maker<expsyn::mechanism_expsyn> },
    { "exp2syn", maker<exp2syn::mechanism_exp2syn> }
};


} // namespace mechanisms
} // namespace mc
} // namespace nest

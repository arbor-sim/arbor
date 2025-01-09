#include <memory>
#include <sstream>
#include <unordered_map>
#include <variant>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/util/pp_util.hpp>

#include "util/strprintf.hpp"

namespace arb {

using value_type = cable_cell::value_type;
using index_type = cable_cell::index_type;
using size_type = cable_cell::size_type;

template <typename T> struct constant_type {
    template <typename> using type = T;
};

// Helper for debugging: print outermost DSL constructor
std::string show(const paintable& item) {
    std::stringstream os;
    std::visit(
        [&] (const auto& p) {
            using T = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<init_membrane_potential, T>) {
                os << "init-membrane-potential";
            }
            else if constexpr (std::is_same_v<axial_resistivity, T>) {
                os << "axial-resistivity";
            }
            else if constexpr (std::is_same_v<temperature, T>) {
                os << "temperature-kelvin";
            }
            else if constexpr (std::is_same_v<membrane_capacitance, T>) {
                os << "membrane-capacitance";
            }
            else if constexpr (std::is_same_v<init_int_concentration, T>) {
                os << "ion-internal-concentration";
            }
            else if constexpr (std::is_same_v<init_ext_concentration, T>) {
                os << "ion-external-concentration";
            }
            else if constexpr (std::is_same_v<init_reversal_potential, T>) {
                os << "ion-reversal-potential";
            }
            else if constexpr (std::is_same_v<density, T>) {
                os << "density:" << p.mech.name();
            }
            else if constexpr (std::is_same_v<voltage_process, T>) {
                os << "voltage-process:" << p.mech.name();
            }
        },
        item);
    return os.str();
}


struct cable_cell_impl {
    using value_type = cable_cell::value_type;
    using index_type = cable_cell::index_type;
    using size_type  = cable_cell::size_type;

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider;

    // Regional assignments.
    cable_cell_region_map region_map;

    // Point assignments.
    cable_cell_location_map location_map;

    // Track number of point assignments by type for lid/target numbers.
    dynamic_typed_map<constant_type<cell_lid_type>::type> placed_count;

    // The label dictionary.
    const label_dict dictionary;

    // The decorations on the cell.
    decor decorations;

    // Discretization
    std::optional<cv_policy> discretization_;

    // The placeable label to lid_range map
    dynamic_typed_map<constant_type<std::unordered_multimap<hash_type, lid_range>>::type> labeled_lid_ranges;

    cable_cell_impl(const arb::morphology& m, const label_dict& labels, const decor& decorations, const std::optional<cv_policy>& cvp):
        provider(m, labels),
        dictionary(labels),
        decorations(decorations),
        discretization_{cvp}
    {
        init();
    }

    cable_cell_impl(): cable_cell_impl({}, {}, {}, {}) {}

    cable_cell_impl(const cable_cell_impl& other) = default;

    cable_cell_impl(cable_cell_impl&& other) = default;

    void init();

    template <typename T>
    mlocation_map<T>& get_location_map(const T&) {
        return location_map.get<T>();
    }

    mlocation_map<synapse>& get_location_map(const synapse& desc) {
        return location_map.get<synapse>()[desc.mech.name()];
    }

    mlocation_map<junction>& get_location_map(const junction& desc) {
        return location_map.get<junction>()[desc.mech.name()];
    }

    template <typename Item>
    void place(const mlocation_list& locs, const Item& item, const hash_type& label) {
        auto& mm = get_location_map(item);
        cell_lid_type& lid = placed_count.get<Item>();
        cell_lid_type first = lid;

        for (const auto& loc: locs) {
            placed<Item> p{loc, lid++, item, label};
            mm.push_back(p);
        }
        auto range = lid_range(first, lid);
        auto& lid_ranges = labeled_lid_ranges.get<Item>();
        lid_ranges.insert(std::make_pair(label, range));
    }

    template <typename T>
    mcable_map<T>& get_region_map(const T&) {
        return region_map.get<T>();
    }

    mcable_map<voltage_process>& get_region_map(const voltage_process& v) {
        return region_map.get<voltage_process>()[v.mech.name()];
    }

    mcable_map<std::pair<density, iexpr_map>> &
    get_region_map(const density &desc) {
      return region_map.get<density>()[desc.mech.name()];
    }

    mcable_map<init_int_concentration>& get_region_map(const init_int_concentration& init) {
        return region_map.get<init_int_concentration>()[init.ion];
    }

    mcable_map<ion_diffusivity>& get_region_map(const ion_diffusivity& init) {
        return region_map.get<ion_diffusivity>()[init.ion];
    }

    mcable_map<init_ext_concentration>& get_region_map(const init_ext_concentration& init) {
        return region_map.get<init_ext_concentration>()[init.ion];
    }

    mcable_map<init_reversal_potential>& get_region_map(const init_reversal_potential& init) {
        return region_map.get<init_reversal_potential>()[init.ion];
    }

    void paint(const mextent& cables, const std::string& str, const density& prop) {
        this->paint(cables, str, scaled_mechanism<density>(prop));
    }

    void paint(const mextent& cables, const std::string& str, const scaled_mechanism<density>& prop) {
        std::unordered_map<std::string, iexpr_ptr> im;
        for (const auto& [label, iex]: prop.scale_expr) {
            im.insert_or_assign(label, thingify(iex, provider));
        }

        auto& mm = get_region_map(prop.t_mech);
        for (const auto& cable: cables) {
            // Skip zero-length cables in extent:
            if (cable.prox_pos == cable.dist_pos) continue;
            if (!mm.insert(cable, {prop.t_mech, im})) {
                throw cable_cell_error(util::pprintf("Setting mechanism '{}' on region '{}' overpaints at cable {}",
                                                     prop.t_mech.mech.name(), str, cable));
            }
        }
    }

    template <typename TaggedMech>
    void paint(const mextent& cables, const std::string& str, const TaggedMech& prop) {
        auto& mm = get_region_map(prop);
        for (const auto& cable: cables) {
            // Skip zero-length cables in extent:
            if (cable.prox_pos == cable.dist_pos) continue;
            if (!mm.insert(cable, prop)) {
                throw cable_cell_error(util::pprintf("Setting property '{}' on region '{}' overpaints at cable {}",
                                                     show(prop), str, cable));
            }
        }
    }

    mlocation_list concrete_locset(const locset& l) const {
        return thingify(l, provider);
    }

    mextent concrete_region(const region& r) const {
        return thingify(r, provider);
    }
};

const std::optional<cv_policy>& cable_cell::discretization() const { return impl_->discretization_; }
void cable_cell::discretization(cv_policy cvp) { impl_->discretization_ = std::move(cvp); }


using impl_ptr = std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)>;
impl_ptr make_impl(cable_cell_impl* c) {
    return impl_ptr(c, [](cable_cell_impl* p){delete p;});
}

void cable_cell_impl::init() {
    // Try to cache with a lookback of one since most models paint/place one
    // region/locset in direct succession. We also key on the stringy view of
    // expressions since in general equality is undecidable.
    std::string last_label = "";
    mextent last_region;
    mlocation_list last_locset;
    for (const auto& [where, what]: decorations.paintings()) {
        if (auto region = util::to_string(where); last_label != region) {
            last_label  = std::move(region);
            last_region = thingify(where, provider);
        }
        std::visit([this, &last_region, &last_label] (auto&& what) { this->paint(last_region, last_label, what); }, what);
    }
    for (const auto& [where, what, label]: decorations.placements()) {
        if (auto locset = util::to_string(where); last_label != locset) {
            last_label  = std::move(locset);
            last_locset = thingify(where, provider);
        }
        std::visit([this, &last_locset, &label=label] (auto&& what) { return this->place(last_locset, what, label); },
                   what);
    }
}

cable_cell::cable_cell(const arb::morphology& m, const decor& decorations, const label_dict& dictionary, const std::optional<cv_policy>& cvp):
    impl_(make_impl(new cable_cell_impl(m, dictionary, decorations, cvp)))
{}

cable_cell::cable_cell(): impl_(make_impl(new cable_cell_impl())) {}

cable_cell::cable_cell(const cable_cell& other):
    impl_(make_impl(new cable_cell_impl(*other.impl_)))
{}

const label_dict& cable_cell::labels() const {
    return impl_->dictionary;
}

const concrete_embedding& cable_cell::embedding() const {
    return impl_->provider.embedding();
}

const arb::morphology& cable_cell::morphology() const {
    return impl_->provider.morphology();
}

const mprovider& cable_cell::provider() const {
    return impl_->provider;
}

mlocation_list cable_cell::concrete_locset(const locset& l) const {
    return impl_->concrete_locset(l);
}

mextent cable_cell::concrete_region(const region& r) const {
    return impl_->concrete_region(r);
}

const cable_cell_location_map& cable_cell::location_assignments() const {
    return impl_->location_map;
}

const cable_cell_region_map& cable_cell::region_assignments() const {
    return impl_->region_map;
}

const decor& cable_cell::decorations() const {
    return impl_->decorations;
}

const cable_cell_parameter_set& cable_cell::default_parameters() const {
    return impl_->decorations.defaults();
}

const cable_cell::lid_range_map& cable_cell::detector_ranges() const {
    return impl_->labeled_lid_ranges.get<threshold_detector>();
}

const cable_cell::lid_range_map& cable_cell::synapse_ranges() const {
    return impl_->labeled_lid_ranges.get<synapse>();
}

const cable_cell::lid_range_map& cable_cell::junction_ranges() const {
    return impl_->labeled_lid_ranges.get<junction>();
}

cell_tag_type decor::tag_of(hash_type hash) const {
    if (!hashes_.count(hash)) throw arbor_internal_error{util::pprintf("Unknown hash for {}.", std::to_string(hash))};
    return hashes_.at(hash);
}

} // namespace arb

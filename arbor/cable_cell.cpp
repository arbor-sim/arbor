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

// Metaprogramming helper to get the index of a type in a variant
template <typename> struct tag { };

template <typename T, typename V> struct get_index;

template <typename T, typename... Ts>
struct get_index<T, std::variant<Ts...>>: std::integral_constant<size_t, std::variant<tag<Ts>...>(tag<T>()).index()> {};

template <typename T, typename V> auto constexpr get_index_v = get_index<T, V>::value;

// injection to determine the return value of
template <typename T>
using region_assignment_value = std::conditional_t<std::is_same_v<T, density>,
                                                   mcable_map<std::pair<T, iexpr_map>>,
                                                   mcable_map<T>>;


struct cable_cell_impl {
    using value_type = cable_cell::value_type;
    using index_type = cable_cell::index_type;
    using size_type  = cable_cell::size_type;

    // The label dictionary.
    label_dict dictionary;

    // Embedded morphology and labelled region/locset lookup.
    mprovider provider;

    // Regional assignments.
    region_assignment<density> densities_;
    region_assignment<voltage_process> voltage_processes_;
    region_assignment<init_int_concentration> init_int_concentrations_;
    region_assignment<init_ext_concentration> init_ext_concentrations_;
    region_assignment<init_reversal_potential> reversal_potentials_;
    region_assignment<ion_diffusivity> diffusivities_;
    region_assignment<temperature> temperatures_;
    region_assignment<init_membrane_potential> init_membrane_potentials_;
    region_assignment<axial_resistivity> axial_resistivities_;
    region_assignment<membrane_capacitance> membrane_capacitances_;
    region_assignment<ion_diffusivity> ion_diffusivities_;
    region_assignment<init_reversal_potential> init_reversal_potentials_;

    // Track number of point assignments by type for lid/target numbers.
    std::array<cell_lid_type, std::variant_size_v<placeable>> placed_counts_ = {};
    // The placeable label to lid_range map
    std::array<std::unordered_multimap<hash_type, lid_range>, std::variant_size_v<placeable>> labeled_lid_ranges_;

    // Point assignments.
    std::unordered_map<std::string, mlocation_map<synapse>> synapses_;
    std::unordered_map<std::string, mlocation_map<junction>> junctions_;
    mlocation_map<threshold_detector> detectors_;
    mlocation_map<i_clamp> i_clamps_;

    // The decorations on the cell.
    decor decorations;

    // Discretization
    std::optional<cv_policy> discretization_;
    cable_cell_impl(const arb::morphology& m, const label_dict& labels, const decor& decorations, const std::optional<cv_policy>& cvp):
        dictionary(labels),
        provider(m, dictionary),
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
    auto& get_location_map(const T& it) {
        static_assert(get_index_v<T, placeable> < std::variant_size_v<placeable>, "Not a placeable item");
        if constexpr (std::is_same_v<T, synapse>) return synapses_[it.mech.name()];
        if constexpr (std::is_same_v<T, junction>) return junctions_[it.mech.name()];
        if constexpr (std::is_same_v<T, i_clamp>) return i_clamps_;
        if constexpr (std::is_same_v<T, threshold_detector>) return detectors_;
    }

    template <typename Item>
    void place(const mlocation_list& locs, const Item& item, const hash_type& label) {
        auto index = get_index_v<Item, placeable>;
        auto& mm = get_location_map(item);
        cell_lid_type& lid = placed_counts_[index];
        cell_lid_type first = lid;

        for (const auto& loc: locs) {
            placed<Item> p{loc, lid++, item, label};
            mm.push_back(p);
        }
        auto range = lid_range(first, lid);
        auto& lid_ranges = labeled_lid_ranges_[index];
        lid_ranges.emplace(label, range);
    }

    template <typename T>
    region_assignment_value<T>& get_region_map(const T& it) {
        static_assert(get_index_v<T, paintable> < std::variant_size_v<paintable>);
        if constexpr (std::is_same_v<T, density>)                 { return densities_[it.mech.name()]; }
        if constexpr (std::is_same_v<T, voltage_process>)         { return voltage_processes_[it.mech.name()]; }
        if constexpr (std::is_same_v<T, init_int_concentration>)  { return init_int_concentrations_[it.ion]; }
        if constexpr (std::is_same_v<T, ion_diffusivity>)         { return ion_diffusivities_[it.ion]; }
        if constexpr (std::is_same_v<T, init_ext_concentration>)  { return init_ext_concentrations_[it.ion]; }
        if constexpr (std::is_same_v<T, init_reversal_potential>) { return init_reversal_potentials_[it.ion]; }
        if constexpr (std::is_same_v<T, init_membrane_potential>) { return init_membrane_potentials_; }
        if constexpr (std::is_same_v<T, axial_resistivity>)       { return axial_resistivities_; }
        if constexpr (std::is_same_v<T, temperature>)             { return temperatures_; }
        if constexpr (std::is_same_v<T, membrane_capacitance>)    { return membrane_capacitances_; }
    }

    void paint(const mextent& cables, const std::string& str, const density& prop) { this->paint(cables, str, scaled_mechanism<density>(prop)); }

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

    mlocation_list concrete_locset(const locset& l) const { return thingify(l, provider); }
    mextent concrete_region(const region& r) const { return thingify(r, provider); }
};

const std::optional<cv_policy>& cable_cell::discretization() const { return impl_->discretization_; }
void cable_cell::discretization(cv_policy cvp) { impl_->discretization_ = std::move(cvp); }


using impl_ptr = std::unique_ptr<cable_cell_impl, void (*)(cable_cell_impl*)>;
impl_ptr make_impl(cable_cell_impl* c) { return impl_ptr(c, [](cable_cell_impl* p){ delete p; }); }

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

const label_dict& cable_cell::labels() const { return impl_->dictionary; }
const concrete_embedding& cable_cell::embedding() const { return impl_->provider.embedding(); }
const arb::morphology& cable_cell::morphology() const { return impl_->provider.morphology(); }
const mprovider& cable_cell::provider() const { return impl_->provider; }

const region_assignment<density> cable_cell::densities() const { return impl_->densities_; }
const region_assignment<voltage_process> cable_cell::voltage_processes() const { return impl_->voltage_processes_; }
const region_assignment<init_int_concentration> cable_cell::init_int_concentrations() const { return impl_->init_int_concentrations_; }
const region_assignment<init_ext_concentration> cable_cell::init_ext_concentrations() const { return impl_->init_ext_concentrations_; }
const region_assignment<init_reversal_potential> cable_cell::reversal_potentials() const { return impl_->init_reversal_potentials_; }
const region_assignment<ion_diffusivity> cable_cell::diffusivities() const { return impl_->ion_diffusivities_; }
const region_assignment<temperature> cable_cell::temperatures() const { return impl_->temperatures_; }
const region_assignment<init_membrane_potential> cable_cell::init_membrane_potentials() const { return impl_->init_membrane_potentials_; }
const region_assignment<axial_resistivity> cable_cell::axial_resistivities() const { return impl_->axial_resistivities_; }
const region_assignment<membrane_capacitance> cable_cell::membrane_capacitances() const { return impl_->membrane_capacitances_; }

mlocation_list cable_cell::concrete_locset(const locset& l) const { return impl_->concrete_locset(l); }
mextent cable_cell::concrete_region(const region& r) const { return impl_->concrete_region(r); }

const decor& cable_cell::decorations() const { return impl_->decorations; }

const cable_cell_parameter_set& cable_cell::default_parameters() const { return impl_->decorations.defaults(); }

//
const cable_cell::lid_range_map& cable_cell::detector_ranges() const { return impl_->labeled_lid_ranges_[get_index_v<threshold_detector, placeable>]; }
const cable_cell::lid_range_map& cable_cell::synapse_ranges() const { return impl_->labeled_lid_ranges_[get_index_v<synapse, placeable>]; }
const cable_cell::lid_range_map& cable_cell::junction_ranges() const { return impl_->labeled_lid_ranges_[get_index_v<junction, placeable>]; }

const std::unordered_map<std::string, mlocation_map<synapse>>& cable_cell::synapses() const { return impl_->synapses_; }
const std::unordered_map<std::string, mlocation_map<junction>>& cable_cell::junctions() const { return impl_->junctions_; }
const mlocation_map<threshold_detector>& cable_cell::detectors() const { return impl_->detectors_; }
const mlocation_map<i_clamp>& cable_cell::stimuli() const { return impl_->i_clamps_; }

cell_tag_type decor::tag_of(hash_type hash) const {
    if (!hashes_.count(hash)) throw arbor_internal_error{util::pprintf("Unknown hash for {}.", std::to_string(hash))};
    return hashes_.at(hash);
}

} // namespace arb

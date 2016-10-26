#include <cell.hpp>
#include <parameter_list.hpp>

namespace nest {
namespace mc {

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: membrane, HH
 *    memrane resistance: 123 Ω·cm
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

inline cell make_cell_soma_only(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(18.8/2.0);
    soma->mechanism("membrane").set("r_L", 123);
    soma->add_mechanism(hh_parameters());

    if (with_stim) {
        c.add_stimulus({0,0.5}, {10., 100., 0.1});
    }

    return c;
}

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Soma:
 *    mechanisms: HH
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: none
 *    diameter: 1 µm
 *    length: 200 µm
 *    membrane resistance: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cell make_cell_ball_and_stick(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    auto dendrite = c.add_cable(0, segmentKind::dendrite, 1.0/2, 1.0/2, 200.0);
    dendrite->add_mechanism(pas_parameters());
    dendrite->mechanism("membrane").set("r_L", 100);
    dendrite->set_compartments(4);

    if (with_stim) {
        c.add_stimulus({1,1}, {5., 80., 0.3});
    }
    return c;
}

/*
 * Create cell with a soma and unbranched tapered dendrite:
 *
 * Soma:
 *    mechanisms: HH
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: none
 *    diameter proximal: 1 µm
 *    diameter distal: 0.2 µm
 *    length: 200 µm
 *    membrane resistance: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cell make_cell_ball_and_taper(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    auto dendrite = c.add_cable(0, segmentKind::dendrite, 1.0/2, 0.2/2, 200.0);
    dendrite->add_mechanism(pas_parameters());
    dendrite->mechanism("membrane").set("r_L", 100);
    dendrite->set_compartments(4);

    if (with_stim) {
        c.add_stimulus({1,1}, {5., 80., 0.3});
    }
    return c;
}

/*
 * Create cell with a soma and three-segment dendrite with single branch point:
 *
 * O----======
 *
 * Soma:
 *    mechanisms: HH
 *    diameter: 12.6157 µm
 *
 * Dendrites:
 *    mechanisms: membrane
 *    diameter: 1 µm
 *    length: 100 µm
 *    membrane resistance: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of first terminal branch, t=[5 ms, 85 ms), 0.45 nA
 *    end of second terminal branch, t=[40 ms, 50 ms), -0.2 nA
 */

inline cell make_cell_ball_and_3stick(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    // add dendrite of length 200 um and diameter 1 um with passive channel
    c.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism(pas_parameters());
            seg->mechanism("membrane").set("r_L", 100);
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        c.add_stimulus({2,1}, {5.,  80., 0.45});
        c.add_stimulus({3,1}, {40., 10.,-0.2});
    }
    return c;
}

/*
 * Attach voltage probes at each cable mid-point and end-point,
 * and at soma mid-point.
 */

inline cell& add_common_voltage_probes(cell& c) {
    auto ns = c.num_segments();
    for (auto i=0u; i<ns; ++i) {
        c.add_probe({{i, 0.5}, probeKind::membrane_voltage});
        if (i>0) {
            c.add_probe({{i, 1.0}, probeKind::membrane_voltage});
        }
    }
    return c;
}

} // namespace mc
} // namespace nest

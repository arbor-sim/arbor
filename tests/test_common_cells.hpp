#include <cmath>

#include <cell.hpp>
#include <math.hpp>
#include <parameter_list.hpp>

namespace nest {
namespace mc {

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm
 *    capacitance: 0.01 F/m² [default]
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

inline cell make_cell_soma_only(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(18.8/2.0);
    soma->mechanism("membrane").set("r_L", 100);
    soma->add_mechanism(hh_parameters());

    if (with_stim) {
        c.add_stimulus({0,0.5}, {10., 100., 0.1});
    }

    return c;
}

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 200 µm
 *    bulk resistivity: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cell make_cell_ball_and_stick(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    c.add_cable(0, segmentKind::dendrite, 1.0/2, 1.0/2, 200.0);

    for (auto& seg: c.segments()) {
        seg->mechanism("membrane").set("r_L", 100);
        if (seg->is_dendrite()) {
            seg->add_mechanism(pas_parameters());
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        c.add_stimulus({1,1}, {5., 80., 0.3});
    }
    return c;
}

/*
 * Create cell with a soma and unbranched tapered dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter proximal: 1 µm
 *    diameter distal: 0.2 µm
 *    length: 200 µm
 *    bulk resistivity: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cell make_cell_ball_and_taper(bool with_stim = true) {
    cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism(hh_parameters());

    c.add_cable(0, segmentKind::dendrite, 1.0/2, 0.2/2, 200.0);

    for (auto& seg: c.segments()) {
        seg->mechanism("membrane").set("r_L", 100);
        if (seg->is_dendrite()) {
            seg->add_mechanism(pas_parameters());
            seg->set_compartments(4);
        }
    }

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
 * Common properties:
 *    bulk resistivity: 100 Ω·cm
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrites:
 *    mechanisms: passive (default params)
 *    diameter: 1 µm
 *    length: 100 µm
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

    c.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, segmentKind::dendrite, 0.5, 0.5, 100);

    for (auto& seg: c.segments()) {
        seg->mechanism("membrane").set("r_L", 100);
        if (seg->is_dendrite()) {
            seg->add_mechanism(pas_parameters());
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
 * Create 'soma-less' cell with single cable, with physical
 * parameters from Rallpack 1 model.
 *
 * Common properties:
 *    mechanisms: passive
 *        membrane conductance: 0.000025 S/cm² ( =  1/(4Ω·m²) )
 *        membrane reversal potential: -65 mV (default)
 *    diameter: 1 µm
 *    length: 1000 µm
 *    bulk resistivity: 100 Ω·cm
 *    capacitance: 0.01 F/m² [default]
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[0 ms, inf), 0.1 nA
 *
 * Note: zero-volume soma added with same mechanisms, as
 * work-around for some existing fvm modelling issues.
 *
 * TODO: Set the correct values when parameters are generally
 * settable! 
 *
 * We can't currently change leak parameters
 * from defaults, so we scale other electrical parameters
 * proportionally.
 */

inline cell make_cell_simple_cable(bool with_stim = true) {
    cell c;

    c.add_soma(0);
    c.add_cable(0, segmentKind::dendrite, 0.5, 0.5, 1000);

    double r_L  = 100;
    double c_m  = 0.01;
    double gbar = 0.000025;
    double I = 0.1;

    // fudge factor! can't change passive membrane
    // conductance from gbar0 = 0.001

    double gbar0 = 0.001;
    double f = gbar/gbar0;

    // scale everything else
    r_L *= f;
    c_m /= f;
    I /= f;

    for (auto& seg: c.segments()) {
        seg->add_mechanism(pas_parameters());
        seg->mechanism("membrane").set("r_L", r_L);
        seg->mechanism("membrane").set("c_m", c_m);
        // seg->mechanism("pas").set("g", gbar);

        if (seg->is_dendrite()) {
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        // stimulus in the middle of our zero-volume 'soma'
        // corresponds to proximal end of cable.
        c.add_stimulus({0,0.5}, {0., math::infinity<>(), I});
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

#include <cmath>

#include <arbor/cable_cell.hpp>
#include <arbor/segment.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/recipe.hpp>

namespace arb {

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Stimuli:
 *    soma centre, t=[10 ms, 110 ms), 0.1 nA
 */

inline cable_cell make_cell_soma_only(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(18.8/2.0);
    soma->add_mechanism("hh");

    if (with_stim) {
        c.add_stimulus({0,0.5}, {10., 100., 0.1});
    }

    return c;
}

/*
 * Create cell with a soma and unbranched dendrite:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
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
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cable_cell make_cell_ball_and_stick(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    c.add_cable(0, section_kind::dendrite, 1.0/2, 1.0/2, 200.0);

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism("pas");
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
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH (default params)
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    diameter proximal: 1 µm
 *    diameter distal: 0.4 µm
 *    length: 200 µm
 *    bulk resistivity: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cable_cell make_cell_ball_and_taper(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    c.add_cable(0, section_kind::dendrite, 1.0/2, 0.4/2, 200.0);

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism("pas");
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        c.add_stimulus({1,1}, {5., 80., 0.3});
    }
    return c;
}

/*
 * Create cell with a soma and unbranched dendrite with varying diameter:
 *
 * Common properties:
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *
 * Soma:
 *    mechanisms: HH
 *    diameter: 12.6157 µm
 *
 * Dendrite:
 *    mechanisms: passive (default params)
 *    length: 100 µm
 *    membrane resistance: 100 Ω·cm
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[5 ms, 85 ms), 0.3 nA
 */

inline cable_cell make_cell_ball_and_squiggle(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    std::vector<cable_cell::value_type> radii;
    std::vector<cable_cell::point_type> points;

    double length = 100.0;
    int npoints = 200;

    for (int i=0; i<npoints; ++i) {
        double x = i*(1.0/(npoints-1));
        double r = std::exp(-x)*(std::sin(40*x)*0.05+0.1)+0.1;

        radii.push_back(r);
        points.push_back({x*length, 0., 0.});
    };

    auto dendrite =
        make_segment<cable_segment>(section_kind::dendrite, radii, points);
    c.add_cable(0, std::move(dendrite));

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism("pas");
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
 *    bulk resistivity: 100 Ω·cm [default]
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

inline cable_cell make_cell_ball_and_3stick(bool with_stim = true) {
    cable_cell c;

    auto soma = c.add_soma(12.6157/2.0);
    soma->add_mechanism("hh");

    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);
    c.add_cable(1, section_kind::dendrite, 0.5, 0.5, 100);

    for (auto& seg: c.segments()) {
        if (seg->is_dendrite()) {
            seg->add_mechanism("pas");
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
 *    bulk resistivity: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 *    compartments: 4
 *
 * Stimulus:
 *    end of dendrite, t=[0 ms, inf), 0.1 nA
 *
 * Note: zero-volume soma added with same mechanisms, as
 * work-around for some existing fvm modelling issues.
 */

inline cable_cell make_cell_simple_cable(bool with_stim = true) {
    cable_cell c;

    c.default_parameters.axial_resistivity = 100;
    c.default_parameters.membrane_capacitance = 0.01;

    c.add_soma(0);
    c.add_cable(0, section_kind::dendrite, 0.5, 0.5, 1000);

    double gbar = 0.000025;
    double I = 0.1;

    mechanism_desc pas("pas");
    pas["g"] = gbar;

    for (auto& seg: c.segments()) {
        seg->add_mechanism(pas);

        if (seg->is_dendrite()) {
            seg->set_compartments(4);
        }
    }

    if (with_stim) {
        // stimulus in the middle of our zero-volume 'soma'
        // corresponds to proximal end of cable.
        c.add_stimulus({0,0.5}, {0., INFINITY, I});
    }
    return c;
}
} // namespace arb

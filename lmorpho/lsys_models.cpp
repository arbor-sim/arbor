#include <math.hpp>

#include "lsystem.hpp"
#include "lsys_models.hpp"

static constexpr double inf = nest::mc::math::infinity<double>();

// Predefined parameters for two classes of neurons. Numbers taken primarily
// from Ascoli et al. 2001, but some details (soma diameters for example)
// taken from earlier literature.
//
// Without mixture distribution support, these aren't entirely faithful to
// the original sources.
//
// References:
//
// Ascoli, G et al. (2001),
//     "Computer generation and quantitative morphometric analysis of virtual neurons."
//      Anatomy and Embryology 204 (4) 283-301.
//
// Cullheim, S et al. (1987),
//     "Membrane area and dendritic structure in type‐identified triceps surae alpha motoneurons."
//     Journal of comparative neurology 255 (1) 68-81.
//
// Rapp, M et al. (1994).
//     "Physiology, morphology and detailed passive models of guinea-pig cerebellar Purkinje cells."
//     The Journal of Physiology, 474(1), 101-118.

lsys_param make_alpha_motoneuron_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    L.diam_soma = { 47.0, 65.5, 57.6, 5.0 };

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 8.0, 16.0 };

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 3.0, 12.0 };

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 25 };

    // Initial roll (intrinsic rotation about x-axis) [degrees].
    L.roll_initial = { -180.0, 180.0 };

    // Initial pitch (intrinsic rotation about y-axis) [degrees].
    L.pitch_initial = { 0.0, 180.0 };

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { -180.0, 180.0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -15.0, 15.0, 0.0, 5.0 };

    // Taper rate.
    L.taper = { -1.25e-3 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { 0.0, 180.0 };

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 172.0, 45.0, 20.0 };

    // Correlated child branch radius distribution parameters.
    L.diam_child_a = -0.2087;
    L.diam_child_r = { 0.2, 1.8, 0.8255, 0.2125 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 2.62e-2;
    L.pterm_k2 = -2.955;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 2.58e-5;
    L.pbranch_ov_k2 = 2.219;
    L.pbranch_nov_k1 = 2.34e-3;
    L.pbranch_nov_k2 = 0.194;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 2000;

    return L;
}

lsys_param alpha_motoneuron_lsys = make_alpha_motoneuron_lsys();

lsys_param make_purkinje_lsys() {
    lsys_param L;

    // Soma diameter [µm].
    // (Gaussian fit to 3 samples rom Rapp 1994); truncate to 2σ.
    L.diam_soma = { 22.12,  27.68, 24.9, 1.39 };

    // Number of dendritic trees (rounded to nearest int).
    L.n_tree = { 1 };

    // Initial dendrite diameter [µm]. (Dstem)
    L.diam_initial = { 4.8, 7.6, 6.167, 1.069 };

    // Dendrite step length [µm]. (ΔL)
    L.length_step = { 25 };

    // Tortuousness: roll within section over ΔL [degrees].
    L.roll_section = { 0 };

    // Tortuousness: pitch within section over ΔL [degrees].
    L.pitch_section = { -inf, inf, 0, 10 };

    // Taper rate.
    L.taper = { -inf, inf, -0.010e-3, 0.022e-3 };

    // Branching torsion: roll at branch point [degrees].
    L.roll_at_branch = { -inf, inf, 0.0, 13.5 };

    // Branch angle between siblings [degrees].
    L.branch_angle = { 1.0, 179.0, 71.0, 40.0 };

    // Correlated child branch radius distribution parameters.
    // THESE NUMBERS CANNOT BE CORRECT. WTF, ASCOLI?
    L.diam_child_a = -0.2087;
    L.diam_child_r = { 0.01, 0.28, 0.112, 0.035 };

    // Termination probability parameters [1/µm].
    L.pterm_k1 = 0.0355;
    L.pterm_k2 = -0.7237;

    // Bifurcation probability parameters [1/μm].
    L.pbranch_ov_k1 = 0.4539;
    L.pbranch_ov_k2 = 1.533;
    L.pbranch_nov_k1 = 0.2349;
    L.pbranch_nov_k2 = 0.0116;

    // Absolute maximum dendritic extent [µm].
    L.max_extent = 2000;

    return L;
}

lsys_param purkinje_lsys = make_purkinje_lsys();

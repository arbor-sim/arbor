#!/usr/env/bin python
# -*- coding: utf-8 -*-
# Author: Torbjørn Ness <torbjorn.ness@nmbu.no>
"""
NEURON and Python - Creating a multi-compartment model with synaptic input
with randomized activation times
"""
# Import modules for plotting and NEURON itself
import matplotlib.pyplot as plt
import neuron
import numpy as np


class Cell:
    """
    Cell class that handles interactions with NEURON. It finds the centre position
    of each cellular compartment (cell.xmid, cell.ymid, cell.zmid), and the transmembrane currents
    cell.imem
    """

    def __init__(self):
        cvode = neuron.h.CVode()
        cvode.use_fast_imem(1)
        self.tstop = 100.0  # simulation duration in ms
        self.v_init = -65  # membrane voltage(s) at t = 0

        neuron.h.dt = 0.1

        self.make_cell()
        neuron.h.define_shape()
        self.seclist = []
        counter = 0
        for sec in neuron.h.allsec():
            self.seclist.append(sec)
            for seg in sec:
                counter += 1
        self.totnsegs = counter  # Total number of compartments in cell model
        self.collect_geometry()

        self.insert_synapse(self.seclist[0])

        self.initiate_recorders()

    def make_cell(self):
        neuron.h(
            """
        create soma[1]
        create apic[1]
        objref all
        all = new SectionList()
        soma[0] {pt3dclear()
          pt3dadd(0, 0, -10, 20)
          pt3dadd(0, 0, 10, 20)}

        apic[0] {pt3dclear()
          pt3dadd(0, 0, 10, 2)
          pt3dadd(0, 0, 500, 2)}

        connect apic[0](0), soma[0](1)

        apic[0] {nseg = 20}
        forall {
            Ra = 100.
            cm = 1.
            all.append()
        }
        apic[0] { insert pas }
        soma[0] { insert hh }

        """
        )

    def initiate_recorders(self):
        self.imem = []  # Record membrane currents
        self.vmem = []  # Record membrane potentials

        for sec in neuron.h.allsec():
            for seg in sec:
                v_ = neuron.h.Vector()
                v_.record(seg._ref_v, neuron.h.dt)
                self.vmem.append(v_)

                i_ = neuron.h.Vector()
                i_.record(seg._ref_i_membrane_, neuron.h.dt)
                self.imem.append(i_)

        self.tvec = neuron.h.Vector()
        self.tvec.record(neuron.h._ref_t)

        if hasattr(self, "syn"):
            self.syn_i = neuron.h.Vector()  # Record synaptic current
            self.syn_i.record(self.syn._ref_i, neuron.h.dt)

    def insert_synapse(self, syn_sec):
        """
        Function to insert a single synapse into cell model, as section syn_sec
        """
        print(syn_sec.diam)
        syn = neuron.h.ExpSyn(0.5, sec=syn_sec)
        syn.e = 0.0  # reversal potential of synapse conductance in mV
        syn.tau = 2.0  # time constant of synapse conductance in ms

        ns = neuron.h.NetStim(0.5)  # spike time generator object (~presynaptic)
        ns.noise = 1.0  # Fractional randomness (intervals from exp dist)
        ns.start = 0.0  # approximate time of first spike
        ns.number = 1000  # number of spikes
        ns.interval = 10.0  # average interspike interval
        nc = neuron.h.NetCon(ns, syn)  # Connect generator to synapse
        nc.weight[0] = 0.005  # Set synapse weight

        # Everything must be stored or NEURON will forget they ever existed
        self.ns = ns
        self.nc = nc
        self.syn = syn

    def collect_geometry(self):
        """
        Function to get positions, diameters etc of each segment in NEURON
        """

        areavec = np.zeros(self.totnsegs)
        diamvec = np.zeros(self.totnsegs)
        lengthvec = np.zeros(self.totnsegs)

        xstartvec = np.zeros(self.totnsegs)
        xendvec = np.zeros(self.totnsegs)
        ystartvec = np.zeros(self.totnsegs)
        yendvec = np.zeros(self.totnsegs)
        zstartvec = np.zeros(self.totnsegs)
        zendvec = np.zeros(self.totnsegs)

        counter = 0

        # loop over all segments
        for sec in neuron.h.allsec():
            n3d = int(neuron.h.n3d())
            nseg = sec.nseg
            gsen2 = 1.0 / 2 / nseg
            if n3d > 0:
                # create interpolation objects for the xyz pt3d info:
                L = np.zeros(n3d)
                x = np.zeros(n3d)
                y = np.zeros(n3d)
                z = np.zeros(n3d)
                for i in range(n3d):
                    L[i] = neuron.h.arc3d(i)
                    x[i] = neuron.h.x3d(i)
                    y[i] = neuron.h.y3d(i)
                    z[i] = neuron.h.z3d(i)

                # normalize as seg.x [0, 1]
                L /= sec.L

                # temporary store position of segment midpoints
                segx = np.zeros(nseg)
                for i, seg in enumerate(sec):
                    segx[i] = seg.x

                # can't be >0 which may happen due to NEURON->Python float transfer:
                segx0 = (segx - gsen2).round(decimals=6)
                segx1 = (segx + gsen2).round(decimals=6)

                # fill vectors with interpolated coordinates of start and end points
                xstartvec[counter : counter + nseg] = np.interp(segx0, L, x)
                xendvec[counter : counter + nseg] = np.interp(segx1, L, x)

                ystartvec[counter : counter + nseg] = np.interp(segx0, L, y)
                yendvec[counter : counter + nseg] = np.interp(segx1, L, y)

                zstartvec[counter : counter + nseg] = np.interp(segx0, L, z)
                zendvec[counter : counter + nseg] = np.interp(segx1, L, z)

                # fill in values area, diam, length
                for i, seg in enumerate(sec):
                    areavec[counter] = neuron.h.area(seg.x)
                    diamvec[counter] = seg.diam
                    lengthvec[counter] = sec.L / nseg
                    counter += 1

        # starting position of each compartment (segment)
        self.xstart = xstartvec
        self.ystart = ystartvec
        self.zstart = zstartvec

        # ending position of each compartment (segment)
        self.xend = xendvec
        self.yend = yendvec
        self.zend = zendvec

        # Calculates the centre position of each compartment (segment)
        self.xmid = 0.5 * (self.xstart + self.xend)
        self.ymid = 0.5 * (self.ystart + self.yend)
        self.zmid = 0.5 * (self.zstart + self.zend)
        self.area = areavec
        self.diam = diamvec

    def simulate(self):

        neuron.h.finitialize(self.v_init)
        neuron.h.fcurrent()

        while neuron.h.t < self.tstop:
            neuron.h.fadvance()

        self.vmem = np.array(self.vmem)
        self.imem = np.array(self.imem)
        self.syn_i = np.array(self.syn_i)
        self.tvec = np.array(self.tvec)[: self.vmem.shape[1]]


class ExtElectrode:
    def __init__(self, elec_x, elec_y, elec_z):
        """

        :param elec_x, elec_y, elec_z : x,y,z-positions (um) of each electrode. Must
        be numpy arrays of equal length

        """
        self.sigma = 0.3  # Extracellular conductivity (S/m)
        self.elec_x = elec_x
        self.elec_y = elec_y
        self.elec_z = elec_z
        self.num_elecs = len(self.elec_x)

        # Give electrodes different colors for plotting purposes:
        self.elec_clr = lambda idx: plt.cm.viridis(idx / self.num_elecs)

    def calc_extracellular_potential(self, cell):
        self.calc_mapping(cell)
        self.extracellular_potential = np.dot(electrode.mapping, cell.imem)

    def calc_mapping(self, cell):
        """
        Calculates 'mapping' of size (number of electrodes) * (number of cell compartments)
        Extracellular potential can then be calculated as
        :param cell: class containing x,y,z-positions (um) of each
        compartment (segment) centre as cell.xmid, cell.ymid, cell.zmid

        """
        self.mapping = np.zeros((self.num_elecs, cell.totnsegs))
        for e_idx in range(self.num_elecs):
            r2 = (
                (cell.xmid - self.elec_x[e_idx]) ** 2
                + (cell.ymid - self.elec_y[e_idx]) ** 2
                + (cell.zmid - self.elec_z[e_idx]) ** 2
            )

            self.mapping[e_idx] = 1 / (4 * np.pi * self.sigma * np.sqrt(r2))


def plot_results(cell, electrode):
    ################################################################################
    # Plot simulated output
    ################################################################################
    fig = plt.figure(figsize=(9, 5))
    fig.subplots_adjust(wspace=0.5, hspace=0.9)
    ax_morph = fig.add_subplot(
        131,
        aspect=1,
        xlim=[-150, 150],
        ylim=[-100, 600],
        title="morphology",
        xlabel=r"x ($\mu$m)",
        ylabel=r"y ($\mu$m)",
    )
    ax_syn = fig.add_subplot(
        332, ylabel="nA", title="synaptic current", xlabel="time (ms)"
    )
    ax_vmem = fig.add_subplot(
        335, ylabel="mV", xlabel="time (ms)", title="membrane potential"
    )
    ax_imem = fig.add_subplot(
        338, ylabel="nA", xlabel="time (ms)", title="membrane current"
    )
    ax_ep = fig.add_subplot(
        133, ylabel=r"$\mu$V", xlabel="time (ms)", title="Extracellular potential"
    )

    plot_comp_idx = 0
    plot_comp_clr = "r"

    for idx in range(cell.totnsegs):
        ax_morph.plot(
            [cell.xstart[idx], cell.xend[idx]],
            [cell.zstart[idx], cell.zend[idx]],
            lw=cell.diam[idx] / 2,
            c="k",
        )
    ax_morph.plot(
        cell.xmid[plot_comp_idx], cell.zmid[plot_comp_idx], marker="*", c=plot_comp_clr
    )

    ax_syn.plot(cell.tvec, cell.syn_i, c="k", lw=2)
    ax_vmem.plot(cell.tvec, cell.vmem[0, :], c=plot_comp_clr, lw=2)
    ax_imem.plot(cell.tvec, cell.imem[0, :], c=plot_comp_clr, lw=2)

    for e_idx in range(electrode.num_elecs):
        e_clr = electrode.elec_clr(e_idx)
        sig = 1000 * electrode.extracellular_potential[e_idx]  # convert to uV
        ax_ep.plot(cell.tvec, sig, c=e_clr)
        ax_morph.plot(
            electrode.elec_x[e_idx], electrode.elec_z[e_idx], marker="o", c=e_clr
        )

    fig.savefig("example_nrn_EP.png")

    plt.close(fig)


if __name__ == "__main__":
    cell = Cell()
    cell.simulate()

    num_elecs = 2
    elec_x = 30 * np.ones(num_elecs)
    elec_y = np.zeros(num_elecs)
    elec_z = np.linspace(0, 100, num_elecs)

    electrode = ExtElectrode(elec_x, elec_y, elec_z)
    electrode.calc_extracellular_potential(cell)

    plot_results(cell, electrode)

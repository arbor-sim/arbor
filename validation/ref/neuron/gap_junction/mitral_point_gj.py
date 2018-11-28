#!/usr/bin/env python
#coding: utf-8

import math
from neuron import h
from builtins import range

# cell is composed of a single soma with nax, kamt and kdrmt mechanisms
class cell:
    def __init__(self, gid, params=None):
        self.pc = h.ParallelContext()
        self.soma = None
        self.gid = gid
        self.sections = {}
        self.stims = []
        self.halfgap_list = []

        self.add_soma()

        geom = [(0, 3), (300, 3)]
        self.add_dendrite('stick0', geom, 200)

        geom_min = [(0, 2), (100, 2)]
        self.add_dendrite('min_stick0', geom_min, 50)
        self.add_dendrite('min_stick1', geom_min, 50)

        geom_hill = [(0, 20), (5, 1.5)]
        self.add_dendrite('hillock0', geom_hill, 50)

        geom_init = [(0, 1.5), (30, 1.5)]
        self.add_seg('seg0', geom_init, 50, 'hillock0')

        geom_tuft = [(0, 0.4), (300, 0.4)]
        self.add_dendrite('tuft0', geom_tuft, 100, 'stick0', 20)

    def add_soma(self):
        soma = h.Section(name='soma', cell=self)
        h.celsius = 35
        soma.diam = 20
        soma.L = 25

        soma.Ra = 100
        soma.cm = 1.8

        # Insert active nax channels in the soma.
        soma.insert('nax')
        soma.gbar_nax = 0.04
        soma.sh_nax = 10

        soma.insert('kamt')
        soma.gbar_kamt = 0.004

        soma.insert('kdrmt')
        soma.gbar_kdrmt = 0.0001

        soma.insert('pas')
        soma.g_pas = 1.0/12000.0
        soma.e_pas = -65

        soma.ena = 50
        soma.ek = -90

        self.soma = soma

        self.ncomp=1
        self.nseg=1

        self.sections['soma'] = soma

    def add_dendrite(self, name, geom, ncomp, to=None, rallbranch=0):
        dend = h.Section(name=name)
        dend.push()
        for x, d in geom:
            h.pt3dadd(x, 0, 0, d)
        h.pop_section()

        if rallbranch != 0:
            dend.rallbranch = rallbranch

        dend.Ra = 100
        dend.cm = 1.8

        # Insert channels in the soma.
        dend.insert('nax')
        dend.gbar_nax = 0.04
        dend.sh_nax = 10

        dend.insert('kamt')
        dend.gbar_kamt = 0.004

        dend.insert('kdrmt')
        dend.gbar_kdrmt = 0.0001

        dend.insert('pas')
        dend.g_pas = 1.0/12000.0
        dend.e_pas = -65

        dend.ena = 50
        dend.ek = -90

        dend.nseg = ncomp

        if to is None:
            if self.soma is not None:
                dend.connect(self.soma(1))
        else:
            dend.connect(self.sections[to](1))

        self.sections[name] = dend

    def add_seg(self, name, geom, ncomp, to=None):
        seg = h.Section(name=name)
        seg.push()
        for x, d in geom:
            h.pt3dadd(x, 0, 0, d)
        h.pop_section()

        seg.Ra = 100
        seg.cm = 1.8

        # Insert channels in the soma.
        seg.insert('nax')
        seg.gbar_nax = 0.4
        seg.sh_nax = 0

        seg.insert('kamt')
        seg.gbar_kamt = 0.04

        seg.insert('kdrmt')
        seg.gbar_kdrmt = 0.0001

        seg.insert('pas')
        seg.g_pas = 1.0/1000.0
        seg.e_pas = -65

        seg.ena = 50
        seg.ek = -90

        seg.nseg = ncomp

        if to is None:
            if self.soma is not None:
                seg.connect(self.soma(1))
        else:
            seg.connect(self.sections[to](1))

        self.sections[name] = seg

    def add_iclamp(self, t0, dt, i, to=None, pos=1):
        # If no section specified, attach to middle of soma
        if to is None:
            sec = self.soma
            pos = 0.5
        else:
            sec = self.sections[to]

        stim = h.IClamp(sec(pos))
        stim.delay = t0
        stim.dur = dt
        stim.amp = i
        self.stims.append(stim)

    # Add a gap_junction between self and other
    # the voltages of 'self' and 'other' need to be visible to the other cell's half gap junction
    # 'source_var' assigns a voltage variable to a unique id
    # 'target_var' attaches a voltage variable (identified using its unique id) to another voltage variable
    # to expose the voltage of 'self' to the half gap_junction at 'other':
    # 1. assign the voltage of a sec on 'self' to a unique id (cell gid) using 'source_var'
    # 2. attach the voltage of the half gap_junction at 'other' to the voltage of a sec of 'self'
    #    using 'target_var' and the unique id (gid) of a sec of 'self'
    def add_point_gap(self, other, ggap, name_sec1=None, name_sec2=None):  #ggap in nS
        if self.pc.gid_exists(self.gid):
            self.mk_halfgap(other, ggap, name_sec1)

        if self.pc.gid_exists(other.gid):
            other.mk_halfgap(self, ggap, name_sec2)

    # assign the voltage at a sec to the gid of the cell
    # create half gap_junction at a sec, and assign its variables: vgap and g
    # vgap gets the voltage assigned to the gid of the 'other' cell
    # g gets ggap
    def mk_halfgap(self, other, ggap, name_sec):
        # sec seg
        if name_sec==None:
            sec_seg = self.pc.gid2cell(self.gid).soma(.5)
        else:
            sec_seg = self.pc.gid2cell(self.gid).sections[name_sec](.95)

        # assign the voltage at the soma to the gid of the cell
        self.pc.source_var(sec_seg._ref_v, self.gid, sec=sec_seg.sec)

        # create half gap_junction on the soma
        hg = h.HalfGap(sec_seg)

        # attach vgap to the voltage assigned for the 'other' cell's gid
        self.pc.target_var(hg, hg._ref_vgap, other.gid)

        # set the conductance of the half gap_junction
        # must match the second half of the gap_junction
        hg.g = ggap

        # save the state
        self.halfgap_list.append(hg)

    def set_recorder(self):
        # set soma and time recording vectors on the cell.
        # return: the soma and time vectors as a tuple.
        soma_v = h.Vector()   # Membrane potential vector at soma
        t = h.Vector()        # Time stamp vector
        soma_v.record(self.soma(0.5)._ref_v)
        t.record(h._ref_t)
        return soma_v, t



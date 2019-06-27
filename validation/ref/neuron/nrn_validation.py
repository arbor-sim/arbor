#!/usr/bin/env python
#coding: utf-8

import json
import sys
import os
import re
import numpy as np
import neuron
from neuron import h

try:
    from builtins import range
except ImportError:
    from __builtin__ import range

# This is super annoying: without neuron.gui, need
# to explicit load 'standard' hoc routines like 'run',
# but this is chatty on stdout, which means we get
# junk in our data if capturing output.

def hoc_execute_quiet(arg):
    with open(os.devnull, 'wb') as null:
        fd = sys.stdout.fileno()
        keep = os.dup(fd)
        sys.stdout.flush()
        os.dup2(null.fileno(), fd)
        h(arg)
        sys.stdout.flush()
        os.dup2(keep, fd)

def hoc_setup():
    hoc_execute_quiet('load_file("stdrun.hoc")')

def hoc_quit():
    hoc_execute_quiet('quit()')
    #h('quit()')

default_model_parameters = {
    'gnabar_hh':  0.12,   # H-H sodium conductance in S/cm^2
    'gkbar_hh':   0.036,  # H-H potassium conductance in S/cm^2
    'gl_hh':      0.0003, # H-H leak conductance in S/cm^2
    'el_hh':    -54.3,    # H-H reversal potential in mV
    'g_pas':      0.001,  # Passive conductance in S/cm^2
    'e_pas':    -65.0,    # Leak reversal potential in mV
    'Ra':       100.0,    # Intracellular resistivity in Ω·cm
    'cm':         1.0,    # Membrane areal capacitance in µF/cm^2
    'tau':        2.0,    # Exponential synapse time constant
    'tau1':       0.5,    # Exp2 synapse tau1
    'tau2':       2.0,    # Exp2 synapse tau2
    'ncomp':   1001,      # Number of compartments (nseg) in dendrites
    'dt':         0.0,    # (Simulation parameter) default dt, 0 => use cvode adaptive
    'abstol':     1e-6    # (Simulation parameter) abstol for cvode if used
}

def override_defaults_from_args(args=sys.argv):
    global default_model_parameters
    keys = default_model_parameters.keys()
    r = re.compile('('+'|'.join(keys)+')=(.*)')
    for m in [r.match(a) for a in args]:
        if m:
            default_model_parameters[m.group(1)]=float(m.group(2))

def combine(*dicts, **kw):
    r = {}
    for d in dicts:
        r.update(d)
    r.update(kw)
    return r

class VModel:
    def __init__(self):
        self.soma = None
        self.sections = {}
        self.stims = []
        self.synapses = []
        self.netcons = []

    def set_ncomp(self, n):
        for s in self.sections.values():
            s.nseg = int(n)

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

    def add_exp_syn(self, secname, pos=0.5, **kw):
        p = combine(default_model_parameters, kw)

        syn = h.ExpSyn(self.sections[secname](pos))
        syn.tau = p['tau']
        self.synapses.append(syn)
        return len(self.synapses)-1

    def add_exp2_syn(self, secname, pos=0.5, **kw):
        p = combine(default_model_parameters, kw)

        syn = h.Exp2Syn(self.sections[secname](pos))
        syn.tau1 = p['tau1']
        syn.tau2 = p['tau2']
        self.synapses.append(syn)
        return len(self.synapses)-1

    def add_spike(self, t, weight, target=0):
        stim = h.NetStim()
        stim.number = 1
        stim.start  = 0

        nc = h.NetCon(stim, self.synapses[target])
        nc.delay = t
        nc.weight[0] = weight

        self.stims.append(stim)
        self.netcons.append(nc)

    def add_soma(self, diam, **kw):
        p = combine(default_model_parameters, kw)

        soma = h.Section(name='soma')
        soma.diam = diam
        soma.L = diam

        soma.Ra = p['Ra']
        soma.cm = p['cm']

        # Insert active Hodgkin-Huxley channels in the soma.
        soma.insert('hh')
        soma.gnabar_hh = p['gnabar_hh']
        soma.gkbar_hh = p['gkbar_hh']
        soma.gl_hh = p['gl_hh']
        soma.el_hh = p['el_hh']

        # For reversal potentials we use those computed using
        # the Nernst equation with the following values:
        #       R   8.3144598
        #       F   96485.33289
        #       nao 140   mM
        #       nai  10   mM
        #       ko    2.5 mM
        #       ki   64.4 nM
        # We don't use the default values for ena and ek taken
        # from the HH paper:
        #   ena    = 115.0mV + -65.0mV,
        #   ek     = -12.0mV + -65.0mV,
        soma.ena =  63.55148117386
        soma.ek  = -74.17164678272

        # This is how we would get NEURON to use Nernst equation, when they
        # correct the Nernst equation implementation.
        #h.ion_style('k_ion', 3, 2, 1, 1, 1)
        #h.ion_style('na_ion', 3, 2, 1, 1, 1)

        self.soma = soma

    def add_dendrite(self, name, geom, to=None, **kw):
        p = combine(default_model_parameters, kw)

        dend = h.Section(name=name)
        dend.push()
        for x, d in geom:
            h.pt3dadd(x, 0, 0, d)
        h.pop_section()

        dend.Ra = p['Ra']
        dend.cm = p['cm']

        # Add passive membrane properties to dendrite.
        dend.insert('pas')
        dend.g_pas = p['g_pas']
        dend.e_pas = p['e_pas']

        dend.nseg = int(p['ncomp'])

        if to is None:
            if self.soma is not None:
                dend.connect(self.soma(1))
        else:
            dend.connect(self.sections[to](1))

        self.sections[name] = dend

# Run 'current' model, return list of traces.
# Samples at cable mid- and end-points taken every `sample_dt`;
# Voltage on all compartments per section reported every `report_dt`.

def run_nrn_sim(tend, sample_dt=0.025, report_t=None, report_dt=None, dt=None, **meta):
    if dt is None:
        dt = default_model_parameters['dt']

    # Instrument mid-point and ends of each section for traces.
    vtraces = []
    vtrace_t_hoc = h.Vector()

    ncomps = set([s.nseg for s in h.allsec() if s.name()!='soma'])
    if len(ncomps)==1:
        common_ncomp = { 'ncomp': ncomps.pop() }
    else:
        common_ncomp = {}

    for s in h.allsec():
        vend = h.Vector()
        vend.record(s(0.5)._ref_v, sample_dt)
        vtraces.append((s.name()+".mid", vend))
        if s.nseg!=1 or s.name()!='soma':
            vmid = h.Vector()
            vmid.record(s(1.0)._ref_v, sample_dt)
            vtraces.append((s.name()+".end", vmid))

    vtrace_t_hoc.record(h._ref_t, sample_dt)

    # Instrument every segment for section voltage reports.
    if report_t is None:
        if report_dt is not None:
            report_t = [report_dt*(1+i) for i in range(int(tend/report_dt))]
        else:
            report_t = []
    elif not isinstance(report_t, list):
        report_t = [report_t]

    vreports = []
    vreport_t_hoc = h.Vector(report_t)

    if report_t:
        for s in h.allsec():
            nseg = s.nseg;
            ps = [0] + [(i+0.5)/nseg for i in range(nseg)] + [1]
            vs = [h.Vector() for p in ps]
            for p, v in zip(ps, vs):
                v.record(s(p)._ref_v, vreport_t_hoc)
            vreports.append((s.name(), s.L, s.nseg, ps, vs))

    # Run sim
    if dt==0:
        # Use CVODE instead
        h.cvode.active(1)
        abstol = default_model_parameters['abstol']
        h.cvode.atol(abstol)
        common_meta = { 'dt': 0, 'cvode': True, 'abstol': abstol }
    else:
        h.dt = dt
        h.steps_per_ms = 1/dt # or else NEURON might noisily fudge dt
        common_meta = { 'dt': dt, 'cvode': False }

    h.secondorder = 2
    h.tstop = tend
    h.run()

    # convert results to traces with metadata
    traces = []

    vtrace_t = list(vtrace_t_hoc)
    traces.append(combine(common_meta, meta, common_ncomp, {
        'name':  'membrane voltage',
        'sim':   'neuron',
        'units': 'mV',
        'data':  combine({n: list(v) for n, v in vtraces}, time=vtrace_t)
    }))

    # and section reports too
    vreport_t = list(vreport_t_hoc)
    for name, length, nseg, ps, vs in vreports:
        obs = np.column_stack([np.array(v) for v in vs])
        xs = [length*p for p in ps]
        for i, t in enumerate(report_t):
            if i>=obs.shape[0]:
                break

            traces.append(combine(common_meta, meta, {
                'name': 'membrane voltage',
                'sim':  'neuron',
                'units': {'x': 'µm', name: 'mV'},
                'ncomp': nseg,
                'time': t,
                'data': {
                    'x': xs,
                    name: list(obs[i,:])
                }
            }))

    return traces

def nrn_assert_no_sections():
    for s in h.allsec():
        assert False, 'a section exists'

def nrn_stop():
    hoc_quit()

# Run hoc setup on load
hoc_setup()


#!env python

import argparse
import json
import numpy as np
import sys
import re
import matplotlib as M
import matplotlib.pyplot as P
from itertools import chain, islice, cycle

# Read timeseries data from multiple files, plot each in one planel, with common
# time axis, and optionally sharing a vertical axis as governed by the configuration.

def parse_clargs():
    def float_or_none(s):
        try: return float(s)
        except ValueError: return None

    def parse_range_spec(s):
        l, r = (float_or_none(x) for x in s.split(','))
        return (l,r)

    P = argparse.ArgumentParser(description='Plot time series data on one or more graphs.')
    P.add_argument('inputs', metavar='FILE', nargs='+',
                   help='time series data in JSON format')
    P.add_argument('-t', '--abscissae', metavar='RANGE', dest='trange',
                   type=parse_range_spec, 
                   help='restrict time axis to RANGE (see below)')
    P.add_argument('-g', '--group', metavar='KEY,...',  dest='groupby',
                   type=lambda s: s.split(','), 
                   help='plot series with same KEYs on the same axes')

    P.add_argument('-o', '--out', metavar='FILE',  dest='outfile',
                   help='save plot to file FILE')

    P.epilog =  'A range is specifed by a pair of floating point numbers min,max where '
    P.epilog += 'either may be omitted to indicate the minimum or maximum of the corresponding '
    P.epilog += 'values in the data.'

    # modify args to avoid argparse having a fit when it encounters an option
    # argument of the form '<negative number>,...'

    argsbis = [' '+a if re.match(r'-[\d.]',a) else a for a in sys.argv[1:]]
    return P.parse_args(argsbis)

def isstring(s):
    return isinstance(s,str) or isinstance(s,unicode)

def take(n, s):
    return islice((i for i in s), 0, n)

class TimeSeries:
    def __init__(self, ts, ys, **kwargs):
        self.t = np.array(ts)
        n = self.t.shape[0]

        self.y = np.full_like(self.t, np.nan)
        ny = min(len(ys), len(self.y))
        self.y[:ny] = ys[:ny]

        self.meta = dict(kwargs)

    def tclip(self, bounds):
        clip = range_meet(self.trange(), bounds)
        self.t = np.ma.masked_outside(self.t, v1=clip[0], v2=clip[1])
        self.y = np.ma.masked_where(np.ma.getmask(self.t), self.y)

    def name(self):
        return self.meta.get('name',"")   # value of 'name' attribute in source

    def label(self):
        return self.meta.get('label',"")  # name of column in source

    def units(self):
        return self.meta.get('units',"")
        
    def trange(self):
        return (min(self.t), max(self.t))

    def yrange(self):
        return (min(self.y), max(self.y))


def read_json_timeseries(source):
    j = json.load(source)

    # Convention:
    #
    # Time series data is represented by an object with a subobject 'data' and optionally
    # other key/value entries comprising metadata for the time series.
    #
    # The 'data' object holds one array of numbers 'time' and zero or more other
    # numeric arrays of sample values corresponding to the values in 'time'. The
    # names of these other arrays are taken to be the labels for the plots.
    #
    # Units can be specified by a top level entry 'units' which is either a string
    # (units are common for all data series in the object) or by a map that
    # takes a label to a unit string.

    ts_list = []
    jdata = j['data']
    ncol = len(jdata)

    times = jdata['time']
    nsample = len(times)

    def units(label):
        try:
           unitmap = j['units']
           if isstring(unitmap):
               return unitmap
           else:
               return unitmap[label]
        except:
           return ""

    i = 1
    for key in jdata.keys():
        if key=="time": continue

        meta = dict(j.items() + {'label': key, 'data': None, 'units': units(key)}.items())
        del meta['data']

        ts_list.append(TimeSeries(times, jdata[key], **meta))

    return ts_list
        
def range_join(r, s):
    return (min(r[0], s[0]), max(r[1], s[1]))

def range_meet(r, s):
    return (max(r[0], s[0]), min(r[1], s[1]))


class PlotData:
    def __init__(self, key_label=""):
        self.series = []
        self.group_key_label = key_label

    def trange(self):
        return reduce(range_join, [s.trange() for s in self.series])

    def yrange(self):
        return reduce(range_join, [s.yrange() for s in self.series])

    def name(self):
        return reduce(lambda n, s: n or s.name(), self.series, "")

    def group_label(self):
        return self.group_key_label

    def unique_labels(self):
        # attempt to create unique labels for plots in the group based on
        # meta data
        labels = [s.label() for s in self.series]
        if len(labels)<2:
            return labels

        n = len(labels)
        keyset = reduce(lambda k, s: k.union(s.meta.keys()), self.series, set())
        keyi = iter(keyset)
        try:
            while len(set(labels)) != n:
                k = next(keyi)
                if k=='label':
                    continue

                vs = [s.meta.get(k,None) for s in self.series]
                if len(set(vs))==1:
                    continue

                for i in xrange(n):
                    prefix = '' if k=='name' else k+'='
                    if vs[i] is not None:
                        labels[i] += ', '+k+'='+str(vs[i])

        except StopIteration:
            pass

        return labels

# Input: list of TimeSeries objects; collection of metadata keys to group on
# Return list of plot info (time series, data extents, metadata), one per plot.

def gather_ts_plots(tss, groupby):
    group_lookup = {}
    plot_groups = []

    for ts in tss:
        key = tuple([ts.meta.get(g) for g in groupby])
        if key is () or None in key or key not in group_lookup:
            pretty_key=', '.join([str(k)+'='+str(v) for k,v in zip(groupby, key) if v is not None])
            pd = PlotData(pretty_key)
            pd.series = [ts]
            plot_groups.append(pd)
            group_lookup[key] = len(plot_groups)-1
        else:
            plot_groups[group_lookup[key]].series.append(ts)

    return plot_groups


def make_palette(cm_name, n, cmin=0, cmax=1):
    smap = M.cm.ScalarMappable(M.colors.Normalize(cmin/float(cmin-cmax),(cmin-1)/float(cmin-cmax)),
                               M.cm.get_cmap(cm_name))
    return [smap.to_rgba((2*i+1)/float(2*n)) for i in xrange(n)]

def plot_plots(plot_groups, save=None):
    nplots = len(plot_groups)
    plot_groups = sorted(plot_groups, key=lambda g: g.group_label())

    # use same global time scale for all plots
    trange = reduce(range_join, [g.trange() for g in plot_groups])

    # use group names for titles?
    group_titles = any((g.group_label() for g in plot_groups))

    figure = P.figure()
    for i in xrange(nplots):
        group = plot_groups[i]
        plot = figure.add_subplot(nplots, 1, i+1)

        title = group.group_label() if group_titles else group.name()
        plot.set_title(title)

        # y-axis label: use timeseries label and units if only
        # one series in group, otherwise use a legend with labels,
        # and units alone on the axes. At most two different unit
        # axes can be drawn.

        def ylabel(unit):
            if len(group.series)==1:
                lab = group.series[0].label()
                if unit:
                    lab += ' (' + unit + ')'
            else:
                lab = unit

            return lab
            
        uniq_units = list(set([s.units() for s in group.series]))
        uniq_units.sort()
        if len(uniq_units)>2:
            logger.warning('more than two different units on the same plot')
            uniq_units = uniq_units[:2]

        # store each series in a slot corresponding to one of the units,
        # together with a best-effort label

        series_by_unit = [[] for i in xrange(len(uniq_units))]
        unique_labels = group.unique_labels()

        for si in xrange(len(group.series)):
            s = group.series[si]
            label = unique_labels[si]
            try:
                series_by_unit[uniq_units.index(s.units())].append((s,label))
            except ValueError:
                pass

        # TODO: need to find a scheme of colour/line allocation that is
        # double y-axis AND greyscale friendly.

        palette = \
            [make_palette(cm, n, 0, 0.5) for
                cm, n in zip(['hot', 'winter'],  [len(x) for x in series_by_unit])]

        lines = cycle(["-",(0,(3,1))])
        
        first_plot = True
        for ui in xrange(len(uniq_units)):
            if not first_plot:
                plot = plot.twinx()

            axis_color = palette[ui][0]
            plot.set_ylabel(ylabel(uniq_units[ui]), color=axis_color)
            for l in plot.get_yticklabels():
                l.set_color(axis_color)

            plot.get_yaxis().get_major_formatter().set_useOffset(False)
            plot.get_yaxis().set_major_locator(M.ticker.MaxNLocator(nbins=6))
            
            plot.set_xlim(trange)

            colours = cycle(palette[ui])
            line = next(lines)
            for s, l in series_by_unit[ui]:
                plot.plot(s.t, s.y, color=next(colours), ls=line, label=l)

            if first_plot:
                plot.legend(loc=2, fontsize='small')
                plot.grid()
            else:
                plot.legend(loc=1, fontsize='small')

            first_plot = False

    # adapted from http://stackoverflow.com/questions/6963035
    axis_ymin = min([ax.get_position().ymin for ax in figure.axes])
    figure.text(0.5, axis_ymin - float(3)/figure.dpi, 'time', ha='center', va='center')
    if save:
        figure.savefig(save)
    else:
        P.show()
        
args = parse_clargs()
tss = []
for filename in args.inputs:
    with open(filename) as f:
        tss.extend(read_json_timeseries(f))

if args.trange:
    for ts in tss:
        ts.tclip(args.trange)

groupby = args.groupby if args.groupby else []
plots = gather_ts_plots(tss, groupby)

if not args.outfile:
    M.interactive(False)

plot_plots(plots, save=args.outfile)

#!/usr/bin/env python3
#coding: utf-8

import json
import argparse
import re
import numpy as np
from itertools import chain

def parse_clargs():
    P = argparse.ArgumentParser(description='Aggregate and analyse MPI profile output.')
    P.add_argument('inputs', metavar='FILE', nargs='+',
                   help='MPI profile output in JSON format')
    P.add_argument('-r', '--raw', action='store_true',
                   help='emit raw times in csv table')

    return P.parse_args()

def parse_profile_json(source):
    j = json.load(source)
    rank = j['rank']
    if rank is None:
        raise ValueError('missing rank information in profile')

    tx = dict()

    def collect_times(j, prefix):
        t = j['time']
        n = j['name']

        if t is None or n is None:
            return

        prefix = prefix + n
        tx[prefix] = t

        try:
            children = j['regions']
            # special case for top level
            if prefix == 'total':
                prefix = ''
            else:
                prefix = prefix + '/'

            for j in children:
                collect_times(j, prefix)
        except KeyError:
            pass

    collect_times(j['regions'], '')
    return rank, tx

def csv_escape(x):
    s = re.sub('"','""',str(x))
    if re.search('["\t\n,]',s):
        s = '"'+s+'"'
    return s

def emit_csv(cols, rows, stdout):
    stdout.write(",".join([csv_escape(c) for c in cols]))
    stdout.write("\n")
    for r in rows:
        stdout.write(",".join([csv_escape(r[c]) if c in r else '' for c in cols]))
        stdout.write("\n")

def main(raw, inputs, stdout):
    rank_times = dict()
    for filename in inputs:
        with open(filename) as f:
            rank, times = parse_profile_json(f)
            rank_times[rank] = times

    if raw:
        rows = [rank_times[rank] for rank in sorted(rank_times.keys())]
        cols = sorted({col for tbl in rows for col in tbl.keys()})
        emit_csv(cols, rows, stdout)
    else:
        rank_entry = [rank_times[rank] for rank in sorted(rank_times.keys())]
        bins = sorted({col for tbl in rank_entry for col in tbl.keys()})

        rows = []
        for b in bins:
            qs = np.percentile([entry[b] for entry in rank_times.values() if b in entry],
                [0., 0.25, 0.5, 0.75, 1.])
            rows.append({
                'region': b,
                'min': qs[0],
                'q25': qs[1],
                'median': qs[2],
                'q75': qs[3],
                'max': qs[4]
            })

        emit_csv(['region','min','q25','median','q75','max'], rows, stdout)

if __name__ == "__main__":
    import sys
    args = parse_clargs()
    main(args.raw, args.inputs, sys.stdout)

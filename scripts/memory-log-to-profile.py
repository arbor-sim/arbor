#!/usr/bin/env python3

import svgwrite as S
import sys
import subprocess as sp

IND = 1
LEN = 100
LIM = 64*1024

def demangle(s):
    res = sp.run(args=['/opt/homebrew/Cellar/binutils/2.40/bin/c++filt', s], capture_output=True).stdout
    res = res.decode('ascii').strip()
    return res


def filter(s):
    bl = ['std::__1::__function', 'std::__1::function', '__invoke', 'arb::threading::task_', 'std::__1::__allocation_result', 'std::__1::__libcpp_allocate', 'std::__1::__split_buffer', 'std::__1::allocator', 'std::__1::__wrap_iter', 'arb::threading::priority_task']
    # bl = []
    return any(b in s for b in bl)


class Node:
    def __init__(self, name) -> None:
        self.name = name
        self.size = 0
        self.count = 0
        self.children = {}

    def add(self, names, size):
        self.size += size
        self.count += 1
        name, *rest = names
        if not name in self.children:
            self.children[name] = Node(name)
        if rest:
            self.children[name].add(rest, size)

    def print(self, off=0, ind=0):
        if self.size <= off:
            return
        name = demangle(self.name)
        if len(name) > LEN:
            name = name[:LEN] + '...'
        if not filter(name):
            print(f"{self.size//1024:<10}{self.count:<10}{'* '*ind}{name}")
            for v in sorted(self.children.values(), key=lambda n: -n.size):
                v.print(off, ind + IND)
        else:
            for v in sorted(self.children.values(), key=lambda n: -n.size):
                v.print(off, ind)

prof = Node("root")

with open(sys.argv[1]) as fd:
    for ln in fd:
        ln = ln.strip()
        if not ln:
            continue
        try:
            size, *stack = ln.split(':')
            prof.add(stack[::-1], int(size))
        except:
            print('Skipping:', ln, file=sys.stderr)

print("Size/MB   Count     ")
prof.print(LIM)

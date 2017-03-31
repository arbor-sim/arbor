# Flat and Interleaved Matrix Storage

This document describes the layout of different storage schemes for matrices use in the GPU back end.

An NxN Hines matrix can be stored very compactly with 3 vectors of length N:
    * `d`: the diagonal of the matrix
    * `u`: the upper part of the matrix (referred to somewhat casually as the super diagonal)
    * `p`: the parent index
Additionally, we often store N*1 vectors that have one value per compartment, e.g. voltage,
solution or rhs vectors.

In NestMC multicompartment a single cell has a matrix structure associated with in, that is derived directly from the connections between its constituent compartments. NestMC groups these cells into groups of cells, called `cell_groups`. The matrices for all the cells in a group are packed together 

The matrix packing applies the same packing operation to each vector associated with a matrix, i.e. the `u`, `d`, `p` and solution, voltage vectors.

In this discussion we use a simple example group of matrices to illustrate the storage methods, because an example is more illustrative than a formal description:
    * 7 vectors labeled {a, b, c, d, e, f, g}
    * the vectors have respective lenghts {8, 7, 6, 6, 5, 5, 3}
    * the ith value in vector a is labelled ai

## Flat storage

take a vector v containing the values 

vals = [a0 a1 a2 a3 a4 a5 a6 a7 | b0 b1 b2 b3 b4 b5 b6 | c0 c1 c2 c3 c4 c5 | d0 d1 d2 d3 d4 d5 | e0 e1 e2 e3 e4 | f0 f1 f2 f3 f4 | g0 g1 g2 ]
indx = [0, 8, 15, 21, 27, 32, 37, 40]

To look up the value of the ith entry in the vector m, we use the following formula to calculate the index

lookup_flt(i,m): indx[m] + i

## Interleaved storage

Block width 4 and padded matrix size of 8:

vals =
[ a0 b0 c0 d0 | a1 b1 c1 d1 | a2 b2 c2 d2 | a3 b3 c3 d3 | a4 b4 c4 d4 | a5 b5 c5 d5 | a6 b6  *  * | a7  *  *  * |
  e0 f0 g0  * | e1 f1 g1  * | e2 f2 g2  * | e3 f3  *  * | e4 f4  *  * |  *  *  *  * |  *  *  *  * |  *  *  *  * ]
sizes = [8, 7, 6, 6, 5, 5, 3]

p(i,m) = floor(m/BW)*BW*N + m-floor(m/BW)*BW + i*BW

block = floor(m/BW)
lane = m-block*BW

lookup_int(i,m): block*BW*N + lane + i*BW

## On parent indexes

Parent index vectors are packed in the same format as other vectors, however the index values must also be modified because parent indexes are relative.

p_flt(i,m) = indx[m] + p_lcl(i, m)
p_int(i,m) = lookup_int(0, m) + BW*p_lcl(i, m)

For example, the following two cells

```
cell 1, 6 nodes:

0--1--2--3
   \
    4--5

cell 2, 8 nodes:

0--1--2--3
   \
    4--5--6
     \
      7
```

p_lcl = [0 0 1 2 1 4 | 0 0 1 2 1 4 5 4]

p_flt = [0 0 1 2 1 4 | 6 6 7 8 7 10 11 10]

p_int = [0 1 * *| 0 1 * * | 4 5 * * | 8 9 * * | 4 5 * * | 16 17 * * | 20 * * * | 16 * * * ]


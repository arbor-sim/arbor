.. _matrix_solver:

Matrix Solvers
==============

Cable Equation
--------------

At the heart of the time evolution step Arbor we find a linear system that must
be solved once per time step. This system arises from the cable equation

.. math::
   C \partial_t V = \frac{\sigma}{2\pi r}\partial_x(r^2\partial_x V) + I

   I: \mbox{External currents}

   r: \mbox{cable radius}

   \sigma: \mbox{conductivity}

after discretisation into CV's, application of the FVM, and choosing an implicit
Euler time step as

.. math::
   \left(\frac{\sigma_i C_i}{\Delta\,t} + \sum_{\delta(i, j)} a_{ij}\right)V_i^{k+1} - \sum_{\delta(i, j)} a_ij V_i^{k+1} = \frac{\sigma_i C_i}{\Delta\,t}V_i^k + \sigma_i I_i

where :math:`\delta(i, j)` indicates whether two CVs are adjacent. It is written
in form of a sparse matrix, symmetric by construction.

Note that the cable equation is non-linear as the currents :math:`I` potentially
depend on :math:`V`. We obtain the current value of :math:`I` by splitting the
full operator into the 'matrix' (described here) and 'mechanism' parts (giving
the currents) and assuming one part being a constant during the update of the
respective other. However, for the matrix solver, we model _linear_ dependencies
:math:`I = gV + J` and collect all higher orders in the pretended constant
:math:`J` to improve accuracy and stability of the solver. This requires the
computation of the symbolic derivative :math:`g = \partial_V I` during
compilation of the mechanisms and an update to :math:`g(t)` when updating
currents using that symbolic expression.

Each *branch* in the morphology leads to a tri-diagonal block in the matrix
describing the system, since *branches* to not contain interior branching
points. Thus, an interior CV couples to only its neighbours (and itself).
However, at branch points, we need to factor in the branch's parents, which
couple blocks via entries outside the tri-diagonal structure. To ensure
un-problematic data dependencies for use of a substitution algorithm, ie each
row depends only on those of larger indices, we enumerate CVs in breadth-first
ordering. This particular form of matrix is called a *Hines matrix*.

Extension to Axial Diffusion
----------------------------

We reuse the matrix solver to model ionic diffusion along the dendrite ('axial diffusion')

.. math::
   \partial_t X_s = \partial_x(K_s\partial_x X_s) + I_s

   X_s: \mbox{External currents for ion species } s

   K_s: \mbox{diffusivity}

This requires computation of the per-species conductivity :math:`g_s` -- similar
to the total conductivity :math:`g` -- to model :math:`I_s(V) = g_sV + J_s`.
Apart from :math:`C=const.=1`, the two equations are identical, so the same
solver is used.

CPU
---

.. note:: See ``arbor/backends/multicore/matrix_state.hpp``:

          * Assembly is found in ``assemble`` and partially ``matrix_state``.
          * The solver lives in ``solve``.

The matrix solver proceeds in two phases: assembly and the actual solving. Since
we are working on cell groups, not individual cells, this is executed for each
cell's matrix.

Assembly
^^^^^^^^

We store the matrix in compressed form, as its upper and main diagonals.
Assembly computes the changing parts of the matrix and the right-hand side of
the matrix equation. Static parts are computed once, mainly the diagonal.

Solving
^^^^^^^

The CPU implementation is a straight-forward implemenation of a modified
Thomas-algorithm, using an extra input for the parent relationship. If each
parent is simply the previous CV, we recover the Thomas algorithm.

.. code:: c++

  void hines(const arb_value_type* diagonal, // main diagonal
             const arb_value_type* upper,    // upper diagonal
                   arb_value_type* rhs,      // rhs / solution
             const arb_index_type* parents,  // CV's parent
             int N) {
    // backward substitution
    for (int i = N-1; i>0; --i) {
      const auto parent  = parents[i];
      const auto factor  = upper[parent] / diagonal[i];
      diagonal[parent]  -= factor * upper[parent];
      rhs[parent]       -= factor * rhs[i];
    }
    // solve root
    b[0] = b[0] / d[0];

    // forward substitution
    for(int i=1; i<N; ++i) {
      const auto parent = parents[i];
      rhs[i] -= upper[i] * rhs[parent];
      rhs[i] /= diagonal[i];
    }
  }

GPU
---

.. note:: See ``arbor/backends/gpu/matrix_fine.hpp``:

          * Assembly is found in ``assemble`` and partially ``matrix_state``.
          * The solver lives in ``solve``.

          There is a simple solver in ``arbor/backends/gpu/matrix_flat.hpp``,
          which is only used to test/verify the optimised solver described
          below.

The GPU implementation of the matrix solver is more complex to improve
performance and make reasonable use of the hardware's capabilities.
In particular it trades a more complex assembly (and structure) for improved
performance.

Looking back at the structure of the Hines matrix, we find that we can solve
blocks in parallel, as long as their parents have been processed. Therefore,
starting at the root, we parallelise over the children of each branching point
and synchronise execution at each such branching point. Each such step is called
a *level*. Execution time is further optimised by packing blocks into threads by
size and splitting overly large blocks to minimise divergence.

A detailled description can be found `here
<https://arxiv.org/ftp/arxiv/papers/1810/1810.12742.pdf>`_ and the references
therein are worthwhile further reading.

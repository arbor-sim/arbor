.. _matrix_solver:

Matrix Solvers
==============

.. _cable_equation:

Cable Equation
--------------

At the heart of the time evolution step in Arbor we find a linear system that must
be solved once per time step. This system arises from the cable equation

.. math::
   C \partial_t V = \frac{\sigma}{2\pi a}\partial_x(a^2\partial_x V) + I

   I: \mbox{External currents}

   V: \mbox{Membrane potential}

   a: \mbox{cable radius}

   \sigma: \mbox{conductivity}

after discretisation into CVs, application of the FVM, and choosing an implicit
Euler time step as

.. math::
   \left(\frac{\sigma_i C_i}{\Delta\,t} + \sum_{\delta(i, j)} a_{ij}\right)V_i^{k+1} - \sum_{\delta(i, j)} a_ij V_i^{k+1}
     = \frac{\sigma_i C_i}{\Delta\,t}V_i^k + \sigma_i I_i

where :math:`\delta(i, j)` indicates whether two CVs are adjacent. It is written
in form of a sparse matrix, symmetric by construction.

The currents :math:`I` originate from the ion channels on the CVs in question,
see the discussion on mechanisms for further details. As :math:`I` potentially
depends on :math:`V`, the cable equation is non-linear. We model these
dependencies up to first order as :math:`I = gV + J` and collect all higher
orders into :math:`J`. This is done to improve accuracy and stability of the
solver. Finding :math:`I` requires the computation of the symbolic derivative
:math:`g = \partial_V I` during compilation of the mechanisms. At runtime
:math:`g` is updated alongside with the currents :math:`I` using that symbolic
expression.

Each *branch* in the morphology leads to a tri-diagonal block in the matrix
describing the system, since *branches* do not contain interior branching
points. Thus, an interior CV couples to only its neighbours (and itself).
However, at branch points, we need to factor in the branch's parents, which
couple blocks via entries outside the tri-diagonal structure. To ensure
un-problematic data dependencies for use of a substitution algorithm, ie each
row depends only on those of larger indices, we enumerate CVs in breadth-first
ordering. This particular form of matrix is called a *Hines matrix*.

CPU
---

.. note:: See ``arbor/backends/multicore/matrix_state.hpp``:

          * ``struct matrix_state``
            * the ``matrix_state`` constructor sets up the static parts
            * the dynamic part is found in ``assemble``
            * the solver lives in ``solve``.

The matrix solver proceeds in two phases: assembly and the actual solving. Since
we are working on cell groups, not individual cells, this is executed for each
cell's matrix.

Assembly
^^^^^^^^

We store the matrix in compressed form, as its upper and main diagonals. The
static parts -- foremost the main diagonal -- are computed once at construction
time and stored. The dynamic parts of the matrix and the right-hand side of the
equation are initialised by calling ``assemble``.

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

          * ``struct matrix_state``
            * the ``matrix_state`` constructor sets up the static parts
            * the dynamic part is found in ``assemble``
            * the solver lives in ``solve``.

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

.. _sde:

Stochastic Differential Equations
=================================

We want to solve a system of SDEs,

.. math::

    \textbf{X}^\prime(t) = \textbf{f}(t, \textbf{X}(t)) + \sum_{i=0}^{M-1} \textbf{l}_i(t,\textbf{X}(t)) W_i(t),

which describes the dynamics of an arbitrary mechanism's state :math:`\textbf{X}`, see also
:ref:`this short SDE overview <mechanisms-sde>` and the corresponding :ref:`Arbor-specific NMODL
extension <format-sde>`.

Analytical Solutions
--------------------

For systems of linear SDEs with 

.. math::

    \begin{align}
    \textbf{f}\left(t, \textbf{X}(t)\right) &= \textbf{A}(t) \textbf{X}(t) + \textbf{a}(t) \\
    \textbf{l}_i\left(t, \textbf{X}(t)\right) &= \textbf{B}_i(t) \textbf{X}(t) + \textbf{b}_i(t)
    \end{align}

 
there exist ordinary differential equations for moments of the process :math:`\textbf{X}`.  The
expectation :math:`\textbf{m}(t) = E\left[\textbf{X}(t)\right]` and the second moment matrix
:math:`\textbf{P}(t) = E\left[\textbf{X}(t) \textbf{X}^T(t)\right]` can be computed with

.. math::

    \begin{align}
    \textbf{m}^\prime(t) = &\textbf{A}(t) \textbf{m}(t) + \textbf{a}(t)\\
    \textbf{m}(0) = &\textbf{x}_0 \\
    \textbf{P}^\prime(t) = &
          \textbf{A}(t)\textbf{P}(t)   + \textbf{P}(t)\textbf{A}^T(t)
        + \textbf{a}(t)\textbf{m}^T(t) + \textbf{m}(t)\textbf{a}^T(t) \\
       &+ \sum_{i=0}^{M-1} \left[
          \textbf{B}_i(t)\textbf{P}(t)\textbf{B}^T_i(t)
        + \textbf{B}_i(t)\textbf{m}(t)\textbf{b}^T_i(t)
        + \textbf{b}_i(t)\textbf{m}^T(t)\textbf{B}^T_i(t)
        + \textbf{b}_i(t)\textbf{b}^T(t) \right] \\
    \textbf{P}(0) = &\textbf{x}_0 \textbf{x}^T_0 
    \end{align}

Thus, we could in principle use our existing solvers for the above moment equations. Note, that the
equations for the second order moment are not linear, in general. Once the moments are computed, we
could then sample from the resulting mean and covariance matrix, using the fact that the solution is
normally distributed, with :math:`\textbf{X}(t) \sim N\left(\textbf{m}(t), \textbf{P}(t) -
\textbf{m}(t)\textbf{m}^T(t)\right)`. This approach has a few drawbacks, however:

* tracking of the moment equations increases the size of the system
* there exist no general solutions for non-linear systems of SDEs

Therefore, we choose to solve the SDEs numerically. While there exist a number of different methods
for scalar SDEs, not all of them can be easily extended to systems of SDEs. One of the simplest
approaches, the Euler-Maruyama method, offers a few advantages:

* works for (non-linear) systems of SDEs
* has simple implementation
* exhibits good performance

On the other hand, this solver only guarantees first order accuracy, which usually requires the time
steps to be small.


Euler-Maruyama Solver
---------------------

The Euler-Maruyama method is a first order stochastic Runge-Kutta method, where the forward Euler
method is its deterministic counterpart. The method can be derived by approximating the solution of
the SDE with an It\^o-Taylor series and truncating at first order.  Higher-order stochastic
Runge-Kutta methods are not as easy to use as their deterministic counterparts, especially for the
vector case (system of SDEs).
            
**Algorithm:** For each time step :math:`t_k = k ~\Delta t`

* draw random variables :math:`\Delta \textbf{W}  \sim N(\textbf{0}, \textbf{Q}\Delta t)`
* compute :math:`\hat{\textbf{X}}(t_{k+1}) = \hat{\textbf{X}}(t_k) + f(t_k, \hat{\textbf{X}}(t_k)) \Delta t + \sum_{i=0}^{M-1} \textbf{l}_i(t_k,\hat{\textbf{X}}(t_k)) \Delta W_{i}`

where :math:`\textbf{Q}` is the correlation matrix of the white noises :math:`W_i`.


Stochastic Mechanisms
---------------------

This Euler-Maruyama solver is implemented in modcc and requires a normally distributed noise source.
Since we assume that this noise is uncorrelated, we need independent random variables for every
location where a mechanism is placed at or painted on. We can achieve this by carefully seeding the
random number generator with a unique fingerprint of each such location, consisting of

* global seed value (per simulation)
* global cell id
* per-cell connection end point index (point mech.) or per-cell control volume index (density mech)
* mechanism id
* per-mechanism variable identifier
* per-mechanism simulation time (in the form of an integral counter)

By using these seed value, we can guarantee that the same random numbers are generated regardless of
concurrency, domain decomposition, and computational backend. Note, that because of our assumption
of independence, the correlation between the white noises is zero, :math:`\textbf{Q} = \textbf{1}`.

Random Number Generation
------------------------

Counter based random number generators (CBPRNGs) as implemented in Random123 offer a simple solution
for both CPU and GPU kernels to generate independent random numbers based on the above seed values.
We use the *Threefry-4x64-12* algorithm because

* it offers 8 64-bit fields for placing the seed values
* it is reasonably fast on both CPU and GPU

Due to the structure of the *Threefry* and other CBPRNGs, we get multiple indpendent uniformly
distributed values per invocation. In particular, *Threefry-4x64-12* returns 4 such values. Throwing
away 3 out of the 4 values would result in a quite significant performance penalty, so we use all 4
values, instead. This is achieved by circling through a cache of 4 values per location, and updating
the cache every 4th time step.

Normal Distribution
-------------------

The generated random numbers :math:`Z_i` must then be transformed into standard normally distributed
values :math:`X_i`.  There exist a number of different algorithms, however, we use the Box-Muller
transform because it requires exactly 2 independent uniformly distributed values to generate 2
independent normally distributed values. Other methods, such as the Ziggurat algorithm, use
rejection sampling which may unevenly exhaust our cache and make parallelization more difficult.

For the Euler-Maruyama solver we need normal random numbers with variance :math:`\sigma^2 = \Delta t`.
Thus, we scale the generated random number accordingly, :math:`\Delta W_{i} = \sqrt{\Delta t} X_i`.

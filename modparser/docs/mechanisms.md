# Mechanism Implementation
There are four functions exposed for each mechanism in the c files generated from NMODL
1. init
2. current
3. jacob
4. state

The mechanism **KdShu2007** from corebluron is used for illustration when describing each part of the interface below.

## init
Initialization of a mechanism, for example setting initial values for state variables, is specified in an INITIAL block of the NMODL language:
```
INITIAL {
  trates(v)
  m=minf
  h=hinf
}
```
This is translated into the `nrn_init` function, which first updates the local value of voltage, before calling `initmodel()`, which implements the operations specified in INITIAL.

## current
The current is implemented as `nrn_cur()`, and is is called for each mechanism from `nrn_rhs()`.  It computes the current `i`, which is a function of `v`. It also computes the derivative of voltage w.r.t time, i.e. the conductance `g`. Each current is saved, and contributions to ionic currents accumulated. The total current is added to the `VEC_RHS` vector.

For simulation purposes, the current doesn't have to be saved, however it is currently saved as state (possibly in case the user wants to query the calculated current for analysis or plotting). Given that requests for this data is very infrequent, and probably never when running in a HPC context, these values shouldn't be saved. This would have dual benefits of greatly reducing memory footprint, and increasing computational intensity of the kernels.

### location in NMODL file
The current contribution is defined in the `BREAKPOINT` block. For example, in the KdShu2007 model, it is a single statement (because only one current, that pertaiing to pottasium, is updated)
```
ik = gkbar * m*h*(v-ek)
```
which is translated into the `nrn_current()` helper function in the C code
```C
static double _nrn_current( blah blah blah) {
  double _current = 0.;
  v = _v;
  ik = gkbar * m * h * (v - ek); // this line matters!
  _current += ik;
  return _current;
}
```
At first glance, the generated c code appears to be overly-elaborate, with just one line for the computation of the current `ik` (with considerable overhead because this is called twice to calculate the derivative of the current w.r.t. voltage). However, for mechanisms that contribute to multiple currents (say both calcium and pottasium, or a private non-specific current), this routine will calculate the individual contribution to each current (`ik` in this case), and an accumulated current, i.e. `_current`.

There are some optimizations that I can see
1. when there is only one current being contributed too, there is no need to use the accumulator `current_`.
2. the `nrn_cur` function calculates the derivative of each current contribution w.r.t. voltage, however for the purpose of simulation only the derivative of the total current `g` is required when updating the diagonal of the Jacobi matrix in `nrn_jacobi`. The other currents could be made as optional outputs that can be calculated _on demand_.
3. in this case, the current is a linear function of `v`, because the values of all other parameters (`gkbar`, `m`, `h`, `ek`) are fixed when calculating the current `ik`. So it should be possible to remove the second call to current, by noting that in this case `g=gkbar*m*h`.
4. as noted below, the contribution to the jacobian is the same for all mechanisms: the value of `g` is added to the diagonal entry of the matrix. Could this be performed directly here? There would be issues with race conditions, but it is worth considering.

## jacob
The jacobi is implemented as `nrn_jacob()`. This routine is called in `nrn_lhs`, when updating the diagonal entries in the linear system. Recall that the off-diagonal entries are constant in time, and computed just once during initialization.

The `jacobi` function is not derived from the NMODL definition, instead it is the same for all mechanisms that contribute to the LHS. In pseudo code, it performs the following:
```
for i = 1:nodecount
  VEC_D[ni[i]] += g[i]
end
```

For this to be performed efficiently, the conductance, `g`, has to be computed and stored when calculating the current in `nrn_rhs()`. The function `setup_tree_matrix()` first updates the current in `nrn_rhs()`, which is where `g` is calcuted (the conductance `g` is the derivative of current w.r.t voltage, i.e. `di/dv`).
```
function setup_tree_matrix ( cell_group )
  nrn_rhs ( cell_group )
  nrn_lhs ( cell_group )
end
```

With this in mind, the value of `g` might best be computed and stored in the call to `nrn_cur` in `nrn_rhs`, for subsequent use in `nrn_jacob`.

### location in NMODL file
As noted above, the `jacobi` function is not derived from the NMODL file.

## state
**The calcuation of state is the most compilicated in terms of translation.**

The routine `nonvint()` in the C driver loops over each mechanism, calling its `nrn_state()` function. In pseudo code:
```
function nonvint(cell_group)
  for mechanism in cell_group.mechanisms
    nrn_state(mechanism)
  end
end
```
The `nrn_state()` function loops over individual mechanism instances. It does something very odd with the time, which as far as I can tell is redundant

```
for i in 1:nodecount
  local break = t + dt/2.
  local save  = t
  while(t < break)
    v[i] = data.v[ni[i]]
    states()
    t += dt
  end
  t = save
end
```
No wonder this code is so slow. Removing this redundancy simplifies our pseudo-code:
```
for i in 1:nodecount
  v[i] = data.v[ni[i]]
  states()
end
```
The `states()` function is derived directly from the DERIVATIVE block. In the KdShu2007 case, the DERIVATIVE block is
```
DERIVATIVE states {
  trates(v)
  m' = (minf-m)/mtau
  h' = (hinf-h)/htau
}
```
Of interest is the expressions for the derivative of each two state variables, `m` and `h`. How these are handled depends on the SOLVE statement in the BREAKPOINT block
```
BREAKPOINT {
  SOLVE states METHOD cnexp
  ik = gkbar * m*h*(v-ek)
}
```
The generated C code looks something like
```
static int states(blah blah)
  trates(_threadargscomma_ v);
  m = m +
      (1. - exp(dt * ((((-1.0))) / mtau))) *
          (-(((minf)) / mtau) / ((((-1.0))) / mtau) - m);
  h = h +
      (1. - exp(dt * ((((-1.0))) / htau))) *
          (-(((hinf)) / htau) / ((((-1.0))) / htau) - h);
  return 0;
}
```
The call to the `trates()` function, a user-defined function that calculates parameters like `minf` that are required for statements that update the state variables `m` and `h`, is directly translated from the BREAKPOINT block.

The update statements for the state variables, e.g.
```
  m = m + (1. - exp(dt * ((((-1.0))) / mtau))) * (-(((minf)) / mtau) / ((((-1.0))) / mtau) - m);
  : which can be reduced simply as follows
  m = minf - (minf -m)*exp(-dt/mtau);
```
integrate the state variable by a step of dt. The generated code is quite complicated, given that all of the ODEs that I can find in the CoreNeuron NMODL files are simple linear ODEs of this form.

The compiler could extract the SOLVE statement from the AST for the BREAKPOINT, removing it in the process. Then the BREAPKPOINT block could be treanslated directly into `nrn_current()`.

#### Oportunities for optimization:
1. The trates() function stores all of the parameters, i.e. `hinf`, `htau` into arrays in global memory (because they are declared as RANGE variables). However, these values are only used to update the state variable, and there is no reason why they should be stored. A better approach would be to store them as stack variables, and provide a mechanism for the user to compute them on the fly if they are required for analysis/visualization.


#### Exponential growth
Very often the derivative of state variables is of the linear form
```
s' = a*s + b
```
which can be solved exactly for a time step (assuming that k and kinf are constant during the timestep. The update function for this is
```
s = -b/a + (s + b/a)exp(a*dt)
```

for equations of the form `s'=(sinf - s)/stau`
```
s = sinf + (s - sinf) * exp(-dt/stau)
```
The linear form of the equation could be obtained by analysing the AST, and this special case used when the appropriate form is detected. I think that the current compiler might attempt something similar, but doesn't simplify very effectively.

#### The problem with BREAKPOINT
The BREAKPOINT block is awkward, because it contains statements that are translated into code in different functions in the C code:
1. the SOLVE statement which is used to generate the `states()` function in `nrn_state()`.
2. the current update function, which is used to generate code in `nrn_cur`.
This is a hangover from the original MODL language. It serves neither the needs of scientists or compiler writers, so it should really be rewritten.

A simple suggestion that doesn't depart too far from the original specs might be
```
SOLVE {
  states METHOD cnexp
}
CURRENT {
  ik = gkbar * m*h*(v-ek)
}

```
Which neatly separates the two tasks, making a simple one-to-one mapping between model and expression for the neuroscientist, and similary between expression and implementation for the computer scientist.

## Functions and procedures
In addition to the externally exported symbols (`nrn_*`), there are user-defined FUNCTION and PROCEDUREs. These are translated directly from the description in the NMODL file.

### Purity please
It would be a good idea to place some restrictions on functions, or giving users the ability to annotate functions with properties, that make it easier to optimize the generated code. For example using pure functions. If users are encouraged to specify intermediate values as pure functions, instead of range variables, memory footprint will be reduced, and a mechanism can be provided for users to compute these values on the fly.

By _pure_, I mean insisting that functions can't have side-effects (they can't rely on or alter global state). This is a familiar concept from functional programming, constexpr functions in C++11, or elemental functions in Fortran. These conditions can be relaxed a bit, to _pure is as pure does_, as in the D programming language and C++14. This allows a pure function to have mutable local state, and use immutable global state.

If parameters are specified as pure functions, they can always be computed when needed on the fly (reducing memory footprint and increasing computational intensity: win-win). These functions can then be exposed to the user, so that the parameters they compute can be generated by the user when and if they need them.

This is a move that would benefit everyone: users get to write relationships in a more modular and clearer manner, and HPC folks can optimize, safe with the guarentees that purity gives us.

# Mechanisms
Here is a list of mechanisms that will be used for testing. The mechanisms are taken directly from the `corebluron/mech/modfile/` path in the HPCNeuron repository:
```
bbpcode.epfl.ch/sim/corebluron
```

##expsyn.mod
Chosen for it's extreme simplicity
* requires no modification to be in "ideal form" for transformation into optimal code
* no dependencies on external currents (calcium, potassium, etc.)
* has a single non-specific current
* has a `NET_RECEIVE` block that is simplicity personified

**requires**
* add `NONSPECIFIC_CURRENT` support
* support for specifying ranges for variables using the range syntax `<min,max>`

##hh.mod
Chosen because
* relatively simple
* writes to two different external currents (calcium and pottasium) _and_ an internal non-specific current
* has a function `vtrap` that shows the ugly workarounds used in NMODL files to work around numeric issues in the models/numerical methods

```
: traps for 0 in denominator of rate eqns.
FUNCTION vtrap(x,y) {
  if (fabs(x/y) < 1e-6) {
    vtrap = y*(1 - x/y/2)
  }else{
    vtrap = x/(exp(x/y) - 1)
  }
}

```
**requires**
* support for `?` to indicate comments, so that the following are equivalent:

```
? a comment
: a comment
```
* support for specifying ranges for variables using the range syntax `<min,max>`

##ProbAMPANDMDA_EMS
* very important, because in the order of 50% of time to solution is spent in this kernel or ones like it.
* it is challenging to do right, but if it is done, we have solved nearly all the challenges.
* Is a synapse, with complicated `NETSTIM` block
* requires useful extensions to the NMODL language to produce portable performance
  * random number support
  * arrays

##stim.mod
This is a _very_ simple mechanism, with one interesting behaviour
* the `BREAKPOINT` block has an `at_time()` call, shown below. This might be an interesting optimization oportunity, where an if branch is chosen depending on the value of `t`, instead of branching according to a spatial variable.

```
BREAKPOINT {
	at_time(del)
	at_time(del+dur)

	if (t < del + dur && t >= del) {
		i = amp
	}else{
		i = 0
	}
}

```

##KdShu2007

### Issues
* There are two parameters, `gkbar` and `ek`, which are declared as `RANGE` and `PARAMETER`, but are not `GLOBAL`. **update** This is fine, because parameters are GLOBAL by default, and the driver modify them by default.

* The `trates()` function sets the values of the range variables `minf`, `hinf`, `mtau` and `htau`. For `minf` and `hinf` this makes sense, because these are functions of voltage, which varies spatially. However, the value of `mtau` and `htau` are always set to the same constant value, with no spatial variation. This wastes the memory required to store the two fields, and also stores. The values in `[hm]tau` might have varied in space during development of the mechanism, and the developer probably forgot to change them to scalar parameters. The `qt` parameter is also never used. **This is the sort of practice that would be caught if mechanisms had to go through proper code review from somebody who understands performance**.

```
PROCEDURE trates(v) {
  LOCAL qt
  qt=q10^((celsius-22)/10)
  minf=1-1/(1+exp((v-vhalfm)/km))
  hinf=1/(1+exp((v-vhalfh)/kh))
  mtau = 0.6
  htau = 1500
}

```
### fixes
* The values of the `[hm]inf` and `[hm]tau` do not need to be stored in arrays, instead they could be computed inline. This would remove four range variables, and the bandwidth required to store the values contained therein. The `trates` procedure is used in two places, once in INITIAL and the other in the DERIVATIVE block. Another approach would be to define functions with the same name as the variables:
```
FUNCTION minf(v_l) {
  minf=1-1/(1+exp((v_l-vhalfm)/km))
}
FUNCTION hinf(v_l) {
  hinf=1/(1+exp((v_l-vhalfh)/kh))
}
```

#Source-2-Source

## PROCEDURE
Ultimately, we want to inline procedure calls. By inlining, then performing optimization techniques like constant folding, we can hopefully get significant reductions to both floating point work and memory allocation. As a motivating example, take the following code snippet from KdShu2007
```
INITIAL {
  trates(v)
  m = minf
}
PROCEDURE trates(v) {
  minf=1-1/(1+exp((v-vhalfm)/km))
  mtau = 0.6
}
```

inlining this would produce:
```
INITIAL {
  minf=1-1/(1+exp((v-vhalfm)/km))
  mtau = 0.6
  m = minf
}
```
constant folding and variable reduction would give
```
INITIAL {
  LOCAL minf=1-1/(1+exp((v-vhalfm)/km))
  m = minf
}
```

In the model, `mtau` and `minf` are `RANGE` variables. But, with the inlined code, the `mtau` array is replaced with a stack variable. The final C++ code generated will look something like:
```
void initial() {
  for(int i=0; i<n_; ++i) {
    double minf=1-1/(1+exp((v[i]-vhalfm)/km));
    m[i] = minf;
  }
}
```
which is a damn site better than the _equivalent_ code that would be generated without inlining
```
void initial() {
  for(int i=0; i<n_; ++i) {
    minf[i]=1-1/(1+exp((v[i]-vhalfm)/km));
    mtau[i]=0.6;
    m[i] = minf[i];
  }
}
```
The nonlined version has two additional stores to main memory (which the C++ compiler can't optimize out). It also requires that memory for the two arrays be allocated. With inlining, it would be possible for `modparser` to detect that the memory for `minf` and `mtau` never has to be allocated, which reduces storage requirements.

However, as a first step, we perform no inlining. The approaches below produce vectorized code for x86 with gcc.
### with no arguments
The implementation of a procedure will be as follows.
```
PROCEDURE foo() {
  LOCAL x
  x = exp(b)
  a = 2*x
}
```
becomes
```
void foo(const int i) {
  double x;
  x = std::exp(b[i]);
  a[i] = double(2)*x;
}
```
* the index i is passed as a constant parameter
* the index is used to index the arrays `a` and `b`, which are members of the mechanism class
* local variables are declared as stack variable with type double
* we use `std::exp` to calculate the exponential

### with arguments
```
PROCEDURE foo(v) {
  htau = 1 - exp(v-60)
}
```
becomes
```
void foo(const double v, const int i) {
  htau[i] = double(1) - std::exp(v-double(60));
}
```
* The key here is that `v` is passed as an argument. This 'shadows' the `v` field, which is the vector of voltage values. This is a usage pattern that is quite common in the Neuron `.mod` files.

##STATES

The state function is computed from the DERIVATIVE block, take our good friend KdShu2007:

```
DERIVATIVE states {
  trates(v)
  m' = (minf-m)/mtau
  h' = (hinf-h)/htau
}
PROCEDURE trates(v) {
  minf=1-1/(1+exp((v-vhalfm)/km))
  hinf=1/(1+exp((v-vhalfh)/kh))
  mtau = 0.6
  htau = 1500
}
```
The `states` method is generated as an intermediate step by the `modparser` compiler:
```
PROCEDURE states() {
  trates(v)
  m = minf + (m-minf)*exp(-(1/mtau)*dt)
  h = hinf + (h-hinf)*exp(-(1/htau)*dt)
}
```
Where we take advantage of the fact that the ordinary differential equations (ODEs) for the state variables are all linear. The compiler actually analyses the ODE, to produce this optimal update above if the ODE is linear.


At the end of the day, we want to fully inline the states procedure:
```
PROCEDURE states() {
  LOCAL minf, hinf
  minf=1-1/(1+exp((v-vhalfm)/km))
  hinf=1/(1+exp((v-vhalfh)/kh))
  m = minf + (m-minf)*exp(-1.66666666666666666666667*dt)
  h = hinf + (h-hinf)*exp(-0.00066666666666666666667*dt)
}
```
Note that the compiler can perform constant folding on terms like `-(1/mtau)`. The constant folding is performed using 80-bit `long double` precision, which ensures that it will be more acurate than without constant folding.

This will generate the following C++ code
```C++
void states() {
  for(int i=0; i<n_; ++i) {
    double minf, hinf;
    minf = 1-1/(1+exp((v[i]-vhalfm)/km))
    hinf = 1/(1+exp((v[i]-vhalfh)/kh))
    m[i] = minf + (m[i]-minf)*exp(-1.66666666666666666666667*dt)
    h[i] = hinf + (h[i]-hinf)*exp(-0.00066666666666666666667*dt)
  }
}
```
This performs 3 loads and 2 (5 memory ops) stores for each loop iteration, compared to 3 loads and 6 stores (9 memory ops). This reduces the bandwidth requirements by a factor of almost two!

**update** the actual C++ code produced would probably look more like the following...
```C++
void states() {
  auto n = size();
  #pragma ivdep
  for(int i=0; i<n; ++i) {
    double l_minf, l_hinf;
    l_minf = double(1)-double(1)/(double(1)+exp((v[i]-info::vhalfm)/info::km))
    l_hinf = double(1)/(double(1)+exp((v[i]-info::vhalfh)/info::kh))
    state_m[i] = l_minf + (state_m[i]-l_minf)*exp(double(-1.66666666666666666666667)*info::dt)
    state_h[i] = l_hinf + (state_h[i]-l_hinf)*exp(double(-0.00066666666666666666667)*info::dt)
  }
}
```

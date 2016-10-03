#State of the art

Some reasons why Neuron is **not HPC**
* poor utilization of possible FLOPs (according to roofline)
* wrong data layout (SoA vs. AoS)
* excessive use of memory
* not performance portable (it isn't possible to run on GPU, and not possible to port in current form)

The team at BBP have done a good job of stripping away the functionality that isn't required to perform the simulation tasks in the HBP. **However, to get a code that is HPC ready, just modify the code and 'getting one more iteration' out of it won't work**.

###Mechanisms: _state_ and _kernels_
Each mechanism has state, which is a set of values defined at each point of a cell discretizatio at which the mechanism is present. Mechanisms also have kernels, which do the following
* modify mechanism state
* modify global state
  * update the system matrix for the cell
  * update current at affected sites

Mechanisms modify state by applying kerenels. Applying the mechanism kernels (`state`, `current`, etc.) dominate time to solution (98% for the PCP benchmarks that I did a while back).

###Mechanisms: performance
There are lots of opportunities to improve performance of the kernels, both in terms of time to solution and memory consumption:
* reducing DRAM pressure
  * kernels are bandwidth bound, and there are lots of redundant loads and stores, that could be removed.
  * use _Structure of Arrays_, to  ensure that cache lines only contain useful data
* improve vectorization
  * use _Structure of Arrays_.
* remove redundant flops
* increase work-per-loop
  * reduce memory footprint
* make it possible for the compiler to reason about possible optimizations

Most of these are low lying fruit.

###The problem
The C code that implements the kernels is very poor. It is very difficult for humans to read, and is very difficult for compilers to reason about. As an example, the following code
```
static void nrn_state(_NrnThread *_nt, _Memb_list *_ml, int _type) {
  double _break, _save;
  double *_p;
  Datum *_ppvar;
  ThreadDatum *_thread;
  double _v;
  int *_ni;
  int _iml, _cntml;
#if CACHEVEC
  _ni = _ml->_nodeindices;
#endif
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data + _iml * _psize;
    _ppvar = _ml->_pdata + _iml * _ppsize;
    _v = VEC_V(_ni[_iml]);
    _break = t + .5 * dt;
    _save = t;
    v = _v;
    {
      {
        {
          for (; t < _break; t += dt) {
            states(_p, _ppvar, _thread, _nt);
          }
        }
        t = _save;
      }
    }
  }
}

static int states(double *_p, Datum *_ppvar, ThreadDatum *_thread,
                  _NrnThread *_nt) {
  {
    trates(_threadargscomma_ v);
    m = m +
        (1. - exp(dt * ((((-1.0))) / mtau))) *
            (-(((minf)) / mtau) / ((((-1.0))) / mtau) - m);
    h = h +
        (1. - exp(dt * ((((-1.0))) / htau))) *
            (-(((hinf)) / htau) / ((((-1.0))) / htau) - h);
  }
  return 0;
}

static int trates(_p, _ppvar, _thread, _nt, _lv) double *_p;
Datum *_ppvar;
ThreadDatum *_thread;
_NrnThread *_nt;
double _lv;
{
  double _lqt;
  _lqt = pow(q10, ((celsius - 22.0) / 10.0));
  minf = 1.0 - 1.0 / (1.0 + exp((_lv - vhalfm) / km));
  hinf = 1.0 / (1.0 + exp((_lv - vhalfh) / kh));
  mtau = 0.6;
  htau = 1500.0;
  return 0;
}
```
is replaced by the following in my rewrite
```
void trates(const int idx, double v) {
    minf[idx] = (1-(1/(1+exp(((v-vhalfm)/km)))));
    hinf[idx] = (1/(1+exp(((v-vhalfh)/kh))));
    mtau[idx] = 0.6;
    htau[idx] = 1500;
}
void state() {
    double l_ba;
    double l_a;
    auto n = node_indices_.size();
    for(int idx=0; idx<n; ++idx) {
        trates(idx, v[idx]);
        l_a = (-1/mtau[idx]);
        l_ba = ((minf[idx]/mtau[idx])/l_a);
        m[idx] = -l_ba + (m[idx] + l_ba)*exp(l_a*dt);
        l_a = (-1/htau[idx]);
        l_ab = ((hinf[idx]/htau[idx])/l_a);
        h[idx] = -l_ba + (h[idx] + l_ba)*exp(l_a*dt);
    }
}
```

The original code is difficult to omptimize
* The data layout is not suitable
  * there is a lot of pointer chasing and indexing
  * AoS introduces further memory inefficiencies
* There a lot of nested function calls
  * given the complexity of the memory accesses, the compiler will struggle to inline.
* There are strange hang-overs (note the `for` loop for time)

The revised version of the code shown above is a 'naive' transformation of the original DSL. With inling and constant folding, the memory bandwidth requirements can be reduced by a factor of almost 2, and the number of floating point operations reduced too.

##My work
I have been working on a new compiler for the NMODL DSL, to replace the current compiler.

Compilation is broken into 3 steps
1. parse and build syntax tree for the .mod file. The aim is to generate an AST that is designed for subsequent steps, and provide users with a good experience:
  * robust
  * accurate error and warnings
  * improved correctness checks
2. transform this into optmized AST representations of the different methods exposed by a mechanism (current, state, init, jacobi).
3. generate hardware-specific implementation of kernels and API for the methods generated in step 2.

The design is such that to support a new hardware backend, steps 1 and 2 do not need to be touched. The user has to implement a Visitor class and wrapper that performs step 3.

####Work to do:
* define an API
  * this is a front-end for the mechanisms, hiding all back end details
* determine the optimal back end implementations for different hardware
  * x86/BGQ
  * CUDA
  * MIC
* fix NMODL. Currently NMODL lacks some features required to make performance portable kernels. Namely, there are many VERBATIM blocks, where the user can insert C code directly into the NMODL source.These indicate features that the language lacks that users require:
  * support for random number generation
  * support for writing custom mathematical operators (my favourite is 'natural log of gamma function')

#A Concern
I don't think that it is possible to acheive the 'HPC aims' without significantly rewriting CoreNeuron, and breaking backwards compatability with Neuron.
* The aim is to keep compatability, so that more features can be ported from Neuron to CoreNeuron (e.g. gap junctions.)
* What about rewriting CoreNeuron to have a more flexible and maintainable design, and reimplementing features from Neuron
* Forget about writing a high-quality CoreNeuron that can be ported back into Neuron.
* However, Neuron is still used for model building.

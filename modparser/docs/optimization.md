#Optimization Strategy

The first version of the compiler generates  kernels that are close to direct translations of the .mod files. There have been some transformations performed, but only so far as is required to generate a working kernel.

The aim is now to generate optimized code

## Rewrite .mod files

The easiest gains are in rewriting the .mod files to reduce the number of loads and stores and floating point operations.

## Optimize generated kernels
### shadow buffers
Implement shadow buffers to increase oportunities for vectorization.
```
for(int i_=0; i_<n; ++i_) {
  v[i_] = vec_v[i_];
  value_type conductance_ = 0.;
  value_type current_ = 0.;
  gIh[i_] = gIhbar[i_]*m[i_]; ihcn[i_] = gIh[i_]*(v[i_]-ehcn);
  current_ = current_+ihcn[i_];
  conductance_ = conductance_+gIh[i_];
  g_[i_] = current_;
  vec_rhs[i_] -= current_;
}
```

can be replaced:

```
auto nblocks = n/RLEN;
for(block=0; block<nblocks; ++block) {
  auto block_start = block*RLEN;
  for(int j_=0; j_<RLEN; ++j_) {
    v[j_] = vec_v[j_+block_start];
  }
  #pragma ivdep
  for(int j_=0; j_<RLEN; ++j_) {
    int i_ = block_start + j_;
    value_type conductance_ = 0.;
    value_type current_ = 0.;
    gIh[i_] = gIhbar[i_]*m[i_];
    ihcn[i_] = gIh[i_]*(v[i_]-ehcn);
    current_ = current_+ihcn[i_];
    conductance_ = conductance_+gIh[i_];
    g_[i_] = current_;
  }
  for(int j_=0; j_<RLEN; ++j_) {
    vec_rhs[j_+block_start] -= g_[j_+block_start];
  }
}
```

try doing this by hand, then report back here whether it works.

##Expression Simplification
Perform constant folding/propogation and zero removal:
```
-0  -> 0
0*x -> 0
0/x -> 0
0+x -> 0
```

Once this is working, we can use it to remove redundant variables/fields

##Compiler/architecture specific
Experiment with flags and directives for specific compilers (e.g. Intel compiler on Haswell).

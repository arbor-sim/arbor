# `lmporho` — generate random neuron morphologies

`lmorpho` implements an L-system geometry generator for the purpose of
generating a series of artificial neuron morphologies as described by a set of
stasticial parameters.

## Method

The basic algorithm used is that of [Burke (1992)][burke1992], with extensions
for embedding in 3-d taken from [Ascoli (2001)][ascoli2001].

A dendritic tree is represented by a piece-wise linear embedding of a rooted
tree into R³×R⁺, describing the position in space and non-negative radius.
Morphologies are represented as a sequence of these trees together with a point
and radius describing a (spherical) soma.

The generation process for a tree successively grows the terminal points by
some length ΔL each step, changing its orientation each step according to a
random distribution. After growing, there is a radius-dependent probability
that it will bifurcate or terminate. Parameters describing these various
probability distributions constitute the L-system model.

The distributions used in [Ascoli (2001)][ascoli2001] can be constant (i.e. not
random), uniform over some interval, truncated normal, or a mixture
distribution of these. In the present version of the code, mixture
distributions are not supported.

## Output

Generated morphologies can be output either in
[SWC format](http://research.mssm.edu/cnic/swc.html), or as 'parent vectors',
which describe the topology of the associated tree.

Entry _i_ of the (zero-indexed) parent vector gives the index of the proximal
unbranched section to which it connects. Somata have a parent index of -1.

Morphology information can be written one each to separate files or
concatenated. If concatenated, the parent vector indices will be shifted as
required to maintain consistency.

## Models

Two L-system parameter sets are provided as 'built-in'; if neither model is
selected, a simple, non-biological default model is used.

As mixture distributions are not yet supported in `lmorpho`, these models are
simplified approximations to those in the literature as described below.

### Motor neuron model

The 'motoneuron' model corresponds to the adult cat spinal alpha-motoneuron
model described in [Ascoli (2001)][ascoli2001], based in turn on the model of
[Burke (1992)][burke1992].

The model parameters in Ascoli are not complete; missing parameters were taken
from the corresponding ArborVitae model parameters in the same paper, and
somatic diameters were taken from [Cullheim (1987)][cullheim1987].

### Purkinke neuron model

The 'purkinke' model corresponds to the guinea pig Purkinje cell model from
[Ascoli (2001)][ascoli2001], which was fit to data from [Rapp
(1994)][rapp1994]. Somatic diameters are a simple fit to the three measurements
in the original source.

Some required parameters are not recorded in [Ascoli (2001)][ascoli2001];
the correlation component `diam_child_a` and the `length_step` ΔL are
taken from the corresponding L-Neuron parameter description in the
[L-Neuron database](l-neuron-database). These should be verified by
fitting against the digitized morphologies as found in the database.

Produced neurons from this model do not look especially realistic.

### Other models

There is not yet support for loading in arbitrary parameter sets, or for
translating or approximating the parameters that can be found in the
[L-Neuron database][l-neuron-database]. Support for parameter estimation
from a population of discretized morphologies would be useful.

## References

1. Ascoli, G. A., Krichmar, J. L., Scorcioni, R., Nasuto, S. J., Senft, S. L.
   and Krichmar, G. L. (2001). Computer generation and quantitative morphometric
   analysis of virtual neurons. _Anatomy and Embryology_ _204_(4), 283–301.
   [DOI: 10.1007/s004290100201][ascoli2001]

2. Burke, R. E., Marks, W. B. and Ulfhake, B. (1992). A parsimonious description
   of motoneuron dendritic morphology using computer simulation.
   _The Journal of Neuroscience_ _12_(6), 2403–2416. [online][burke1992]

3. Cullheim, S., Fleshman, J. W., Glenn, L. L., and Burke, R. E. (1987).
   Membrane area and dendritic structure in type-identified triceps surae alpha
   motoneurons. _The Journal of Comparitive Neurology_ _255_(1), 68–81.
   [DOI: 10.1002/cne.902550106][cullheim1987]

4. Rapp, M., Segev, I. and Yaom, Y. (1994). Physiology, morphology and detailed
   passive models of guinea-pig cerebellar Purkinje cells.
   _The Journal of Physiology_, _474_, 101–118.
   [DOI: 10.1113/jphysiol.1994.sp020006][rapp1994].


[ascoli2001]: http://dx.doi.org/10.1007/s004290100201
    "Ascoli et al. (2001). Computer generation […] of virtual neurons."

[burke1992]: http://www.jneurosci.org/content/12/6/2403
    "Burke et al. (1992). A parsimonious description of motoneuron dendritic morphology […]"

[cullheim1987]: http://dx.doi.org/10.1002/cne.902550106
    "Cullheim et al. (1987). Membrane area and dendritic structure in […] alpha motoneurons."

[rapp1994]: http://dx.doi.org/10.1113/jphysiol.1994.sp020006
    "Rapp et al. (1994). Physiology, morphology […] of guinea-pig cerebellar Purkinje cells."

[l-neuron-database]: http://krasnow1.gmu.edu/cn3/L-Neuron/database/index.html

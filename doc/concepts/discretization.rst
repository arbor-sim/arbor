.. _discretization:

.. _cablecell-discretization:

Discretization
==============

Before Arbor can actually simulate the behavior of a cell using the interplay of
morpholgy, mechanisms, etc it needs to turn the contiguous morphology into a
collection of discrete, conical sub-volumes. These are called 'compartments' or
'control volumes' (CV). The granularity of the subdivision controls the
precision of the simulation and its computational cost.

Arbor offers a set of composable discretization policies

.. label:: (cv-policy-explicit locset region)

   Use the subdivision as given by ``locset``

.. label:: (cv-policy-max-extent ext region flags)

   Subdivision into CVs of at most ``ext`` :math:`\mu m`. In the vicinity of
   fork points, smaller CVs might be used to avoid producing CVs containing
   forks, unless ``flags`` is ``(flag-interior-forks)``.

.. label:: (cv-policy-fixed-per-branch n region)

   Subdivide each branch into ``n`` equal CVs. In the vicinity of fork points,
   smaller CVs might be used to avoid producing CVs containing forks, unless
   ``flags`` is ``(flag-interior-forks)``.

.. label:: (cv-policy-every-segment region)

   Each segment --- as given during morphology construction --- will produce
   one CV.

.. label:: (cv-policy-default) = (cv-policy-fixed-per-branch 1)

   Each branch will produce one CV.

.. label:: (cv-policy-single region)

   The whole region will produce one CV.

In all cases ``region`` is optional and defaults to ``(all)``, i.e. the whole
cell. These policies compose through

.. label:: (join cvp1 cvp2)

   Use the union of the boundary points defined by ``cvp1`` and ``cvp2``.

.. label:: (replace cvp1 cvp2)

   Use the boundary points defined by ``cvp1`` everywhere except where ``cvp2``
   is defined. There, use those of ``cvp2``.

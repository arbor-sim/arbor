import neuroml
import neuroml.writers as writers
from neuroml import Cell, Morphology, Segment, Point3DWithDiam as P

# the C++ recipe uses the following simple ball and stick model

# morph.soma.r = 12.6157/2;
# double x = morph.soma.r;
# morph.add_section({{x, 0, 0, 0.5}, {x+200, 0, 0, 0.5}});
# x += 200;
# morph.add_section({{x, 0, 0, 0.5}, {x+100, 0, 0, 0.25}}, 1u);
# morph.add_section({{x, 0, 0, 0.5}, {x+100, 0, 0, 0.25}}, 1u);

soma_diam = 12.6157
soma_center = P(x=0,y=0,z=0,diameter=soma_diam)
soma = neuroml.Segment(proximal=soma_center, distal=soma_center)
soma.name = 'soma'
soma.id = 0

# our cell has the structure below, where p3 and p4 are the same point
# i.e. there are two branches from p2, but they lie on top of one another
#
#               p3
#              /
# s p1 ----- p2
#              \
#               p4

p1 = P(soma_center.x+soma_diam, y=0, z=0, diameter=0.5)
p2 = P(p1.x+200, y=0, z=0, diameter=0.5)
p3 = P(p1.x+200, y=0, z=0, diameter=0.25)

branch1 = Segment(id=1, proximal=p1, distal=p2, parent = neuroml.SegmentParent(segments=soma.id))
branch2 = Segment(id=2, proximal=p2, distal=p3, parent = neuroml.SegmentParent(segments=branch1.id))
branch3 = Segment(id=3, proximal=p2, distal=p3, parent = neuroml.SegmentParent(segments=branch1.id))

morpho = Morphology(id='test', segments=[soma, branch1, branch2, branch3])

cell = Cell(id='cell', morphology=morpho)

doc = neuroml.NeuroMLDocument(id = "TestNeuroMLDocument", cells=[cell])

writers.NeuroMLWriter.write(doc, 'test.nml')

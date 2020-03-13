import svgwrite
import math
import tree_inputs as trees

# Make the 7-sample sample tree image with annotations used to introduce sample
# tree concepts in the documentation.

def make_annotated(filename):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    sc=18
    line_width=0.2*sc
    fudge=1.5*sc

    colors = ['black', 'cornflowerblue', 'tomato', 'mediumseagreen']

    X = [ 0,  5,   10,   15, 18,   23,   20]
    Y = [ 0, -1,    0.5,  0,  5,    8,   -4]
    R = [ 3,  1.2,  1.2,  1,  1,  0.7,  0.8]
    P = [-1,  0,    1,    2,  3,    4,    3]
    C = [ 1,  0,    0,    2,  0,    3,    3]

    X = [sc*x for x in X]
    Y = [-sc*x for x in Y]
    R = [sc*x for x in R]
    nsamp = len(X)

    minx = min([X[i]-R[i] for i in range(nsamp)]) - fudge
    miny = min([Y[i]-R[i] for i in range(nsamp)]) - fudge
    maxx = max([X[i]+R[i] for i in range(nsamp)]) + fudge
    maxy = max([Y[i]+R[i] for i in range(nsamp)]) + fudge
    width = maxx-minx
    height = maxy-miny

    # Draw a line from every (non-root) sample to its parent
    lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_linecap='round', stroke_width=line_width))
    for i in range(1, nsamp):
        start = (X[P[i]], Y[P[i]])
        end   = (X[i],    Y[i])
        lines.add((dwg.line(start=start, end=end)))

    # Draw a circle for each sample
    circles = dwg.add(dwg.g(id='samples', stroke='black', fill='white', stroke_width=line_width))
    numbers = dwg.add(dwg.g(id='numbers', text_anchor='middle'))
    for i in range(nsamp):
        center = (X[i], Y[i])
        circles.add(dwg.circle(center=center, r=R[i], stroke=colors[C[i]]))
        tcenter = (X[i], Y[i]+0.3*sc)
        numbers.add(dwg.text(str(i), insert=tcenter))

    labels = ['root', 'fork', 'terminal', 'terminal']
    lX = [-1,  12.5, 19,  15.2]
    lY = [ 3.5, 1.5,  9,  -5]
    lC = [ 1,   2,    3,   3]
    nlab = len(labels)

    lX = [sc*x for x in lX]
    lY = [-sc*x for x in lY]

    for i in range(nlab):
        c = colors[lC[i]]
        dwg.add(dwg.text(labels[i], insert=(lX[i], lY[i]), stroke=c, fill=c))

    dwg.viewbox(minx, miny, width, height)

    dwg.save()


#
# Find the angle between two points
#

def angle(x, y):
    L = pow(x*x+y*y, 0.5)
    return math.acos(x/L)

def angle_norm(X, Y, i, j):
    x = X[j] - X[i]
    y = Y[j] - Y[i]
    theta = angle(x,y)
    if y<0: return 2*math.pi-theta
    return theta

#
# Find the points that are not collocated with i that have i as a spatial parent
#

def is_collocated(X, Y, i, j):
    return X[i]==X[j] and Y[i]==Y[j]

def coll_children(X, Y, children, i):
    result = [i]
    for c in children[i]:
        if is_collocated(X,Y,i,c):
            result += coll_children(X,Y,children,c)
    return result

def find_collocated(X, Y, P, children):
    nsamp = len(X)
    collocated = [[] for i in range(nsamp)]
    for i in range(nsamp):
        # If collcated with parent, we share the same collocated list.
        p = P[i]
        if i>0 and is_collocated(X,Y,i,p):
            collocated[i] = collocated[p]
        # Else build recursively
        else:
            collocated[i] = coll_children(X,Y,children,i)

    return collocated


def arm_children(X, Y, children, i):
    result = []
    for j in children[i]:
        if is_collocated(X,Y,i,j):
            result += arm_children(X, Y, children, j)
        else:
            result.append(j)
    return result

def find_arms(X, Y, P, children):
    arms = []
    for i in range(len(X)):
        result = []

        # find parent arm
        p = P[i]
        while p!=-1 and is_collocated(X,Y,i,p):
            p = P[p]
        if not p==-1:
            result.append(p)

        # find child arms
        result += arm_children(X, Y, children, i)

        arms.append(result)

    return arms

def unit_at_angle(theta, r=1):
    return [r*math.cos(theta), r*math.sin(theta)]

def sample_color(children, i):
    nkids=len(children[i])
    if i==0:
        return 'cornflowerblue'
    elif nkids==0:
        return 'mediumseagreen'
    elif nkids>1:
        return 'tomato'
    return 'gray'

def align_number(end, r):
    x = end[0]-r/4
    y = end[1]+r/3

    return (x,y)
#
# ############################################
#

def make_image(samples, filename, sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.15*sc

    # Padding around image.
    fudge=1.5*sc

    # Extract sample locations and radii
    X = samples['x']
    Y = samples['y']
    R = samples['r']
    P = samples['p']
    nsamp = len(X)

    # Scale locations and radii for drawing
    X = [ sc*x for x in X]
    Y = [-sc*x for x in Y]
    R = [ sc*x for x in R]

    children   = [[] for i in range(nsamp)]
    for i in range(1, nsamp):
        children[P[i]].append(i)

    # Find list of collocated samples for every sample.
    collocated = find_collocated(X,Y,P,children)

    # Find list of samples that branch off from each sample.
    arms = find_arms(X, Y, P, children);

    # Draw a line from every (non-root) sample to its parent
    lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_linecap='round', stroke_width=line_width))
    for i in range(1, nsamp):
        p = P[i]
        start = (X[p], Y[p])
        end   = (X[i], Y[i])
        lines.add(dwg.line(start=start, end=end))

    # Draw a circle for each sample
    circles = dwg.add(dwg.g(id='samples', stroke='black', fill='white', stroke_width=line_width))
    for i in range(nsamp):
        center = (X[i], Y[i])
        color = sample_color(children, i)
        radius = R[i]
        circles.add(dwg.circle(center=center, r=radius, stroke=color))

    # Now number the collocated samples.
    numbers = dwg.add(dwg.g(id='numbers', text_anchor='middle'))
    label_lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_linecap='round', stroke_width=line_width))
    for i in range(nsamp):
        ns = len(collocated[i])
        iscol = ns>1

        # Place a number next to each sample.
        # The location of the number should not be on top of a branch, and
        # if there are collocated points, we need to place all collacted labels
        # at the same time.
        # When points are collocated, we draw all labels of collocated points when
        # we find the first sample in a collocated set (hence skipping labeling if
        # a sample is collacted with its parent, which means it has already been drawn)
        if i==0 or not is_collocated(X,Y,i,P[i]):
            angles = [angle_norm(X,Y,i,j) for j in arms[i]]
            if len(angles)==0:
                angles = [math.pi/2]
            angles.sort()
            angles.append(angles[0]+2*math.pi)

            diff = [angles[j]-angles[j-1] for j in range(1,len(angles))]

            label_ids = collocated[i]
            max_rad = max([R[j] for j in label_ids])
            label_rad = sc/2
            out_rad = label_rad+max_rad

            center= (X[i], Y[i])
            for j in range(len(label_ids)):
                lab = label_ids[j]
                rad = R[lab]
                # Color of label
                color = sample_color(children,lab)
                # Direction relative to sample at which label will be drawn
                u = unit_at_angle(angles[j]+diff[j]/2)
                start = (X[i]+rad*u[0], Y[i]+rad*u[1])
                if rad<max_rad:
                    in_rad = R[lab]
                    end   = (X[i]+(max_rad+label_rad)*u[0], Y[i]+(max_rad+label_rad)*u[1])
                    label_lines.add(dwg.line(start=start, end=end, stroke=color, stroke_width=line_width/2))
                else:
                    end = start

                end = (end[0]+label_rad*u[0], end[1]+label_rad*u[1])
                text_pos = align_number(end, sc)
                numbers.add(dwg.text(str(lab), insert=text_pos, stroke=color, fill=color, text_anchor='start'))

    # Find extent of image.
    minx = min([X[i]-R[i] for i in range(nsamp)]) - fudge
    miny = min([Y[i]-R[i] for i in range(nsamp)]) - fudge
    maxx = max([X[i]+R[i] for i in range(nsamp)]) + fudge
    maxy = max([Y[i]+R[i] for i in range(nsamp)]) + fudge
    width = maxx-minx
    height = maxy-miny
    dwg.viewbox(minx, miny, width, height)

    # Write the image to file.
    dwg.save()

def make_table(tree, name):
    X = tree['x']
    Y = tree['y']
    R = tree['r']
    P = tree['p']
    T = tree['t']

    n = len(X)

    print(name)
    for i in range(n):
        p = str(P[i]) if i>0 else 'npos'
        s = '{ID:4d}, {p:>8s}, {x:4.1f}, {y:4.1f}, {z:4.1f}, {r:4.1f}, {tag:4d}'.format(ID=i, p=p, x=X[i], y=Y[i], z=0, r=R[i], tag=T[i])
        print(s)
    print()

def generate(path=''):

    make_annotated(path+'/stree.svg')
    make_image(trees.tree1, path+'/tree1.svg')
    make_image(trees.tree2a, path+'/tree2a.svg')
    make_image(trees.tree2b, path+'/tree2b.svg')
    make_image(trees.tree2c, path+'/tree2c.svg')
    make_image(trees.tree3, path+'/tree3.svg')
    make_image(trees.tree4a, path+'/tree4a.svg')
    make_image(trees.tree4b, path+'/tree4b.svg')
    make_image(trees.tree5, path+'/tree5.svg')
    make_image(trees.tree6a, path+'/tree6a.svg')
    make_image(trees.tree6b, path+'/tree6b.svg')

    make_table(trees.tree1,  'tree1')
    make_table(trees.tree2a, 'tree2a')
    make_table(trees.tree2b, 'tree2b')
    make_table(trees.tree2c, 'tree2c')
    make_table(trees.tree3,  'tree3')
    make_table(trees.tree4a, 'tree4a')
    make_table(trees.tree4b, 'tree4b')
    make_table(trees.tree5,  'tree5')
    make_table(trees.tree6a, 'tree6a')
    make_table(trees.tree6b, 'tree6b')

if __name__ == '__main__':
    generate('.')

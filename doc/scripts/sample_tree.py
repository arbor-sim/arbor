import svgwrite
import math

# Make the 7-sample sample tree image with annotations used to introduce sample
# tree concepts in the documentation.

def make_annotated(filename):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    sc=18
    line_width=0.2*sc
    fudge=1*sc

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

def angle(X, Y, i, j):
    x = X[j] - X[i]
    y = Y[j] - Y[i]
    L = pow(x*x+y*y, 0.5)

    theta = math.acos(x/L)
    if y<0: return 2*math.pi-theta
    return theta

#
# Find the points that are not collocated with i that have i as a spatial parent
#

def coll_children(X, Y, children, i):
    result = [i]
    for c in children[i]:
        if X[c]==X[i] and Y[c]==Y[i]:
            result += coll_children(X,Y,children,c)
    return result

def coll_parents(X, Y, P, i):
    p = P[i]
    if p!=-1 and X[p]==X[i] and Y[p]==Y[i]:
        return [p] + coll_parents(X,Y,P,p)
    return []

def sub_arms(X, Y, children, collocated, i):
    if not collocated[i]:
        return [i]
    result = []
    for j in children[i]:
        result += sub_arms(X,Y,children,collocated,j)
    return result

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

#
# ############################################
#

def make_image(samples, filename, sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    line_width=0.2*sc
    fudge=0.5*sc

    X = samples['x']
    Y = samples['y']
    R = samples['r']
    P = samples['p']

    X = [sc*x for x in X]
    Y = [-sc*x for x in Y]
    R = [sc*x for x in R]
    nsamp = len(X)

    collocated = nsamp * [False]
    children   = [[] for i in range(nsamp)]
    for i in range(1, nsamp):
        p = P[i]
        if X[i]==X[p] and Y[i]==Y[p]:
            collocated[p] = True;
            collocated[i] = True;
        children[p].append(i)

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
        lines.add(dwg.line(start=start, end=end))

    # Draw a circle for each sample
    circles = dwg.add(dwg.g(id='samples', stroke='black', fill='white', stroke_width=line_width))
    numbers = dwg.add(dwg.g(id='numbers', text_anchor='middle'))
    for i in range(nsamp):
        center = (X[i], Y[i])
        nkids=len(children[i])
        stroke = sample_color(children, i)
        circles.add(dwg.circle(center=center, r=R[i], stroke=stroke))
        if not collocated[i]:
            tcenter = (X[i], Y[i]+0.3*sc)
            c = sample_color(children,i)
            numbers.add(dwg.text(str(i), insert=tcenter, fill=c, stroke=c))

    label_lines = dwg.add(dwg.g(id='lines', stroke='black', stroke_linecap='round', stroke_width=line_width))
    # Now number the collocated samples.
    for i in range(1, nsamp):
        nkids=len(children[i])

        if nkids>1 and collocated[i]:
            # find all of the links from this ponits
            arms = []
            # the parent arm (if it exists)
            p = P[i]
            while p!=-1 and collocated[p]:
                p = P[p]
            if not p==-1:
                arms.append(p)

            # find the child arms
            arms += sub_arms(X,Y,children,collocated,i)

            angles = [angle(X,Y,i,j) for j in arms]
            angles.sort()
            angles.append(angles[0]+2*math.pi)

            diff = [angles[j]-angles[j-1] for j in range(1,len(angles))]

            label_ids = coll_parents(X,Y,P,i)+coll_children(X,Y,children,i)
            out_rad = 1.5*max([R[j] for j in label_ids])

            for j in range(len(label_ids)):
                l = label_ids[j]
                stroke = sample_color(children, l)
                in_rad = R[l]
                u = unit_at_angle(angles[j]+diff[j]/2)
                center=(X[i], Y[i])
                start = (X[i]+in_rad *u[0], Y[i]+in_rad *u[1])
                end   = (X[i]+out_rad*u[0], Y[i]+out_rad*u[1])
                label_lines.add(dwg.line(start=start, end=end, stroke=stroke))
                text_pos = (end[0]+sc*u[0], end[1]+sc*u[1])
                numbers.add(dwg.text(str(l), insert=text_pos, stroke=stroke, fill=stroke))

    dwg.viewbox(minx, miny, width, height)

    dwg.save()

def generate(path=''):

    npos = -1

    tree1 = {
        'p': [npos],
        'x': [0],
        'y': [0],
        'r': [2],
    }

    tree2 = {
        'p': [npos, 0],
        'x': [0, 10],
        'y': [0, 0],
        'r': [2, 1],
    }

    tree3 = {
        'p': [npos, 0, 1, 1],
        'x': [0, 10, 15, 15],
        'y': [0, 0, 3, -3],
        'r': [2, 1.5, 1, 1],
    }

    tree4 = {
        'p': [npos, 0, 1, 2, 1, 4,],
        'x': [0, 10, 10, 15, 10, 15,],
        'y': [0, 0, 0, 3, 0, -3,],
        'r': [2, 1.5, 1, 1, 1, 1],
    }

    treexx = {
        'p': [npos, 0,    1, 2,  2],
        'x': [0,   10,   10, 15, 15],
        'y': [0,    0,    0,  3, -3],
        'r': [2,    1.5,  1,  1,  1],
    }

    tree5 = {
        'p': [npos, 0, 1],
        'x': [0, 2, 10],
        'y': [0, 0, 0],
        'r': [2, 1, 1]
    }

    make_annotated(path+'/stree.svg')
    make_image(tree1, path+'/tree1.svg')
    make_image(tree2, path+'/tree2.svg')
    make_image(tree3, path+'/tree3.svg')
    make_image(tree4, path+'/tree4.svg')
    make_image(tree5, path+'/tree5.svg')

if __name__ == '__main__':
    generate('.')

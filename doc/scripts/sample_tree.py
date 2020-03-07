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

def align_number(end, u, h):
    shift = 0.4*h
    x = end[0]+shift*u[0]
    y = end[1]+shift*u[1]
    theta = angle(u[0], u[1])/math.pi*180
    if theta<45:
        anchor='start'
    elif theta<135:
        anchor='middle'
    else:
        anchor='end'

    if u[1]>0:
        y += h

    return (x,y), anchor
#
# ############################################
#

def make_image(samples, filename, sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.15*sc

    # Padding around image.
    fudge=0.5*sc

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
        iscol = len(collocated[i])>1

        if iscol and (i==0 or not is_collocated(X,Y,i,P[i])):
            angles = [angle_norm(X,Y,i,j) for j in arms[i]]
            angles.sort()
            angles.append(angles[0]+2*math.pi)

            diff = [angles[j]-angles[j-1] for j in range(1,len(angles))]

            label_ids = collocated[i]
            out_rad = 1.25*max([R[j] for j in label_ids])

            center= (X[i], Y[i])
            for j in range(len(label_ids)):
                l = label_ids[j]
                color = sample_color(children,l)
                in_rad = R[l]
                u = unit_at_angle(angles[j]+diff[j]/2)
                start = (X[i]+in_rad *u[0], Y[i]+in_rad *u[1])
                end   = (X[i]+out_rad*u[0], Y[i]+out_rad*u[1])
                label_lines.add(dwg.line(start=start, end=end, stroke=color))
                text_pos, anchor = align_number(end, u, 0.7*sc)
                numbers.add(dwg.text(str(l), insert=text_pos, stroke=color, fill=color, text_anchor=anchor))

        elif not iscol:
            color = sample_color(children,i)
            tcenter = (X[i], Y[i]+0.3*sc)
            numbers.add(dwg.text(str(i), insert=tcenter, fill=color, stroke=color))


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

    tree4a = {
        'p': [npos, 0, 1,  2,  1,  4,],
        'x': [0, 10,  10, 15, 10, 15,],
        'y': [0, 0,    0,  3,  0, -3,],
        'r': [2, 1.5,  1,  1,  1,  1],
    }

    tree4b = {
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
    make_image(tree4a, path+'/tree4a.svg')
    make_image(tree4b, path+'/tree4b.svg')
    make_image(tree5, path+'/tree5.svg')

if __name__ == '__main__':
    generate('.')

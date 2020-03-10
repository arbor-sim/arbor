import svgwrite
import math

#
# Helpers for working with 2D vectors
#

def norm_vec(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

def unit_vec(v):
    L = norm_vec(v)
    return (v[0]/L, v[1]/L)

def rot90_vec(v):
    return (v[1], -v[0])

def add_vec(u, v):
    return (u[0]+v[0], u[1]+v[1])

def sub_vec(u, v):
    return (u[0]-v[0], u[1]-v[1])

def scal_vec(alpha, v):
    return (alpha*v[0], alpha*v[1])

#
# ############################################
#

def make_image(morph, filename, sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.1*sc

    # Padding around image.
    fudge=1*sc

    nbranches = len(morph)

    linecolor='black'
    pointcolor='red'
    lines = dwg.add(dwg.g(id='lines',
                          stroke=linecolor,
                          fill='white',
                          stroke_width=line_width))
    points = dwg.add(dwg.g(id='points',
                           stroke=pointcolor,
                           fill=pointcolor,
                           stroke_width=line_width))

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    minx = 0
    miny = 0
    maxx = 0
    maxy = 0

    for i in range(nbranches):
        branch = morph[i]

        # Extract sample locations and radii
        X = branch['x']
        Y = branch['y']
        R = branch['r']
        nsamp = len(X)

        # Scale locations and radii for drawing
        X = [ sc*x for x in X]
        Y = [-sc*x for x in Y]
        R = [ sc*x for x in R]

        minx = min([minx]+[X[i]-R[i] for i in range(nsamp)])
        miny = min([miny]+[Y[i]-R[i] for i in range(nsamp)])
        maxx = max([maxx]+[X[i]+R[i] for i in range(nsamp)])
        maxy = max([maxy]+[Y[i]+R[i] for i in range(nsamp)])

        is_sphere = branch['kind'] == 'sphere'

        if is_sphere:
            center = (X[0], Y[0])
            radius = R[0]
            lines.add(dwg.circle(center=center, r=radius))

        else:
            for j in range(1, nsamp):
                b = (X[j-1], Y[j-1])
                e = (X[j],   Y[j])
                d = sub_vec(e,b)
                o = rot90_vec(unit_vec(d))
                rb = R[j-1]
                re = R[j]
                p1 = add_vec(b, scal_vec(rb, o))
                p2 = add_vec(e, scal_vec(re, o))
                p3 = sub_vec(e, scal_vec(re, o))
                p4 = sub_vec(b, scal_vec(rb, o))
                lines.add(dwg.polygon(points=[p1,p2,p3,p4]))

        for j in range(nsamp):
            points.add(dwg.circle(center=(X[j], Y[j]), r=sc*0.05))

    # Find extent of image.
    minx -= fudge
    miny -= fudge
    maxx += fudge
    maxy += fudge
    width = maxx-minx
    height = maxy-miny
    dwg.viewbox(minx, miny, width, height)

    # Write the image to file.
    dwg.save()

def generate(path=''):

    npos = -1

    morph1 = [
        {'kind': 'sphere',
         'x': [0],
         'y': [0],
         'r': [3],
         'parent':-1},
    ]

    morph2 = [
        {'kind': 'cable',
         'x': [0, 10],
         'y': [0, 0],
         'r': [1, 0.5],
         'parent':-1},
    ]

    morph3 = [
        {'kind': 'cable',
         'x': [0, 10],
         'y': [0, 0],
         'r': [1, 0.5],
         'parent':-1},
        {'kind': 'cable',
         'x': [10, 17],
         'y': [0, 3],
         'r': [0.5, 0.25],
         'parent':1},
        {'kind': 'cable',
         'x': [10, 17],
         'y': [0, -3],
         'r': [0.5, 0.25],
         'parent':1},
    ]

    morph4 = [
        {'kind': 'cable',
         'x': [0, 10],
         'y': [0, 0],
         'r': [1, 0.5],
         'parent':-1},
        {'kind': 'cable',
         'x': [10, 17],
         'y': [0, 3],
         'r': [0.25, 0.25],
         'parent':1},
        {'kind': 'cable',
         'x': [10, 17],
         'y': [0, -3],
         'r': [0.25, 0.25],
         'parent':1},
    ]

    morph5 = [
        {'kind': 'sphere',
         'x': [0],
         'y': [0],
         'r': [2],
         'parent':-1},
        {'kind': 'cable',
         'x': [2, 10],
         'y': [0, 0],
         'r': [1, 0.5],
         'parent':0},
    ]

    morphx = [
        {'kind': 'sphere',
         'x': [0],
         'y': [0],
         'r': [2],
         'parent':-1},
        {'kind': 'cable',
         'x': [2, 15],
         'y': [0, 0],
         'r': [0.7, 0.5],
         'parent':0},
        {'kind': 'cable',
         'x': [15, 17, 20],
         'y': [0, 1, 4],
         'r': [0.5, 0.2, 0.2],
         'parent':1},
        {'kind': 'cable',
         'x': [15, 17, 20],
         'y': [0, -2, -3],
         'r': [0.3, 0.1, 0.1],
         'parent':1},
    ]

    make_image(morph1, path+'/morph1.svg')
    make_image(morph2, path+'/morph2.svg')
    make_image(morph3, path+'/morph3.svg')
    make_image(morph4, path+'/morph4.svg')
    make_image(morph5, path+'/morph5.svg')

if __name__ == '__main__':
    generate('.')

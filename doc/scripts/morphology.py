import svgwrite
import math
import tree_inputs as trees

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

def is_colloc(X,Y,i,j):
    return X[i]==X[j] and Y[i]==Y[j]

#
# ############################################
#

def morph_image(morph, filename, draw_segments=[True,True], sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.1*sc

    # Padding around image.
    fudge=1.5*sc

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
    numbers = dwg.add(dwg.g(id='numbers',
                             text_anchor='middle',
                             alignment_baseline='middle'))

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    offset = 0

    methods = []
    if draw_segments[0]: methods.append('withseg')
    if draw_segments[1]: methods.append('noseg')

    bcolor = 'mediumslateblue'
    pcolor = 'white'
    segfillcolor = 'lightgray'

    for method in methods:
        for i in range(nbranches):
            branch = morph[i]

            # Extract sample locations and radii
            X = branch['x']
            Y = branch['y']
            R = branch['r']
            nsamp = len(X)

            # Scale locations and radii for drawing
            X = [ sc*x + offset for x in X]
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
                lines.add(dwg.circle(center=center, r=radius, fill=segfillcolor))
                if method=='withseg':
                    points.add(dwg.circle(center=(X[0], Y[0]), stroke='black', r=sc*0.2, fill=pcolor))
                else:
                    # Setting alignment-baseline doesn't have any effect on text positioning,
                    # so we adjust manually by nudging the Y positin of the text.
                    points.add(dwg.circle(center=(X[0], Y[0]), stroke=bcolor, r=sc*0.55, fill=bcolor))
                    numbers.add(dwg.text(str(i), insert=(X[0], Y[0]+sc/3), stroke='white', fill='white', alignment_baseline='middle'))

            else:
                if method=='withseg':
                    for j in range(1, nsamp):
                        b = (X[j-1], Y[j-1])
                        e = (X[j],   Y[j])
                        d = sub_vec(e,b)
                        if norm_vec(d)>0.00001: # only draw nonzero length segments
                            o = rot90_vec(unit_vec(d))
                            rb = R[j-1]
                            re = R[j]
                            p1 = add_vec(b, scal_vec(rb, o))
                            p2 = add_vec(e, scal_vec(re, o))
                            p3 = sub_vec(e, scal_vec(re, o))
                            p4 = sub_vec(b, scal_vec(rb, o))
                            lines.add(dwg.polygon(points=[p1,p2,p3,p4], fill=segfillcolor))

                    for j in range(nsamp):
                        points.add(dwg.circle(center=(X[j], Y[j]), stroke='black', r=sc*0.2, fill=pcolor))

                elif method=='noseg':
                    index = []
                    for j in range(nsamp-1):
                        if not is_colloc(X,Y,j,j+1):
                            index.append([j, j+1])
                    nseg = len(index)
                    left = []
                    right = []
                    for k in range(nseg):
                        bi = index[k][0]
                        ei = index[k][1]
                        b = (X[bi], Y[bi])
                        e = (X[ei], Y[ei])
                        d = sub_vec(e,b)
                        o = rot90_vec(unit_vec(d))
                        rb = R[bi]
                        re = R[ei]
                        p1 = add_vec(b, scal_vec(rb, o))
                        p2 = add_vec(e, scal_vec(re, o))
                        p3 = sub_vec(e, scal_vec(re, o))
                        p4 = sub_vec(b, scal_vec(rb, o))
                        left += [p1, p2]
                        right += [p4, p3]
                    right.reverse()
                    lines.add(dwg.polygon(points=left+right,fill=segfillcolor))
                    # Place the number in the "middle" of the branch
                    if not nseg%2:
                        # Even number of segments: location lies at interface between segments
                        k = index[int(nseg/2)][0]
                        pos = (X[k], Y[k])
                    else:
                        # odd number of segments: location lies in center of middle segment
                        k1, k2 = index[int((nseg-1)/2)]
                        pos = ((X[k1]+X[k2])/2, (Y[k1]+Y[k2])/2)
                    label_pos = (pos[0], pos[1]+sc/3)
                    points.add(dwg.circle(center=pos, stroke=bcolor, r=sc*0.55, fill=bcolor))
                    numbers.add(dwg.text(str(i), insert=label_pos, stroke='white', fill='white', alignment_baseline='middle'))

        offset = maxx - minx + sc


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

    # spherical morpho: no need for two images
    morph_image(trees.morph1,  path+'/morph1.svg', draw_segments=[True,False])

    # single cable segment
    morph_image(trees.morph2a, path+'/morph2a.svg')
    # cables with multipe segments
    morph_image(trees.morph2b, path+'/morph2b.svg')
    morph_image(trees.morph2c, path+'/morph2c.svg')

    # the y-shaped cells have one segment per branch
    morph_image(trees.morph3,  path+'/morph3.svg')
    morph_image(trees.morph4a, path+'/morph4.svg')

    morph_image(trees.morph5a, path+'/morph5a.svg')
    morph_image(trees.morph5b, path+'/morph5b.svg')
    morph_image(trees.morph6a, path+'/morph6a.svg')
    morph_image(trees.morph6b, path+'/morph6b.svg')
    morph_image(trees.morph6c, path+'/morph6c.svg')

if __name__ == '__main__':
    generate('.')

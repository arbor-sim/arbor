import copy
import svgwrite
import math
import tree_inputs as trees

tag_colors = ['white', '#ffc2c2', 'gray', '#c2caff']

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

def is_collocated(x, y):
    return x[0]==y[0] and x[1]==y[1]

#
# ############################################
#

def branch_meta(morph):
    stats = []

    for branch in morph:
        # Extract sample locations and radii
        X = branch['x']
        Y = branch['y']
        R = branch['r']
        T = branch['t']
        nsamp = len(X)

        L = 0
        lens = [0]

        for i in range(nsamp-1):
            l = norm_vec((X[i+1]-X[i], Y[i+1]-Y[i]))
            L += l
            lens.append(L)

        if L>0:
            for i in range(len(lens)):
                lens[i] /= L

        # handle zero length branches... heaven forbid actually trying.
        lens[-1] = 1

        stats.append(lens)

    return stats

# 0 ≤ pos ≤ 1
# 0 ≤ lens[i] ≤ 1; lens[i] ≤ lens[i+1]
# return largest i that satisfies lens[i]<pos
def find_pos(lens, pos):
    for i,L in enumerate(lens):
        if L>pos: return i-1
    return i

# Return {index, x, y, r} of the sample at pos ∈ [0,1] on branch.
# If pos lies between two samples the x,y,r values are interpolated
# and index is the index of the proximal end of the cable segment.
def sample_by_pos(branch, lens, pos):
    X = branch['x']
    Y = branch['y']
    R = branch['r']

    # find segment in branch that contains pos
    i = find_pos(lens, pos)

    if pos==lens[i]:
        return (i, X[i], Y[i], R[i])

    rel = (pos-lens[i])/(lens[i+1]-lens[i])
    r = R[i] + rel*(R[i+1]-R[i])
    x = X[i] + rel*(X[i+1]-X[i])
    y = Y[i] + rel*(Y[i+1]-Y[i])

    return (i, x, y, r)

# todo: handle zero-length cable
def cable_corners(b, e, rb, re, left, right):
    if is_collocated(e,b):
        return

    o = rot90_vec(unit_vec(sub_vec(e,b)))
    p1 = add_vec(b, scal_vec(rb, o))
    p2 = add_vec(e, scal_vec(re, o))
    p3 = sub_vec(e, scal_vec(re, o))
    p4 = sub_vec(b, scal_vec(rb, o))
    left += [p1, p2]
    right += [p4, p3]
#
# ############################################
#

def morph_image(morphs, methods, filename, sc=20):
    assert(len(morphs)==len(methods))

    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.1*sc

    # Padding around image.
    fudge=1.5*sc

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

    bcolor = 'mediumslateblue'
    pcolor = 'white'
    branchfillcolor = 'lightgray'

    nmorph = len(morphs)

    for l in range(nmorph):
        morph = morphs[l]
        method = methods[l]

        nbranches = len(morph)

        for i in range(nbranches):
            branch = morph[i]

            # Extract sample locations and radii
            X = branch['x']
            Y = branch['y']
            R = branch['r']
            T = branch['t']
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
                color = tag_colors[T[0]] if method=='segments' else branchfillcolor
                lines.add(dwg.circle(center=center, r=radius, fill=color))
                if method=='segments':
                    points.add(dwg.circle(center=(X[0], Y[0]), stroke='black', r=sc*0.2, fill=pcolor))
                else:
                    # Setting alignment-baseline doesn't have any effect on text positioning,
                    # so we adjust manually by nudging the Y positin of the text.
                    points.add(dwg.circle(center=(X[0], Y[0]), stroke=bcolor, r=sc*0.55, fill=bcolor))
                    # alignment_baseline
                    #   - works on Chrome/Chromium
                    #   - doesn't work on Firefox
                    numbers.add(dwg.text(str(i), insert=(X[0], Y[0]+sc/3), stroke='white', fill='white'))
                    numbers.add(dwg.text(str(i), insert=(X[0], Y[0]+sc/3), stroke='white', fill='white', alignment_baseline='middle'))

            else:
                if method=='segments':
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
                            lines.add(dwg.polygon(points=[p1,p2,p3,p4], fill=tag_colors[T[j]]))

                    for j in range(nsamp):
                        points.add(dwg.circle(center=(X[j], Y[j]), stroke='black', r=sc*0.2, fill=pcolor))

                elif method=='branches':
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
                    lines.add(dwg.polygon(points=left+right,fill=branchfillcolor))
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
                    # alignment_baseline
                    #   - works on Chrome/Chromium
                    #   - doesn't work on Firefox
                    numbers.add(dwg.text(str(i), insert=label_pos, stroke='white', fill='white'))
                    #numbers.add(dwg.text(str(i), insert=label_pos, stroke='white', fill='white', alignment_baseline='middle'))

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

def label_image(morphology, labels, filename, sc=20):
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)
    morph = copy.deepcopy(morphology)

    # Width of lines and circle strokes.
    line_width=0.1*sc

    # Padding around image.
    fudge=1.5*sc

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

    offset = None

    branchfillcolor = 'lightgray'
    bcol = branchfillcolor

    # Scale sample locations and radius for drawing
    for branch in morph:
        branch['x'] = [ sc*x for x in branch['x']]
        branch['y'] = [-sc*x for x in branch['y']]
        branch['r'] = [ sc*x for x in branch['r']]
    meta = branch_meta(morph)

    nimage = len(labels)
    for l in range(nimage):
        lab = labels[l]

        nbranches = len(morph)

        # Draw the outline of the cell
        for i in range(nbranches):
            branch = morph[i]

            # Extract sample locations and radii
            X = branch['x']
            Y = branch['y']
            R = branch['r']
            T = branch['t']
            nsamp = len(X)

            minx = min([minx]+[X[i]-R[i] for i in range(nsamp)])
            miny = min([miny]+[Y[i]-R[i] for i in range(nsamp)])
            maxx = max([maxx]+[X[i]+R[i] for i in range(nsamp)])
            maxy = max([maxy]+[Y[i]+R[i] for i in range(nsamp)])

            is_sphere = branch['kind'] == 'sphere'

            if is_sphere:
                center = (X[0], Y[0])
                radius = R[0]
                lines.add(dwg.circle(center=center, r=radius, fill=bcol, stroke=bcol))

            else:
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
                lines.add(dwg.polygon(points=left+right, fill=bcol, stroke=bcol))

        # Draw locset if requested
        if lab['type'] == 'locset':
            for loc in lab['value']:
                bid = loc[0]
                pos = loc[1]
                idx, x, y, r = sample_by_pos(morph[bid], meta[bid], pos)

                points.add(dwg.circle(center=(x,y), stroke='black', r=sc/3, fill='black'))

        if lab['type'] == 'region':
            for cab in lab['value']:
                # skip zero length cables
                if cab[1]==cab[2]: continue

                bid  = cab[0]
                branch = morph[bid]
                lens   = meta[bid]
                prox = sample_by_pos(branch, lens, cab[1])
                dist = sample_by_pos(branch, lens, cab[2])

                X = branch['x']
                Y = branch['y']
                R = branch['r']

                pointl  = [(prox[1], prox[2])]
                radl    = [prox[3]]
                for k in range(prox[0]+1, dist[0]+1):
                    pointl += [(X[k], Y[k])]
                    radl   += [R[k]]
                pointl += [(dist[1], dist[2])]
                radl   += [dist[3]]
                left = []
                right = []
                np = len(pointl)
                for k in range(np-1):
                    cable_corners(pointl[k], pointl[k+1], radl[k], radl[k+1], left, right)

                right.reverse()
                lines.add(dwg.polygon(points=left+right, fill='black', stroke='lightgray'))

        if offset==None:
            offset = maxx - minx + sc

        for branch in morph:
            branch['x'] = [ x+offset for x in branch['x']]


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
    morph_image([trees.morph1], ['branchs'],  path+'/morph1.svg')

    # single cable segment
    morph_image([trees.morph2a, trees.morph2a], ['segments','branches'], path+'/morph2a.svg')
    # cables with multipe segments
    morph_image([trees.morph2b, trees.morph2b], ['segments','branches'], path+'/morph2b.svg')
    morph_image([trees.morph2c, trees.morph2c], ['segments','branches'], path+'/morph2c.svg')

    # the y-shaped cells have one segment per branch
    morph_image([trees.morph3a,  trees.morph3a],['segments','branches'], path+'/morph3a.svg')
    morph_image([trees.morph3b, trees.morph3b], ['segments','branches'], path+'/morph3b.svg')

    morph_image([trees.morph4a, trees.morph4a], ['segments','branches'], path+'/morph4a.svg')
    morph_image([trees.morph4b, trees.morph4b], ['segments','branches'], path+'/morph4b.svg')

    morph_image([trees.morph5a_sphere, trees.morph5a_sphere], ['segments','branches'], path+'/morph5a_sphere.svg')
    morph_image([trees.morph5a_cable,  trees.morph5a_cable],  ['segments','branches'], path+'/morph5a_cable.svg')
    morph_image([trees.morph5b_sphere, trees.morph5b_sphere], ['segments','branches'], path+'/morph5b_sphere.svg')
    morph_image([trees.morph5b_cable,  trees.morph5b_cable],  ['segments','branches'], path+'/morph5b_cable.svg')

    morph_image([trees.morph5a_cable, trees.morph5a_sphere], ['branches','branches'], path+'/morph-branches.svg')
    morph_image([trees.morph5a_cable, trees.morph5a_sphere], ['segments','segments'], path+'/morph-segments.svg')

    morph_image([trees.morph6, trees.morph6], ['segments','branches'], path+'/morph6.svg')

    morph_image([trees.morphlab, trees.morphlab], ['segments','branches'], path+'/morphlab.svg')

    term_lab  = {'type':'locset', 'value': [( 1,1.000), ( 3,1.000), ( 4,1.000), ( 5,1.000), ]}
    rand_dend_lab = {'type':'locset', 'value': [( 0,0.457), ( 0,0.629), ( 0,0.636), ( 0,0.768), ( 0,0.901), ( 0,0.973), ( 1,0.001), ( 1,0.019), ( 1,0.082), ( 1,0.152), ( 1,0.191), ( 1,0.206), ( 1,0.309), ( 1,0.396), ( 1,0.428), ( 1,0.560), ( 1,0.646), ( 1,0.790), ( 1,0.794), ( 1,0.818), ( 1,0.825), ( 1,0.947), ( 2,0.101), ( 2,0.104), ( 2,0.470), ( 2,0.529), ( 2,0.560), ( 2,0.575), ( 2,0.681), ( 2,0.696), ( 2,0.771), ( 2,0.788), ( 2,0.819), ( 2,0.824), ( 2,0.868), ( 2,0.898), ( 2,0.923), ( 3,0.046), ( 3,0.046), ( 3,0.047), ( 3,0.151), ( 3,0.228), ( 3,0.249), ( 3,0.699), ( 4,0.005), ( 4,0.255), ( 4,0.345), ( 4,0.408), ( 4,0.845), ( 4,0.890), ]}

    label_image(trees.morphlab, [term_lab, rand_dend_lab], path+'/locset_label_examples.svg')

    axon_lab   = {'type':'region', 'value': [( 0,0.000,0.000), ( 5,0.000,1.000), ]}
    dend_lab   = {'type':'region', 'value': [( 0,0.332,1.000), ( 1,0.000,1.000), ( 2,0.000,1.000), ( 3,0.000,1.000), ( 4,0.000,1.000), ]}
    radlt5_lab = {'type':'region', 'value': [( 1,0.439,1.000), ( 2,1.000,1.000), ( 3,0.000,1.000), ( 4,0.000,1.000), ( 5,0.656,1.000), ]}
    radle5_lab = {'type':'region', 'value': [( 0,1.000,1.000), ( 1,0.000,0.000), ( 1,0.439,1.000), ( 2,0.000,1.000), ( 3,0.000,1.000), ( 4,0.000,1.000), ( 5,0.656,1.000), ]}

    label_image(trees.morphlab, [dend_lab, radlt5_lab], path+'/region_label_examples.svg')

    root_lab   = {'type':'locset', 'value': [( 0,0.000), ]}
    label_image(trees.morphlab, [root_lab], path+'/root_label.svg')
    label_image(trees.morphlab, [term_lab], path+'/term_label.svg')

    location_lab   = {'type':'locset', 'value': [(1, 0.5)]}
    label_image(trees.morphlab, [location_lab], path+'/location_label.svg')

    radin3_5_lab = {'type':'region', 'value': [( 0,1.000,1.000), ( 1,0.000,0.000), ( 1,0.439,0.793), ( 2,0.000,1.000), ( 3,0.000,0.667), ( 4,0.000,0.391), ( 5,0.656,1.000), ]}
    distloc_lab = {'type':'locset', 'value': [( 1,0.793), ( 3,0.667), ( 4,0.391), ( 5,1.000), ]}
    proxloc_lab = {'type':'locset', 'value': [( 0,1.000), ]}
    label_image(trees.morphlab, [radin3_5_lab, proxloc_lab], path+'/prox_label.svg')
    label_image(trees.morphlab, [radin3_5_lab, distloc_lab], path+'/dist_label.svg')

    uniform0_lab = {'type':'locset', 'value': [( 0,0.975), ( 1,0.200), ( 1,0.841), ( 2,0.924), ( 2,0.927), ( 2,0.991), ( 3,0.993), ( 4,0.364), ( 4,0.479), ( 4,0.514), ]}
    uniform1_lab = {'type':'locset', 'value': [( 0,0.455), ( 0,0.942), ( 1,0.201), ( 1,0.281), ( 1,0.496), ( 1,0.685), ( 2,0.224), ( 3,0.795), ( 4,0.013), ( 4,0.319), ]}
    label_image(trees.morphlab, [uniform0_lab, uniform1_lab], path+'/uniform_label.svg')

if __name__ == '__main__':
    generate('.')

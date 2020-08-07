import copy
import svgwrite
import math
import inputs

tag_colors = ['white', '#ffc2c2', 'gray', '#c2caff']

#
# ############################################
#

def translate(x, f, xshift):
    return (f*x[0]+xshift, -f*x[1])

def translate_all(points, f, xshift):
    return [translate(x, f, xshift) for x in points]

# Draw one or more morphologies, side by side.
# Each morphology can be drawn as segments or branches.
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
    branchfillcolor = 'lightgray'

    nmorph = len(morphs)

    for l in range(nmorph):
        morph = morphs[l]
        method = methods[l]

        nbranches = len(morph)

        for i in range(nbranches):
            branch = morph[i]

            lx, ux, ly, uy = branch.minmax()
            minx = min(minx,  sc*lx+offset)
            miny = min(miny,  sc*ly)
            maxx = max(maxx,  sc*ux+offset)
            maxy = max(maxy,  sc*uy)

            if method=='segments':
                for sec in branch.sections:
                    for seg in sec:
                        if seg.length>0.00001: # only draw nonzero length segments
                            line = translate_all(seg.corners(), sc, offset)
                            lines.add(dwg.polygon(points=line, fill=tag_colors[seg.tag]))

            elif method=='branches':
                for line in branch.outline():
                    lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                          fill=branchfillcolor))

                pos = translate(branch.location(0.5), sc, offset)
                points.add(dwg.circle(center=pos,
                                      stroke=bcolor,
                                      r=sc*0.55,
                                      fill=bcolor))
                # The svg alignment_baseline attribute:
                #   - works on Chrome/Chromium
                #   - doesn't work on Firefox
                # so for now we just shift the relative position by sc/3
                label_pos = (pos[0], pos[1]+sc/3)
                numbers.add(dwg.text(str(i),
                                      insert=label_pos,
                                      stroke='white',
                                      fill='white'))
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

# Generate an image that illustrates regions and locsets on a morphology.
#
# Can't handle morpholgies with gaps, where segemnts with a parent-child
# ordering don't have collocated distal-proximal locations respectively.
# Handling this case would make rendering regions more complex, but would
# not bee too hard to support.
def label_image(morphology, labels, filename, sc=20):
    morph = morphology
    print('image:', filename)
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    # Width of lines and circle strokes.
    line_width=0.2*sc

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

    offset = 0

    branchfillcolor = 'lightgray'

    nimage = len(labels)
    for l in range(nimage):
        lab = labels[l]

        nbranches = len(morph)

        # Draw the outline of the cell
        for i in range(nbranches):
            branch = morph[i]

            lx, ux, ly, uy = branch.minmax()
            minx = min(minx,  sc*lx+offset)
            miny = min(miny,  sc*ly)
            maxx = max(maxx,  sc*ux+offset)
            maxy = max(maxy,  sc*uy)

            for line in branch.outline():
                lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                      fill=branchfillcolor,
                                      stroke=branchfillcolor))

        # Draw the root
        root = translate(morph[0].location(0), sc, offset)
        points.add(dwg.circle(center=root, stroke='red', r=sc/2.5, fill='white'))

        if lab['type'] == 'locset':
            for loc in lab['value']:
                bid = loc[0]
                pos = loc[1]

                loc = translate(morph[bid].location(pos), sc, offset)
                points.add(dwg.circle(center=loc, stroke='black', r=sc/3, fill='white'))

        if lab['type'] == 'region':
            for cab in lab['value']:
                # skip zero length cables
                bid  = cab[0]
                ppos = cab[1]
                dpos = cab[2]

                # Don't draw zero-length cables
                # How should these be drawn: with a line or a circle?
                if ppos==dpos: continue

                for line in morph[bid].outline(ppos, dpos):
                    lines.add(dwg.polygon(points=translate_all(line, sc, offset),
                                          fill='black',
                                          stroke=branchfillcolor))

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
    # TODO: make this a cylinder
    #morph_image([trees.morph1], ['branches'],  path+'/morph1.svg')

    # single cable segment
    #morph_image([trees.morph2a, trees.morph2a], ['segments','branches'], path+'/morph2a.svg')
    # cables with multipe segments
    #morph_image([trees.morph2b, trees.morph2b], ['segments','branches'], path+'/morph2b.svg')
    #morph_image([trees.morph2c, trees.morph2c], ['segments','branches'], path+'/morph2c.svg')

    # the y-shaped cells have one segment per branch
    #morph_image([trees.morph3a,  trees.morph3a],['segments','branches'], path+'/morph3a.svg')
    #morph_image([trees.morph3b, trees.morph3b], ['segments','branches'], path+'/morph3b.svg')

    #morph_image([trees.morph4a, trees.morph4a], ['segments','branches'], path+'/morph4a.svg')

    #morph_image([trees.morph5a_cable,  trees.morph5a_cable],  ['segments','branches'], path+'/morph5a_cable.svg')
    #morph_image([trees.morph5b_cable,  trees.morph5b_cable],  ['segments','branches'], path+'/morph5b_cable.svg')

    #morph_image([trees.morph6, trees.morph6], ['segments','branches'], path+'/morph6.svg')

    morph_image([inputs.label_morph, inputs.label_morph], ['segments', 'branches'], path+'/label_morph.svg')

    ####################### locsets

    label_image(inputs.label_morph, [inputs.ls_term, inputs.ls_rand_dend], path+'/locset_label_examples.svg')

    label_image(inputs.label_morph, [inputs.reg_dend, inputs.reg_radlt5], path+'/region_label_examples.svg')

    label_image(inputs.label_morph, [inputs.ls_root], path+'/root_label.svg')
    label_image(inputs.label_morph, [inputs.ls_term], path+'/term_label.svg')

    label_image(inputs.label_morph, [inputs.ls_loc15], path+'/location_label.svg')

    label_image(inputs.label_morph, [inputs.reg_rad36, inputs.ls_distal], path+'/distal_label.svg')
    label_image(inputs.label_morph, [inputs.reg_rad36, inputs.ls_proximal], path+'/proximal_label.svg')
    label_image(inputs.label_morph, [inputs.ls_uniform0, inputs.ls_uniform1], path+'/uniform_label.svg')
    label_image(inputs.label_morph, [inputs.ls_branchmid], path+'/on_branches_label.svg')

    label_image(inputs.label_morph, [inputs.ls_term, inputs.reg_tag3, inputs.ls_restrict], path+'/restrict_label.svg')

    ####################### regions

    label_image(inputs.label_morph, [inputs.reg_empty, inputs.reg_all], path+'/nil_all_label.svg')

    label_image(inputs.label_morph, [inputs.reg_tag1, inputs.reg_tag2, inputs.reg_tag3], path+'/tag_label.svg')

    label_image(inputs.label_morph, [inputs.reg_branch0, inputs.reg_branch3], path+'/branch_label.svg')

    label_image(inputs.label_morph, [inputs.reg_cable_1_01, inputs.reg_cable_1_31, inputs.reg_cable_1_37], path+'/cable_label.svg')

    label_image(inputs.label_morph, [inputs.ls_proxint_in, inputs.reg_proxint],    path+'/proxint_label.svg')
    label_image(inputs.label_morph, [inputs.ls_proxint_in, inputs.reg_proxintinf], path+'/proxintinf_label.svg')
    label_image(inputs.label_morph, [inputs.ls_distint_in, inputs.reg_distint],    path+'/distint_label.svg')
    label_image(inputs.label_morph, [inputs.ls_distint_in, inputs.reg_distintinf], path+'/distintinf_label.svg')

    label_image(inputs.label_morph, [inputs.reg_lhs, inputs.reg_rhs, inputs.reg_or],  path+'/union_label.svg')
    label_image(inputs.label_morph, [inputs.reg_lhs, inputs.reg_rhs, inputs.reg_and], path+'/intersect_label.svg')

    label_image(inputs.label_morph, [inputs.reg_radlt5],  path+'/radiuslt_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radle5],  path+'/radiusle_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radgt5],  path+'/radiusgt_label.svg')
    label_image(inputs.label_morph, [inputs.reg_radge5],  path+'/radiusge_label.svg')


if __name__ == '__main__':
    generate('.')

import svgwrite

foobar = 'hello-world'

def generate(path):
    filename = path+'/stree.svg'
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
    lX = [-1,  12.5, 18.5, 15]
    lY = [ 3.5, 1.5,  9,   -5]
    lC = [ 1,   2,    3,    3]
    nlab = len(labels)

    lX = [sc*x for x in lX]
    lY = [-sc*x for x in lY]

    for i in range(nlab):
        c = colors[lC[i]]
        dwg.add(dwg.text(labels[i], insert=(lX[i], lY[i]), stroke=c, fill=c))

    dwg.viewbox(minx, miny, width, height)

    dwg.save()


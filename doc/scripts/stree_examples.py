import svgwrite
from svgwrite import cm, mm

def make_image(samples, filename, sc=20):
    dwg = svgwrite.Drawing(filename=filename, debug=True)

    line_width=0.1*sc
    fudge=0.5*sc

    colors = ['black', 'cornflowerblue', 'tomato', 'mediumseagreen']

    X = samples['x']
    Y = samples['y']
    R = samples['r']
    P = samples['p'] #C = samples['c']

    X = [sc*x for x in X]
    Y = [-sc*x for x in Y]
    R = [sc*x for x in R]
    nsamp = len(X)

    collocated = nsamp * [False]
    for i in range(nsamp):
        p = P[i]
        if X[i]==X[p] and Y[i]==Y[p]:
            collocated[p] = True;
            collocated[i] = True;

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
        circles.add(dwg.circle(center=center, r=R[i]))
        if not collocated[i]:
            tcenter = (X[i], Y[i]+0.3*sc)
            numbers.add(dwg.text(str(i), insert=tcenter))

    #labels = ['root', 'fork', 'terminal', 'terminal']
    #lX = [-1,  12.5, 18.5, 17]
    #lY = [ 3.5, 1.5,  9,   -6]
    #lC = [ 1,   2,    3,    3]
    #nlab = len(labels)

    #lX = [sc*x for x in lX]
    #lY = [-sc*x for x in lY]

    #for i in range(nlab):
    #    c = colors[lC[i]]
    #    dwg.add(dwg.text(labels[i], insert=(lX[i], lY[i]), stroke=c, fill=c))

    dwg.viewbox(minx, miny, width, height)

    dwg.save()

def generate(path=''):

    npos = -1

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
        'r': [2, 2, 1, 1],
    }

    tree4 = {
        'p': [npos, 0, 1, 2, 1, 4,],
        'x': [0, 10, 10, 15, 10, 15,],
        'y': [0, 0, 0, 3, 0, -3,],
        'r': [2, 2, 1, 1, 1, 1],
    }

    tree5 = {
        'p': [npos, 0, 1],
        'x': [0, 3, 10],
        'y': [0, 0, 0],
        'r': [3, 1, 1]
    }

    make_image(tree2, path+'/tree2.svg')
    make_image(tree3, path+'/tree3.svg')
    make_image(tree4, path+'/tree4.svg')
    make_image(tree5, path+'/tree5.svg')

if __name__ == '__main__':
    generate('.')

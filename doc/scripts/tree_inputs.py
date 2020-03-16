npos = -1

def make_morph(tree, branches):
    X = tree['x']
    Y = tree['y']
    R = tree['r']
    nb = len(branches)
    m = []
    for i in range(nb):
        b = {}
        ids = branches[i]
        if len(ids)==1:
            b = {
                'kind': 'sphere',
                'x': [X[0]],
                'y': [Y[0]],
                'r': [R[0]]
            }
        else:
            b = {
                'kind': 'cable',
                'x': [X[j] for j in ids],
                'y': [Y[j] for j in ids],
                'r': [R[j] for j in ids]
            }
        m.append(b)

    return m

tree1 = {
    'p': [npos],
    'x': [0],
    'y': [0],
    'r': [2],
    't': [1],
}
branches1 = [[0]]

# single cable segment
tree2a = {
    'p': [npos, 0],
    'x': [0, 10],
    'y': [0, 0],
    'r': [1, 0.5],
    't': [1, 1],
}
branches2a = [[0, 1]]

# unbranched cable with 4 segments
tree2b = {
    'p': [npos, 0, 1, 2, 3],
    'x': [0, 3, 5, 8, 10],
    'y': [0, 0.2, -0.1, 0,   0],
    'r': [1, 0.8, 0.7,  0.6, 0.5],
    't': [1,  1,  1,  1,  1],
}
branches2b = [[0, 1, 2, 3, 4]]

# unbranched cable with discontinuity of radius.
tree2c = {
    'p': [npos, 0, 1, 2, 3, 4],
    'x': [0, 3, 5, 8, 8, 10],
    'y': [0, 0.2, -0.1, 0, 0,  0],
    'r': [1, 0.8, 0.7,  0.6, 0.3, 0.5],
    't': [1,  1,  1,  1,  1, 1],
}
branches2c = [[0, 1, 2, 3, 4, 5]]

tree3 = {
    'p': [npos, 0, 1, 1],
    'x': [0, 10, 15, 15],
    'y': [0, 0, 3, -3],
    'r': [1, 0.5, 0.25, 0.25],
    't': [1,  1,  1,  1],
}
branches3 = [[0, 1], [1, 2], [1, 3]]

tree4a = {
    'p': [npos, 0, 1,  2,  1,  4,],
    'x': [0, 10,  10, 15, 10, 15,],
    'y': [0, 0,    0,  3,  0, -3,],
    'r': [1, 0.5,  0.25, 0.25,  0.25,  0.25],
    't': [1,  1,  1,  1,  1,  1],
}
branches4a = [[0, 1], [1, 2, 3], [1, 4, 5]]

tree4b = {
    'p': [npos, 0,      1,    2,  2],
    'x': [0,   10,     10,   15, 15],
    'y': [0,    0,      0,    3, -3],
    'r': [1,    0.5, 0.25, 0.25, 0.25],
    't': [1,    1, 1, 1, 1],
}
branches4b = [[0, 1, 2], [2, 3], [2, 4]]

tree5 = {
    'p': [npos, 0, 1],
    'x': [0, 2, 10],
    'y': [0, 0, 0],
    'r': [2, 1, 1],
    't': [1, 1, 1]
}
branches5a = [[0, 1, 2]]
branches5b = [[0], [1, 2]]

tree6a = {
   'x': [ 0,  5,   10,   15, 18,   23,   20],
   'y': [ 0, -1,    0.5,  0,  5,    8,   -4],
   'r': [ 3,  0.8,  0.5,  0.5,  0.3,  0.3,  0.3],
   'p': [-1,  0,    1,    2,  3,    4,    3],
   't': [ 1,  1,    1,    1,  1,    1,    1]
}

tree6b = {
   'x': [ 0,  3,     5,  10  , 15, 18,   23,   20],
   'y': [ 0,  -.8,    -1,   0.5,  0,  5,    8,   -4],
   'r': [ 3,  1.2, 1.2,   1.2,  1,  1,  0.7,  0.8],
   'r': [ 3,  0.8, 0.8,  0.5,  0.5,  0.3,  0.3,  0.3],
   'p': [-1,  0,     1,   2  ,  3,    4,  5,    4],
   't': [ 1,  1,     1,   1  ,  1,    1,   1,   1]
}
branches6a = [[0, 1, 2, 3], [3, 4, 5], [3, 6]]
branches6b = [[0], [1, 2, 3, 4], [4, 5, 6], [4, 7]]
branches6c = [[0], [2, 3, 4], [4, 5, 6], [4, 7]]

morph1 =  make_morph(tree1,  branches1)
morph2a = make_morph(tree2a, branches2a)
morph2b = make_morph(tree2b, branches2b)
morph2c = make_morph(tree2c, branches2c)
morph3 =  make_morph(tree3,  branches3)
morph4a = make_morph(tree4a, branches4a)
morph4b = make_morph(tree4b, branches4b)
morph5a = make_morph(tree5,  branches5a)
morph5b = make_morph(tree5,  branches5b)
morph6a = make_morph(tree6a, branches6a)
morph6b = make_morph(tree6b, branches6b)
morph6c = make_morph(tree6b, branches6c)


npos = -1

def make_morph(tree, branches):
    X = tree['x']
    Y = tree['y']
    R = tree['r']
    T = tree['t']
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
                'r': [R[0]],
                't': [T[0]]
            }
        else:
            b = {
                'kind': 'cable',
                'x': [X[j] for j in ids],
                'y': [Y[j] for j in ids],
                'r': [R[j] for j in ids],
                't': [T[j] for j in ids]
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
    't': [3, 3],
}
branches2a = [[0, 1]]

# unbranched cable with 4 segments
tree2b = {
    'p': [npos, 0, 1, 2, 3],
    'x': [0, 3, 5, 8, 10],
    'y': [0, 0.2, -0.1, 0,   0],
    'r': [1, 0.8, 0.7,  0.6, 0.5],
    't': [1,  1,  2,  2,  3],
}
branches2b = [[0, 1, 2, 3, 4]]

# unbranched cable with discontinuity of radius.
tree2c = {
    'p': [npos, 0, 1, 2, 3, 4],
    'x': [0, 3, 5, 8, 8, 10],
    'y': [0, 0.2, -0.1, 0, 0,  0],
    'r': [1, 0.8, 0.7,  0.6, 0.3, 0.5],
    't': [1,  1,  2,  2,  3, 3],
}
branches2c = [[0, 1, 2, 3, 4, 5]]

# Y shaped cells
tree3a = {
    'p': [npos, 0, 1, 1],
    'x': [0, 10, 15, 15],
    'y': [0, 0, 3, -3],
    'r': [1, 0.5, 0.25, 0.25],
    't': [3,  3,  3,  3],
}
branches3a = [[0, 1], [1, 2], [1, 3]]

tree3b = {
    'p': [npos, 0, 1,  2,  1,  4,],
    'x': [0, 10,  10, 15, 10, 15,],
    'y': [0, 0,    0,  3,  0, -3,],
    'r': [1, 0.5,  0.25, 0.25,  0.25,  0.25],
    't': [3,  3,  3,  3,  3,  3],
}
branches3b = [[0, 1], [1, 2, 3], [1, 4, 5]]

tree3c = {
    'p': [npos, 0,      1,    2,  2],
    'x': [0,   10,     10,   15, 15],
    'y': [0,    0,      0,    3, -3],
    'r': [1,    0.5, 0.25, 0.25, 0.25],
    't': [3,    3, 3, 3, 3],
}
branches3c = [[0, 1, 2], [2, 3], [2, 4]]

# ball and stick
tree4 = {
    'p': [npos, 0, 1],
    'x': [0, 2, 10],
    'y': [0, 0, 0],
    'r': [2, 1, 1],
    't': [1, 1, 3]
}
branches4a = [[0, 1, 2]]
branches4b = [[0], [1, 2]]

# soma + Y shaped dendrite
tree5a = {
   'x': [ 0,  5,   10,   15, 18,   23,   20],
   'y': [ 0, -1,    0.5,  0,  5,    8,   -4],
   'r': [ 3,  0.8,  0.5,  0.5,  0.3,  0.3,  0.3],
   'p': [-1,  0,    1,    2,  3,    4,    3],
   't': [ 1,  1,    3,    3,  2,    2,    3]
}

tree5b = {
   'x': [ 0,  3,     5,  10  , 15, 18,   23,   20],
   'y': [ 0,  -.8,    -1,   0.5,  0,  5,    8,   -4],
   'r': [ 3,  0.8, 0.8,  0.5,  0.5,  0.3,  0.3,  0.3],
   'p': [-1,  0,     1,   2  ,  3,    4,  5,    4],
   't': [ 1,  1,     3,   3  ,  3,    2,   2,   3]
}
branches5a_cable = [[0, 1, 2, 3], [3, 4, 5], [3, 6]]
branches5a_sphere= [[0], [1, 2, 3], [3, 4, 5], [3, 6]]
branches5b_cable = [[0, 1, 2, 3, 4], [4, 5, 6], [4, 7]]
branches5b_sphere= [[0], [1, 2, 3, 4], [4, 5, 6], [4, 7]]

tree6 = {
   'x': [ 0, 6,   6, 15.0, 20.0, 21.0,  -5.0],
   'y': [ 0, 0,   0,  5.0,  7.0, -3.0,   0.0],
   'r': [ 2, 2, 0.5,  0.5,  0.3,  0.3,   0.5],
   'p': [-1, 0,   1,    2,    3,    2,     0],
   't': [ 1, 1,   3,    3,    3,    3,     2]
}
branches6 = [[0, 1], [2, 3, 4], [2, 5], [0, 6]]

morph1 =  make_morph(tree1,  branches1)
morph2a = make_morph(tree2a, branches2a)
morph2b = make_morph(tree2b, branches2b)
morph2c = make_morph(tree2c, branches2c)
morph3a = make_morph(tree3a, branches3a)
morph3b = make_morph(tree3b, branches3b)
morph3c = make_morph(tree3c, branches3c)
morph4a = make_morph(tree4,  branches4a)
morph4b = make_morph(tree4,  branches4b)
morph5a_sphere= make_morph(tree5a, branches5a_sphere)
morph5a_cable = make_morph(tree5a, branches5a_cable)
morph5b_sphere= make_morph(tree5b, branches5b_sphere)
morph5b_cable = make_morph(tree5b, branches5b_cable)
morph6 = make_morph(tree6, branches6)


import numpy as np
from sympy import Eijk

nodes = np.array([[1, 2, 3],
                [1, 3, 2],
                [2, 1, 3],
                [2, 3, 1],
                [3, 1, 2],
                [3, 2, 1]])

def Eijk_custom(p1, p2, p3):
    if (p1 < p2) and (p2 < p3):    # (1, 2, 3)
        return 1
    elif (p1<p3) and (p3<p2):    # (1, 3, 2)
        return -1
    elif (p2<p1) and (p1<p3):    # (2, 1, 3)
        return -1
    elif (p3<p1) and (p1<p2):    # (2, 3, 1)
        return 1
    elif (p2<p3) and (p3<p1):    # (3, 1, 2)
        return 1
    elif (p3<p2) and (p2<p1):    # (3, 2, 1)
        return -1



# (1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), and (3, 2, 1)

for nodes_on_face in nodes:
    sign = np.sign(Eijk(nodes_on_face[0], nodes_on_face[1], nodes_on_face[2]))
    print(sign)
print()
for nodes_on_face in nodes:
    sign = np.sign(Eijk_custom(nodes_on_face[0], nodes_on_face[1], nodes_on_face[2]))
    print(sign)
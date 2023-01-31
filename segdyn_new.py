import numpy as np
import copy
from rigidbody import RigidBody

def segdyn(state: (list[RigidBody], ((float, float), (float, float))), V, Acons=None, Bcons=None):
    bodies, (base_pos, base_vel) = state
    n = len(bodies)
    unknowns = [i for i in range(len(V)) if V[i] is None]
    knowns = [i for i in range(len(V)) if V[i] is not None]
    assert len(V) == 7*n+5
    assert len(unknowns) == 3*n
    # Generate A_star
    A_star = np.zeros((3 * n, 7 * n + 5))
    # Generate B
    b_star = np.zeros((3 * n,))
    for i, rb in enumerate(bodies):
        col_index = 0
        # Fx
        A_star[i, col_index+i:col_index+i + 2] = [1, -1]
        A_star[i+2*n,col_index+i:col_index+i+2] = [rb.d*rb.sin(), (rb.L-rb.d)*rb.sin()]
        col_index += n+1
        # Fy
        A_star[i + n, col_index+i:col_index+i+2] = [1, -1]
        A_star[i+2*n, col_index+i:col_index+i+2] = [-rb.d*rb.cos(), -(rb.L-rb.d)*rb.cos()]
        col_index += n+1
        # M
        A_star[i+2*n, col_index+i:col_index+i+2] = [1, -1]
        col_index += n+1
        # Fx,ext
        A_star[i, col_index+i] = 1
        col_index += n
        # Fy,ext
        A_star[i+n, col_index+i] = 1
        col_index += n
        # Mext
        A_star[i+2*n, col_index+i] = 1
        col_index += n
        # phidd
        A_star[i, col_index+i] = rb.m*rb.d*rb.sin()
        A_star[i+n, col_index+i] = -rb.m*rb.d*rb.cos()
        A_star[i+2*n, col_index+i] = -rb.J
        col_index += n
        # xbdd
        A_star[i, col_index] = -rb.m
        col_index += 1
        # ybdd
        A_star[i+n, col_index] = -rb.m
        # Fill in b_star
        b_star[i] = -rb.m*rb.d*rb.cos()*rb.phid**2
        b_star[n+i] = -rb.m*rb.d*rb.sin()*rb.phid**2
    print(A_star)
    print(b_star)
    # TODO: Add constraints
    A = A_star[:,unknowns]
    b = b_star - A_star[:,knowns]@[V[i] for i in knowns]
    print(b)
    x = np.linalg.solve(A, b)
    Vnew = V.copy()
    for i, u in enumerate(unknowns):
        Vnew[u] = x[i]
    bodiesd = []
    base_acc = (Vnew[-2], Vnew[-1])
    for (rb, phidd) in zip(bodies, Vnew[6*n+3:7*n+3]):
        phid = rb.phid
        phidd = phidd
        bodiesd.append((phid, phidd))
    return (bodiesd, (base_vel, base_acc)), Vnew



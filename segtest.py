import segdyn_new
from rigidbody import RigidBody
import scipy.integrate as integrate

def test_shell(t, state):
    V=[0, 0, 0, 0, 0, 0, 0, -9.81, 0, None, None, None]
    stated, Vnew = segdyn_new.segdyn(state, V)
    print(stated, Vnew)
    return stated, Vnew


if __name__ == "__main__":
    stick = RigidBody(1, 1, 0.5, 1)
    base = ((0, 0), (0, 0))
    state = [[stick], base]
    V = [0, 0, 0, 0, 0, 0, 0, -9.81, 0, None, None, None]
    stated, Vnew = segdyn_new.segdyn(state, V)
    print(stated)
    print(Vnew)
    # sol = integrate.solve_ivp(test_shell, t_span, state)



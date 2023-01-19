import segdyn_new
from rigidbody import RigidBody
import scipy.integrate as integrate

def test_shell(t, state):
    V=[0, 0, 0, 0, 0, 0, 0, -9.81, 0, None, None, None]
    stated, Vnew = segdyn_new.segdyn(state, V)
    return stated, Vnew


if __name__ == "__main__":
    stick = RigidBody(1, 1, 0.5, 1)
    base = ((0, 0), (0, 0))
    state = [stick, base]
    t_span = [0, 4]
    sol = integrate.solve_ivp(test_shell, t_span, state)



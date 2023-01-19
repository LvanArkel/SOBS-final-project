import numpy as np
import scipy.integrate as integrate
from rigidbody import RigidBody
import matplotlib.pyplot as plt
import segdyn_new
from segdyn import *

# Global Settings
g = -9.81

# Settings for the base
base_pos = [0, 0]
base_vel = [0, 0]

# Settings for the rope
rope_length = 4  # m
rope_segments = 4
rope_angle = -0.25 * np.pi  # rad from vertical
rope_density = 0.15  # kg/m

# Settings for the body
body_mass = np.array([10, 43, 14, 10])  # kg
body_length = np.array([0.7, 0.7, 0.5, 0.4])  # m
body_com = np.array([0.45, 0.5, 0.2, 0.25])  # m
body_inert = np.array([0.5, 2.5, 0.35, 0.17])  # kg m2

arms = 0
torso = 1
upper_leg = 2
lower_leg = 3

def make_rope() -> list[RigidBody]:
    segment_length = rope_length / rope_segments
    rope = []
    for _ in range(rope_segments):
        m = segment_length * rope_density
        L = segment_length
        d = segment_length / 2
        J = m * L ** 2 / 12
        segment = RigidBody(m, L, d, J)
        segment.phi = rope_angle/rope_segments
        rope.append(segment)
    rope[0].phi -= 0.5*np.pi
    return rope


def make_body() -> list[RigidBody]:
    body = []
    for i in range(len(body_mass)):
        segment = RigidBody(
            body_mass[i], body_length[i], body_com[i], body_inert[i])
        body.append(segment)
    body[0].phi = -rope_angle
    return body


def get_state(system) -> list[float]:
    phis = []
    phids = []
    for i, rb in enumerate(system):
        phis.append(rb.phi)
        phids.append(rb.phid)
    return phis + phids + base_pos + base_vel


def get_segparms(system):
    nseg = len(system)
    m = np.array([])
    L = np.array([])
    J = np.array([])
    d = np.array([])
    for rb in system:
        m = np.append(m, rb.m)
        L = np.append(L, rb.L)
        J = np.append(J, rb.J)
        d = np.append(d, rb.d)

    segparms = {'nseg': nseg,  # number of segments
                'm': m,  # mass of each segment [kg]
                'L': L,  # length of each segment [m]
                'd': d,  # distance of COM of segment from proximal joint [m]
                'J': J,  # moment of inertia about COM of segment [kgm**2]
                'g': g}  # gravitational acceleration [m/s**2]

    return segparms


def plot_single_state(system: list[RigidBody]):
    x = 0.0
    y = 0.0
    angle = 0.0
    xs = [x]
    ys = [y]
    for rb in system:
        angle += rb.phi
        x += np.cos(angle) * rb.L
        y += np.sin(angle) * rb.L
        xs.append(x)
        ys.append(y)
    plt.plot(xs, ys)
    plt.show()


def readsegparms(segparms):
    nseg = segparms['nseg']
    m = segparms['m']
    L = segparms['L']
    J = segparms['J']
    d = segparms['d']
    g = segparms['g']
    return nseg, m, L, J, d, g


def readsegdynstate(segdynstate, nseg):
    phis = segdynstate[0:nseg]
    phids = segdynstate[nseg:2 * nseg]
    base_pos = segdynstate[2 * nseg:2 * nseg + 2]
    base_vel = segdynstate[2 * nseg + 2:2 * nseg + 4]
    return phis, phids, base_pos, base_vel


def swingparms(system):
    segparms = get_segparms(system)
    swingparms = {
        'segparms': segparms
    }
    return swingparms

def swingstate(system):
    segdynstate = get_state(system)
    state = np.copy(segdynstate)

    return state

def swingshell(t, state, parms):
    segparms = parms['segparms']
    nseg, m, L, J, d, g = readsegparms(segparms)

    segdynstate = state[0: 2 * nseg + 4]
    phis, phids, base_pos, base_vel = readsegdynstate(segdynstate, nseg)

    # V 7 * nseg + 5
    Fx = np.concatenate((np.full(nseg, np.nan), np.array([0])))        # Fx nseg + 1,
    Fy = np.concatenate((np.full(nseg, np.nan), np.array([0])))       # Fy nseg + 1,
    M = np.zeros(nseg + 1)        # M nseg + 1,
    Fextx = np.zeros(nseg)    # Fextx nseg,
    Fexty = m * g           # Fexty nseg,
    Mext = np.zeros(nseg)   # Mext nseg,
    phidd = np.full(nseg, np.nan)    # phidd nseg,
    base_acc = [0, 0]       # xbdd, ybdd
    V = np.concatenate((Fx, Fy, M, Fextx, Fexty, Mext, phidd, base_acc))

    segdynstated, Vnew = segdyn(segdynstate, segparms, V)

    stated = np.copy(segdynstated)
    output = np.copy(Vnew)

    return stated, output


def swing(system):
    initial_state = get_state(system)
    parms = swingparms(system)

    print(parms)

    t_span = [0, 3]
    ODE = lambda t, state: swingshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    output = animate(sol.t, sol.y, parms['segparms'])


if __name__ == "__main__":
    x = 0
    rope = make_rope()
    body = make_body()
    system = rope + body

    swing(system)


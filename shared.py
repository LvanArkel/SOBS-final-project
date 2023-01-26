import numpy as np
from rigidbody import RigidBody
import matplotlib.pyplot as plt
from segdyn import *


# Global Settings
g = -9.81

# Settings for the rope
rope_length = 2  # m
rope_segments = 4
rope_angle = -0.6 * np.pi  # rad from vertical
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
        segment.phi = rope_angle
        rope.append(segment)
    return rope


def make_body() -> list[RigidBody]:
    body = []
    for i in range(len(body_mass)):
        segment = RigidBody(
            body_mass[i], body_length[i], body_com[i], body_inert[i], phi=rope_angle)
        body.append(segment)
    return body

def get_state(system, base_pos, base_vel) -> list[float]:
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

def plot_single_state(system: list[RigidBody], base_pos):
    x = base_pos[0]
    y = base_pos[1]
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

def plot_energies(segdynstate, segparms, time):
    Ekinx, Ekiny, Erot, Epot, Etot = energy(segdynstate, segparms)
    # plt.plot(time, np.add(Ekinx, Ekiny, Erot))
    # plt.plot(time, Epot)
    plt.plot(time, Etot)

def plot_feet_y(segdynstate, segparms, time):
    joint, *_ = jointcoord(segdynstate, segparms)
    _, jointy = joint
    plt.plot(time, jointy[-1, :])
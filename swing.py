import numpy as np
from rigidbody import RigidBody
import matplotlib.pyplot as plt
import segdyn_new

# Global Settings
g = -9.81

# Settings for the base
base_pos = [0, 0]
base_vel = [0, 0]

# Settings for the rope
rope_length = 4  # m
rope_segments = 10
rope_angle = -0.25 * np.pi  # rad from vertical
rope_density = 0.15  # kg/m

# Settings for the body
body_mass = [10, 43, 14, 10]  # kg
body_length = [0.7, 0.7, 0.5, 0.4]  # m
body_com = [0.45, 0.5, 0.2, 0.25]  # m
body_inert = [0.5, 2.5, 0.35, 0.17]  # kg m2

arms = 0
torso = 1
upper_leg = 2
lower_leg = 3

segparms = {'nseg': rope_segments + len(body_mass),  # number of segments
            'm': body_mass,  # mass of each segment [kg]
            'L': body_length,  # length of each segment [m]
            'd': body_com,  # distance of COM of segment from proximal joint [m]
            'J': body_inert,  # moment of inertia about COM of segment [kgm**2]
            'g': g}  # gravitational acceleration [m/s**2]


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


if __name__ == "__main__":
    x = 0
    rope = make_rope()
    body = make_body()
    system = rope + body
    plot_single_state(system)


    segdyn_new.segdyn(system, )

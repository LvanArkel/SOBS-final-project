import numpy as np
from rigidbody import RigidBody

# Settings for the rope
rope_length = 4  # m
rope_segments = 1
rope_angle = 0.75 * np.pi  # rad
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


def make_rope() -> list[RigidBody]:
    segment_length = rope_length / rope_segments
    rope = []
    for _ in range(rope_segments):
        m = segment_length * rope_density
        L = segment_length
        d = segment_length / 2
        J = m * L ** 2 / 12
        segment = RigidBody(m, L, d, J)
        rope.append(segment)
    rope[0].set_angle(rope_angle)
    return rope


def make_body() -> list[RigidBody]:
    body = []
    for i in range(len(body_mass)):
        segment = RigidBody(
            body_mass[i], body_length[i], body_com[i], body_inert[i])
        body.append(segment)
    return body


if __name__ == "__main__":
    x = 0
    rope = make_rope()
    body = make_body()
    system = rope + body
    n_segments = rope_segments + len(body_mass)







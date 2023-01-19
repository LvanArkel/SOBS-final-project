import numpy as np


class Force:
    def __init__(self, x, y, offset):
        self.x = x
        self.y = y
        self.offset = offset


class RigidBody:
    def __init__(self, m, L, d, J, phi=0, phid=0, forces = [], gravity = True):
        self.m = m
        self.L = L
        self.d = d
        self.J = J
        self.phi = phi
        self.phid = phid
        self.forces = forces
        if gravity:
            forces.append(Force(0, -9.81, 0))

    def sin(self) -> float:
        return np.sin(self.phi)

    def cos(self) -> float:
        return np.cos(self.phi)

    def phidsqr(self) -> float:
        return np.square(self.phid)

    def get_center(self) -> list[float]:
        return self.d * np.array([self.cos(), self.sin()])





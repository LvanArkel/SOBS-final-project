import numpy as np

class RigidBody:
    def __init__(self, m, L, d, J, phi=0, phid=0):
        self.m = m
        self.L = L
        self.d = d
        self.J = J
        self.phi = phi
        self.phid = phid

    def sin(self) -> float:
        return np.sin(self.phi)

    def cos(self) -> float:
        return np.cos(self.phi)

    def phidsqr(self) -> float:
        return np.square(self.phid)

    def get_center(self) -> list[float]:
        return self.d * np.array([self.cos(), self.sin()])


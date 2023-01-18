class RigidBody:
    def __init__(self, m, L, d, J):
        self.m = m,
        self.L = L,
        self.d = d,
        self.J = J,
        self.state = [0, 0]

    def angle(self) -> int:
        return self.state[0]

    def set_angle(self, angle):
        self.state[0] = angle

    def ang_vel(self) -> int:
        return self.state[1]

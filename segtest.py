import segdyn_new
from rigidbody import RigidBody

if __name__ == "__main__":
    stick = RigidBody(1, 1, 0.5, 1)
    V = [0, 0, 0, 0, 0, 0, 0, -9.81, 0, None, None, None]
    print(segdyn_new.segdyn(([stick], [0, 0]), V))

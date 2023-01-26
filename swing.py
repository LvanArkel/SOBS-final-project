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
    rope_stiffness = 50
    rope_damping = 10
    frequency = 1.25
    hip_moment_mult = 150
    shoulder_moment_mult = 0.7
    swingparms = {
        'segparms': segparms,
        'rope_stiffness': rope_stiffness,
        'rope_damping': rope_damping,
        'frequency': frequency,
        'hip_moment_mult': hip_moment_mult,
        'shoulder_moment_mult': shoulder_moment_mult
    }
    return swingparms


def swingstate(system):
    segdynstate = get_state(system)
    # Ms = np.zeros(rope_segments)
    state = segdynstate
    return state


def swingshell(t, state, parms):
    segparms = parms['segparms']
    nseg, m, L, J, d, g = readsegparms(segparms)
    rope_stiffness = parms['rope_stiffness']
    rope_damping = parms['rope_damping']
    frequency = parms['frequency']
    hip_moment_mult = parms['hip_moment_mult']

    segdynstate = state[0: 2 * nseg + 4]
    phis, phids, base_pos, base_vel = readsegdynstate(segdynstate, nseg)
    # Ms = state[2 * nseg + 4: 2 * nseg + 4 + rope_segments]

    Ms = np.array([])
    for index, rb in enumerate(system):
        # if index < rope_segments:
        phi = - 0.5 * np.pi - phis[index]
        phid = phids[index]
        if index != 0:
            phi = phis[index - 1] - phis[index]
            phid = phids[index - 1] - phids[index]

        stiff = rope_stiffness * phi
        damp = rope_damping * phid
        moment = stiff + damp
        Ms = np.append(Ms, moment)

    rope_ang = phis[0] + 0.5 * np.pi
    rope_vel = phids[0]
    hip_moment = np.sin(frequency * t) * hip_moment_mult
    # hip_moment = rope_ang * hip_moment_mult
    # hip_moment = rope_vel * hip_moment_mult

    Ms[-2] += hip_moment
    # Ms[-3] += shoulder_moment
    # print(f"Moment {hip_moment}, Rope Ang {rope_ang}, Rope Acc {rope_acc}")


    # V 7 * nseg + 5
    Fx = np.concatenate((np.full(nseg, np.nan), np.array([0])))        # Fx nseg + 1,
    Fy = np.concatenate((np.full(nseg, np.nan), np.array([0])))       # Fy nseg + 1,
    M = np.concatenate((Ms, np.zeros(1)))        # M nseg + 1,
    Fextx = np.zeros(nseg)    # Fextx nseg,
    Fexty = m * g           # Fexty nseg,
    Mext = np.zeros(nseg)   # Mext nseg,
    phidd = np.full(nseg, np.nan)    # phidd nseg,
    base_acc = [0, 0]       # xbdd, ybdd

    V = np.concatenate((Fx, Fy, M, Fextx, Fexty, Mext, phidd, base_acc))
    # print(f"Length V: {len(V)}, should be: {7 * nseg + 5}")

    segdynstated, Vnew = segdyn(segdynstate, segparms, V)

    stated = np.copy(segdynstated)
    output = np.copy(Vnew)

    return stated, output


def swing(system):
    initial_state = swingstate(system)
    parms = swingparms(system)
    print(parms)

    t_span = [0, 30]
    ODE = lambda t, state: swingshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    plot_energies(segdynstate, segparms, sol.t)

    return sol, segparms


def plot_energies(segdynstate, segparms, time):
    Ekinx, Ekiny, Erot, Epot, Etot = energy(segdynstate, segparms)
    # plt.plot(time, np.add(Ekinx, Ekiny, Erot))
    # plt.plot(time, Epot)
    plt.plot(time, Etot)

def our_animate(t,segdynstate,segparms,axlim=2):
    # nr of frames and time interval, such that total simulation time is accurate
    # interpolate data to match framerate of animator
    time_interval = 0.05  # 50 milliseconds for each frame
    nseg = segparms['nseg']
    nr_frames = np.ceil((t[-1] - t[0]) / time_interval).astype(int) + 1
    t_new = np.linspace(0, t[-1], num=nr_frames)
    segdynstate_new = np.zeros((2 * nseg + 4, nr_frames))
    for i in range(2 * nseg + 4):
        segdynstate_new[i, :] = np.interp(t_new, t, segdynstate[i])

    # calculate joint coordinates
    joint, *_ = jointcoord(segdynstate_new, segparms)
    jointx, jointy = joint

    # determine range of image?? / user defined initial frame??

    # initiate figure
    fig = plt.figure()
    ax = plt.axes(xlim=(-axlim, axlim),
                  ylim=(-axlim, axlim))
    ax.set_aspect('equal', adjustable='box')
    line_rope, = ax.plot([], [], lw=2)
    line_body, = ax.plot([], [], lw=2)

    # initiate line
    def init_fig():
        line_rope.set_data([], [])
        line_body.set_data([], [])
        return [line_rope, line_body]

    def animate(i, jointx, jointy):
        # appending values to the previously
        # empty x and y data holders
        xdata = jointx[:, i]
        ydata = jointy[:, i]
        line_rope.set_data(xdata[:-4], ydata[:-4])
        line_body.set_data(xdata[-5:], ydata[-5:])
        return [line_rope, line_body]

    # calling the animation function
    anim = animation.FuncAnimation(fig, animate, init_func=init_fig,
                                   fargs=(jointx, jointy),
                                   frames=nr_frames,
                                   interval=time_interval * 1000,
                                   blit=True)
    # make sure plot renders and shows
    plt.draw()
    plt.show()

    # save the animation somewhere. To be implemented later ??
    # anim.save('whateverName.mp4', writer = 'ffmpeg', fps = 1/time_interval/1000??)

    return anim  # NECESARRY to return!!


if __name__ == "__main__":
    x = 0
    rope = make_rope()
    body = make_body()
    system = rope + body

    # plot_single_state(system)

    sol, segparms = swing(system)

    # state0 = sol.y[0]
    # phis, phids, base_pos, base_vel = readsegdynstate(state0, len(system))
    # print(f"Phis: {phis}")

    our_animate(sol.t, sol.y, segparms, axlim=7)


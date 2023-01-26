from segdyn import *
from shared import *


# Settings for the base
base_pos = [0, 0]
base_vel = [0, 0]


def swingparms(system):
    segparms = get_segparms(system)
    rope_stiffness = 50
    rope_damping = 10
    frequency = 1.6675
    hip_moment_mult = 150
    shoulder_moment_mult = 30
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
    segdynstate = [-1.94036625, -1.99794535, -2.00456897, -2.00666025, -2.01304106, -1.94749094,
 -3.39876688, -3.21799289,  0.50835549,  0.55571912,  0.56852215,  0.57119324,
  0.54845258,  0.26056143,  0.65145562,  0.12220817,  0.,          0.,
  0.,          0.        ]

    # segdynstate = get_state(system)
    # Ms = np.zeros(rope_segments)
    # segdynstate = get_state(system, base_pos, base_vel)
    # segdynstate = [-2.0541526, -2.09858616, -2.09903548, -2.085879, -2.13957014, -1.26554274,
    #  -1.4586276,   0.89461135,  0.89946407,  0.87659005,  0.94474495,  1.47547233,
    #  3.44077558,  3.44315349,  0,          0,          0,          0]
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
    shoulder_moment_mult = parms['shoulder_moment_mult']

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

    t_span = [0, 1]
    ODE = lambda t, state: swingshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    plt.figure()
    plot_energies(segdynstate, segparms, sol.t)
    plt.figure()
    plot_feet_y(segdynstate, segparms, sol.t)

    print(f"Last State: {segdynstate[:, -1]}")


    print(segdynstate[:, -1])


    return sol, segparms

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


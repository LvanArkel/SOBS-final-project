from shared import *



def flightparms(system):
    segparms = get_segparms(system)
    stiffness = 50
    damping = 10
    frequency = 1.7
    hip_moment_mult = 150
    shoulder_moment_mult = 0.7
    swingparms = {
        'segparms': segparms,
        'stiffness': stiffness,
        'damping': damping,
        'hip_moment_mult': hip_moment_mult,
        'shoulder_moment_mult': shoulder_moment_mult
    }
    return swingparms


def flightstate(system):
    segdynstate = get_state(system)
    # segdynstate = [-2.0541526, -2.09858616, -2.09903548, -2.085879, -2.13957014, -1.26554274,
    #  -1.4586276,   0.89461135,  0.89946407,  0.87659005,  0.94474495,  1.47547233,
    #  3.44077558,  3.44315349,  0,          0,          0,          0]
     # Ms = np.zeros(rope_segments)
    state = segdynstate
    return state


def flightshell(t, state, parms):
    segparms = parms['segparms']
    nseg, m, L, J, d, g = readsegparms(segparms)
    stiffness = parms['stiffness']
    damping = parms['damping']
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

        stiff = stiffness * phi
        damp = damping * phid
        moment = stiff + damp
        Ms = np.append(Ms, moment)

    rope_ang = phis[0] + 0.5 * np.pi
    rope_vel = phids[0]
    # hip_moment = np.sin(frequency * t) * hip_moment_mult
    # hip_moment = rope_ang * hip_moment_mult
    # hip_moment = rope_vel * hip_moment_mult

    # Ms[-2] += hip_moment
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


def flight(system):
    initial_state = flightstate(system)
    parms = flightparms(system)
    print(parms)

    t_span = [45, 55]
    ODE = lambda t, state: flightshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    plt.figure()
    plot_energies(segdynstate, segparms, sol.t)
    plt.figure()
    plot_feet_y(segdynstate, segparms, sol.t)

    print(f"Last State: {segdynstate[:, -1]}")

    return sol, segparms

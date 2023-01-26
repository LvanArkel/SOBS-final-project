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
        'hip_moment_mult': hip_moment_mult
    }
    return swingparms


def swingstate(system, base_pos, base_vel):
    segdynstate = get_state(system, base_pos, base_vel)
    segdynstate = [-2.25964921, -2.35331292, -2.36967297, -2.37276211, -2.36394001, -2.19942433,
 -3.40328011, -3.10805272,  0.16127535,  0.04865359,  0.0291509,  -0.00940969,
 -0.13438966, -0.59134242, -1.70180314, -2.32473594,  0,          0,
  0,          0]

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


def plot_angmom(systemstates, solt, parms):
    # systemstates = (nseg, nsamples
    print(f"Systemstates {systemstates}, Len {len(systemstates)}")
    print(f"Parms, {parms}")
    parms = copy.deepcopy(parms)
    resize = ['m', 'L', 'd', 'J']
    for val in resize:
        parms[val] = parms[val][-4:]
    parms['nseg'] = 4
    # print(f"Parms {parms}")
    body_phis = systemstates[4:8]
    body_phids = systemstates[12:16]
    base_pos_vel = systemstates[16:]
    # print(f"Phis {body_phis}, Phids {body_phids}, pos_vel {base_pos_vel}")
    bodystates = np.concatenate((body_phis, body_phids, base_pos_vel))
    # print(f"Body states {len(bodystates)}")

    ang, _ = angmom(bodystates, parms)
    plt.plot(solt, ang)
    idx = np.argmax(ang)
    print(f"Index, {idx}, Time of most ang mom {solt[idx]}")
    # last_y = [[item[idx]] for item in bodystates]
    # print(f"LAst y {last_y}")
    #
    # _, (jointxd, jointyd), _ = jointcoord([last_y], parms)
    # arm_base_vel = [jointxd[4], jointyd[4]]
    # print(f"Arm base Vel {repr(arm_base_vel)}")
    # print(f"State of most ang mom {repr([item[0] for item in last_y])}")



def swing(system, base_pos, base_vel):
    initial_state = swingstate(system, base_pos, base_vel)
    parms = swingparms(system)
    print(parms)

    t_span = [120, 124.57723045423722]
    ODE = lambda t, state: swingshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    plt.figure()
    plot_energies(segdynstate, segparms, sol.t)
    plt.figure()
    plot_feet_y(segdynstate, segparms, sol.t)

    last = segdynstate[:, -1]
    print(f"Last State: {repr(last)}")
    last_y = [[item[-1]] for item in sol.y]
    _, (jointxd, jointyd), _ = jointcoord(last_y, parms['segparms'])
    arm_base_vel = [jointxd[4], jointyd[4]]
    print(f"Arm base Vel {repr(arm_base_vel)}")

    plt.figure()
    plot_angmom(sol.y, sol.t, parms['segparms'])

    return sol, segparms

if __name__ == "__main__":
    x = 0
    base_pos = [0, 0]
    base_vel = [0, 0]
    rope = make_rope()
    body = make_body()
    system = rope + body

    # plot_single_state(system)

    sol, segparms = swing(system, base_pos, base_vel)

    # state0 = sol.y[0]
    # phis, phids, base_pos, base_vel = readsegdynstate(state0, len(system))
    # print(f"Phis: {phis}")



    our_animate(sol.t, sol.y, segparms, axlim=7)


import numpy as np

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
    swingstate = [-1.93631052, -1.9427537,  -1.94559113, -1.94372898, -1.91598567, -1.72270379,
 -2.94029564, -2.80618342, -1.17536461, -1.32594788, -1.34492584, -1.3792784,
 -1.44323542, -0.9633936,   0.16177069,  1.29321739 , 0.,          0.,
  0.,          0.        ]
    swingbase_vel = [-2.43451612, 0.94845105]
    segdynstate = swingstate_to_flight_state(swingstate, swingbase_vel)
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

    # hip_moment = np.sin(frequency * t) * hip_moment_mult
    # hip_moment = rope_ang * hip_moment_mult
    # hip_moment = rope_vel * hip_moment_mult

    # Ms[-2] += hip_moment
    # Ms[-3] += shoulder_moment
    # print(f"Moment {hip_moment}, Rope Ang {rope_ang}, Rope Acc {rope_acc}")


    # V 7 * nseg + 5
    Fx = np.concatenate( (np.array([0]), np.full(nseg - 1, np.nan), np.array([0])))        # Fx nseg + 1,
    Fy = np.concatenate((np.array([0]), np.full(nseg - 1, np.nan), np.array([0])))       # Fy nseg + 1,
    M = np.concatenate((Ms, np.zeros(1)))   # M nseg + 1,
    Fextx = np.zeros(nseg)    # Fextx nseg,
    Fexty = m * g           # Fexty nseg,
    Mext = np.zeros(nseg)   # Mext nseg,
    phidd = np.full(nseg, np.nan)    # phidd nseg,
    base_acc = [np.nan, np.nan]       # xbdd, ybdd

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

    t_span = [0, 2]
    ODE = lambda t, state: flightshell(t, state, parms)[0]

    sol = integrate.solve_ivp(ODE, t_span, initial_state, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    plt.figure()
    plot_energies(segdynstate, segparms, sol.t)
    plt.figure()
    plot_feet_y(segdynstate, segparms, sol.t)

    print(f"Last State: {repr(segdynstate[:, -1])}")

    return sol, segparms


if __name__ == "__main__":
    x = 0
    base_pos = [0, 6]
    base_vel = [0, 0]
    body = make_body()
    system = body

    # plot_single_state(system)

    sol, segparms = flight(system)

    # state0 = sol.y[0]
    # phis, phids, base_pos, base_vel = readsegdynstate(state0, len(system))
    # print(f"Phis: {phis}")

    our_animate(sol.t, sol.y, segparms, axlim=7)

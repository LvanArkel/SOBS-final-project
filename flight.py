import numpy as np

import segdyn
from shared import *

def flightparms(system):
    segparms = get_segparms(system)
    stiffness = 50
    damping = 10
    frequency = 1.7
    hip_moment_mult = 30
    shoulder_moment_mult = 20
    knee_moment_mult = 30

    knee_angle = -0.5 * np.pi
    shoulder_angle = -0 * np.pi
    hip_angle = -0.5 * np.pi
    swingparms = {
        'segparms': segparms,
        'stiffness': stiffness,
        'damping': damping,
        'hip_moment_mult': hip_moment_mult,
        'shoulder_moment_mult': shoulder_moment_mult,
        'knee_moment_mult': knee_moment_mult,
        'shoulder_angle': shoulder_angle,
        'hip_angle': hip_angle,
        'knee_angle': knee_angle,
    }
    return swingparms




def flightstate(system):
    # LARS
    # swingstate = [-0.9121063 , -0.85812016, -0.85011183, -0.85049494, -0.85832449,
    #    -1.04205367, -0.12453037, -0.51850652,  0.50177495,  0.67365286,
    #     0.7051935 ,  0.68995334,  0.59797652,  0.46833777,  1.26711656,
    #     1.33613255,  0.        ,  8.        ,  0.        ,  0.        ]
    # swingbase_vel = [0.97746026, 0.83402999]
    # swingbase_pos = [1.29272637, 6.47481468]

    # GIES
    swingstate = [-1.86503057, -1.89215125, -1.892467  , -1.89009032, -1.89254894,
       -1.93107123, -1.76578462, -1.7960798 ,  0.57694302,  0.63301128,
        0.62706037,  0.66608912,  0.96329365,  2.06334383,  6.34132209,
        6.88543933,  0.        ,  0.        ,  0.        ,  0.        ]
    swingbase_vel = [1.19003784, -0.3872925]
    swingbase_pos = [0, 11]

    segdynstate = swingstate_to_flight_state(swingstate, swingbase_pos, swingbase_vel)
     # Ms = np.zeros(rope_segments)
    state = segdynstate
    return state


def flightshell(t, state, parms):
    segparms = parms['segparms']
    nseg, m, L, J, d, g = readsegparms(segparms)
    stiffness = parms['stiffness']
    damping = parms['damping']
    hip_moment_mult = parms['hip_moment_mult']
    shoulder_moment_mult = parms['shoulder_moment_mult']
    knee_moment_mul = parms['knee_moment_mult']
    knee_angle = parms['knee_angle']
    shoulder_angle = parms['shoulder_angle']
    hip_angle = parms['hip_angle']

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
    curr_s_angle = phis[1] - phis[0]
    curr_h_angle = phis[2] - phis[1]
    curr_k_angle = phis[3] - phis[2]
    # LARS
    # Ms[1] += (shoulder_angle - current_shoulder_angle) * shoulder_moment_mult
    # Ms[2] += (hip_angle - current_hip_angle) * hip_moment_mult

    # GIES
    shoulder_diff = shoulder_angle - curr_s_angle
    Ms[1] += curr_s_angle * shoulder_moment_mult * 3
    hip_diff = hip_angle - curr_h_angle
    Ms[2] += hip_diff * hip_moment_mult
    knee_diff = knee_angle - curr_k_angle
    Ms[3] += knee_diff * knee_moment_mul
    print(f"Angles Shoulder {curr_s_angle}, Hip {curr_h_angle}, Knee {curr_k_angle}")
    print(f"Angle Diffs Shoulder {shoulder_diff}, Hip {hip_diff}, Knee {knee_diff}")
    # print(f"Moment {hip_moment}, Rope Ang {rope_ang}, Rope Acc {rope_acc}")

    # V 7 * nseg + 5
    Fx = np.concatenate((np.array([0]), np.full(nseg - 1, np.nan), np.array([0])))        # Fx nseg + 1,
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

def collision_event(t, state, parms):
    segparms = parms['segparms']
    (jointx, jointy), *_ = jointcoord(state, segparms)
    print(f"Jointx {jointx},  Jointy, {jointy}")
    criterion = jointy.min()
    print(f"Criterion {criterion}")
    if criterion < 1 and criterion > -1:
        print(criterion)
    return criterion

def gies_collision_event(t, state, parms):
    segparms = parms['segparms']
    (jointx, jointy), *_ = jointcoord(state, segparms)

def flight(system):
    initial_state = flightstate(system)
    parms = flightparms(system)
    print(parms)

    t_span = [0, 2]
    ODE = lambda t, state, parms: flightshell(t, state, parms)[0]
    # collision_event.terminal = True

    sol = integrate.solve_ivp(ODE, t_span, initial_state, args=(parms,), events=collision_event, rtol=1e-8, atol=1e-8)

    segparms = parms['segparms']
    segdynstate = sol.y[0: 2 * segparms['nseg'] + 4]
    (_, jointy), *_ = jointcoord(segdynstate, segparms)
    lowestjoint = np.argmin(jointy[:, -1])
    jointmap = {
        0: "Hands",
        1: "Shoulders",
        2: "Hips",
        3: "Knees",
        4: "Feet"
    }
    print(f"Lowest joint: {jointmap[lowestjoint]}")

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

    our_animate(sol.t, sol.y, segparms, axlim=13)

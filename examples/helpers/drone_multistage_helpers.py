from typing import Tuple, Callable, Sequence
import matplotlib.pyplot as plt
import casadi as c
import numpy as np


def get_rotation_matrix_from_euler(
        orient_eul: np.array) -> Tuple[c.DM, c.DM, c.DM, c.DM]:
    '''From the roll, pitch yaw euler angles (orient_eul), get the rotation
    matrix R and the unit vectors ex, ey, ez (colums of R).
    '''
    [phi, theta, psi] = [orient_eul[0], orient_eul[1], orient_eul[2]]
    cr = c.cos(phi)
    sr = c.sin(phi)
    cp = c.cos(theta)
    sp = c.sin(theta)
    cy = c.cos(psi)
    sy = c.sin(psi)
    # Gate orientation matrix
    ex = c.vertcat(cy*cp, sy*cp, -sp)
    ey = c.vertcat(cy*sp*sr-sy*cr, sy*sp*sr + cy*cr, cp*sr)
    ez = c.vertcat(cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr)
    R = c.horzcat(ex, ey, ez)

    return R, ex, ey, ez


def get_w_approach(pg: c.DM, exg: c.DM, dg: c.DM, margin: float) -> c.DM:
    '''Get the w matrix for an approach stage.
    '''
    n = exg
    s = pg - (dg/2 + margin) * n
    w = c.vertcat(n, - n.T @ s)

    return w


def get_w_through(pg: c.DM, eyg: c.DM, ezg: c.DM,
                  hg: float, wg: float, margin: float) -> c.DM:
    '''Get the w matrix for a through stage.
    '''
    nu = ezg
    su = pg + (hg/2 - margin) * nu
    wu = c.vertcat(nu, -nu.T @ su)

    nd = -nu
    sd = pg + (hg/2 - margin) * nd
    wd = c.vertcat(nd, -nd.T @ sd)

    nl = eyg
    sl = pg + (wg/2 - margin) * nl
    wl = c.vertcat(nl, -nl.T @ sl)

    nr = - nl
    sr = pg + (wg/2 - margin) * nr
    wr = c.vertcat(nr, -nr.T @ sr)

    w = c.horzcat(wu, wd, wl, wr)

    return w


def get_w_flyaway_approach(pg1: c.DM, exg1: c.DM, dg1: float,
                           pg2: c.DM, exg2: c.DM, dg2: float,
                           margin: float) -> c.DM:
    '''Get the w matrix for a fly away stage that is also the approach stage
    for a next gate.
    '''
    n = -exg1
    s = pg1 - (dg1/2 + margin) * n
    w = c.vertcat(n, - n.T @ s)

    nbis = exg2
    sbis = pg2 - (dg2/2 + margin) * nbis
    wbis = c.vertcat(nbis, - nbis.T @ sbis)

    wtot = c.horzcat(w, wbis)

    return wtot


def get_w_flyaway(pg: c.DM, exg: c.DM, dg: float, margin: float) -> c.DM:
    '''Get the w matrix for a fly away stage.
    '''
    n = -exg
    s = pg - (dg/2 + margin) * n
    w = c.vertcat(n, - n.T @ s)

    return w


def get_sol_ctrl_grid(gist: c.DM,
                      samplers: Tuple[(Callable,)*5],
                      Ts, T) -> Tuple[(np.ndarray,)*6]:
    '''Get the OCP solution sampled with control sampling time Ts.
    '''
    (T1, T2, T3, T4, T5) = T
    (sampler1, sampler2, sampler3, sampler4, sampler5) = samplers

    psol_ctrl = []
    vsol_ctrl = []
    atsol_ctrl = []
    eulsol_ctrl = []
    Rsol_ctrl = []
    t_ctrl = []

    # Handle empty stages (Ti < Ts) & add solutions together.
    t1_ctrl = np.arange(0., T1, Ts)
    if len(t1_ctrl) < 1:
        t1_f = 0
    else:
        [p1sol_ctrl, v1sol_ctrl,
         at1sol_ctrl, eul1sol_ctrl, R1sol_ctrl] = sampler1(gist, t1_ctrl)
        psol_ctrl.append(p1sol_ctrl)
        vsol_ctrl.append(v1sol_ctrl)
        atsol_ctrl.append(at1sol_ctrl)
        eulsol_ctrl.append(eul1sol_ctrl)
        Rsol_ctrl.append(R1sol_ctrl)
        t_ctrl.append(t1_ctrl)
        t1_f = t1_ctrl[-1]
    t2_ctrl = np.arange(t1_f + Ts, T2, Ts)
    if len(t2_ctrl) < 1:
        t2_f = t1_f
    else:
        [p2sol_ctrl, v2sol_ctrl,
         at2sol_ctrl, eul2sol_ctrl, R2sol_ctrl] = sampler2(gist, t2_ctrl)
        psol_ctrl.append(p2sol_ctrl)
        vsol_ctrl.append(v2sol_ctrl)
        atsol_ctrl.append(at2sol_ctrl)
        eulsol_ctrl.append(eul2sol_ctrl)
        Rsol_ctrl.append(R2sol_ctrl)
        t_ctrl.append(t2_ctrl)

        t2_f = t2_ctrl[-1]
    t3_ctrl = np.arange(t2_f + Ts, T3, Ts)
    if len(t3_ctrl) < 1:
        t3_f = t2_f
    else:
        [p3sol_ctrl, v3sol_ctrl,
         at3sol_ctrl, eul3sol_ctrl, R3sol_ctrl] = sampler3(gist, t3_ctrl)
        psol_ctrl.append(p3sol_ctrl)
        vsol_ctrl.append(v3sol_ctrl)
        atsol_ctrl.append(at3sol_ctrl)
        eulsol_ctrl.append(eul3sol_ctrl)
        Rsol_ctrl.append(R3sol_ctrl)
        t_ctrl.append(t3_ctrl)
        t3_f = t3_ctrl[-1]
    t4_ctrl = np.arange(t3_f + Ts, T4, Ts)
    if len(t4_ctrl) < 1:
        t4_f = t3_f
    else:
        [p4sol_ctrl, v4sol_ctrl,
         at4sol_ctrl, eul4sol_ctrl, R4sol_ctrl] = sampler4(gist, t4_ctrl)
        psol_ctrl.append(p4sol_ctrl)
        vsol_ctrl.append(v4sol_ctrl)
        atsol_ctrl.append(at4sol_ctrl)
        eulsol_ctrl.append(eul4sol_ctrl)
        Rsol_ctrl.append(R4sol_ctrl)
        t_ctrl.append(t4_ctrl)
        t4_f = t4_ctrl[-1]
    t5_ctrl = np.arange(t4_f + Ts, T5, Ts)
    if len(t5_ctrl) > 1:
        [p5sol_ctrl, v5sol_ctrl,
         at5sol_ctrl, eul5sol_ctrl, R5sol_ctrl] = sampler5(gist, t5_ctrl)
        psol_ctrl.append(p5sol_ctrl)
        vsol_ctrl.append(v5sol_ctrl)
        atsol_ctrl.append(at5sol_ctrl)
        eulsol_ctrl.append(eul5sol_ctrl)
        Rsol_ctrl.append(R5sol_ctrl)
        t_ctrl.append(t5_ctrl)

    # Concatenate
    psol_ctrl = np.concatenate(psol_ctrl)
    vsol_ctrl = np.concatenate(vsol_ctrl)
    atsol_ctrl = np.concatenate(atsol_ctrl)
    eulsol_ctrl = np.concatenate(eulsol_ctrl)
    Rsol_ctrl = np.concatenate(Rsol_ctrl)
    t_ctrl = np.concatenate(t_ctrl)

    return (psol_ctrl, vsol_ctrl, atsol_ctrl, eulsol_ctrl, Rsol_ctrl, t_ctrl)


def get_sol_OCP_grid(gist: c.DM, samplers: Tuple[(Callable,)*5],
                     T: Tuple[(float,)*5], N: Tuple[(int,)*5]):
    '''Get the OCP solution on the OCP control grid.
    '''
    (T1, T2, T3, T4, T5) = T
    (N1, N2, N3, N4, N5) = N
    (sampler1, sampler2, sampler3, sampler4, sampler5) = samplers

    t1 = np.linspace(0., T1, N1)
    t2 = np.linspace(T1, T2, N2)
    t3 = np.linspace(T2, T3, N3)
    t4 = np.linspace(T3, T4, N4)
    t5 = np.linspace(T4, T5, N5)

    [p1sol, v1sol, at1sol, eul1sol, R1sol] = sampler1(gist, t1)
    [p2sol, v2sol, at2sol, eul2sol, R2sol] = sampler2(gist, t2)
    [p3sol, v3sol, at3sol, eul3sol, R3sol] = sampler3(gist, t3)
    [p4sol, v4sol, at4sol, eul4sol, R4sol] = sampler4(gist, t4)
    [p5sol, v5sol, at5sol, eul5sol, R5sol] = sampler5(gist, t5)

    sol = ([p1sol, v1sol, at1sol, eul1sol, R1sol],
           [p2sol, v2sol, at2sol, eul2sol, R2sol],
           [p3sol, v3sol, at3sol, eul3sol, R3sol],
           [p4sol, v4sol, at4sol, eul4sol, R4sol],
           [p5sol, v5sol, at5sol, eul5sol, R5sol])

    psol = np.concatenate([p1sol, p2sol, p3sol, p4sol, p5sol])
    vsol = np.concatenate([v1sol, v2sol, v3sol, v4sol, v5sol])
    atsol = np.concatenate([at1sol, at2sol, at3sol, at4sol, at5sol])
    eulsol = np.concatenate([eul1sol, eul2sol, eul3sol, eul4sol, eul5sol])
    Rsol = np.concatenate([R1sol, R2sol, R3sol, R4sol, R5sol])
    t = np.concatenate([t1, t2, t3, t4, t5])

    return (psol, vsol, atsol, eulsol, Rsol, t, sol)


############
# Plotting #
############
blue = [0.0, 0.47, 0.66]
red = [0.6350, 0.0780, 0.1840]
yellow = [0.9290, 0.6940, 0.1250]


def plot_quad3d(x: list, y: list, z: list, R: list,
                ax, axlim: list, step: int = 1) -> None:
    '''Plot simple quadrotors along the solution trajectory. One plotted quad
    for every 'step' samples.
    '''
    # Quad dimensions
    df = 0.18
    dp = 0.1778/2
    hp = 0.05
    # Quad geometry drawing
    framepoints1 = [np.array([-df, 0, hp]),
                    np.array([-df, 0, 0]), np.array([df, 0, 0]),
                    np.array([df, 0, hp])]
    framepoints2 = [np.array([0, -df, hp]),
                    np.array([0, -df, 0]), np.array([0, df, 0]),
                    np.array([0, df, hp])]
    framepoints = [framepoints1, framepoints2]

    th = np.arange(0, 2*np.pi + 0.1, 0.1)
    rotorpoints = [np.array([dp*np.cos(th[i]), dp*np.sin(th[i]), hp])
                   for i in range(len(th))]
    rotor1points = [point + np.array([df, 0, 0]) for point in rotorpoints]
    rotor2points = [point + np.array([0, df, 0]) for point in rotorpoints]
    rotor3points = [point + np.array([-df, 0, 0]) for point in rotorpoints]
    rotor4points = [point + np.array([0, -df, 0]) for point in rotorpoints]
    rotorpoints = [rotor1points, rotor2points, rotor3points, rotor4points]

    x = x[::step]
    y = y[::step]
    z = z[::step]
    R = R[::step]

    for i in range(len(x)):
        frames = [[], []]
        rotors = [[], [], [], []]
        # Frame is drawn in '+' configuration, rotate 45deg for
        # 'x' configuration.
        R45 = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],
                        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
                        [0, 0, 1]])
        for j in range(len(frames)):
            for point in framepoints[j]:
                point = np.matmul(R45, point)
                frames[j].append(
                    np.matmul(R[i], point) + np.array([x[i], y[i], z[i]]))

            a, b, c = zip(*frames[j])
            ax.plot(a, b, c, color=red, linewidth=1)

        for j in range(len(rotors)):
            for point in rotorpoints[j]:
                point = np.matmul(R45, point)
                rotors[j].append(
                    np.matmul(R[i], point) + np.array([x[i], y[i], z[i]]))
            a, b, c = zip(*rotors[j])
            ax.plot(a, b, c, color=red, linewidth=1)

        ax.set_xlim(axlim[0:2])
        ax.set_ylim(axlim[2:4])
        ax.set_zlim(axlim[4:6])


def plot_gate(pg: c.DM, Rg: c.DM, wg: float, hg: float, dg: float,
              ax: plt.Axes, alpha: float = 1) -> None:
    '''Plot a rectangular gate in 3D.
    '''
    gatepoints = [np.array([dg/2, wg/2, hg/2]),
                  np.array([dg/2, -wg/2, hg/2]),
                  np.array([dg/2, -wg/2, -hg/2]),
                  np.array([dg/2, wg/2, -hg/2]),
                  np.array([-dg/2, wg/2, hg/2]),
                  np.array([-dg/2, -wg/2, hg/2]),
                  np.array([-dg/2, -wg/2, -hg/2]),
                  np.array([-dg/2, wg/2, -hg/2])]
    gate = []
    for point in gatepoints:
        point = np.matmul(point, Rg) + np.array(pg).flatten()
        gate.append(point)
    xx, yy, zz = zip(*gate)
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    i1 = [0, 3, 4, 7]
    i2 = [2, 3, 6, 7]
    i3 = [1, 2, 5, 6]
    i4 = [0, 1, 4, 5]
    ax.plot_surface(xx[i1].reshape(2, 2),
                    yy[i1].reshape(2, 2),
                    zz[i1].reshape(2, 2), alpha=alpha, color='grey')
    ax.plot_surface(xx[i2].reshape(2, 2),
                    yy[i2].reshape(2, 2),
                    zz[i2].reshape(2, 2), alpha=alpha, color='grey')
    ax.plot_surface(xx[i3].reshape(2, 2),
                    yy[i3].reshape(2, 2),
                    zz[i3].reshape(2, 2), alpha=alpha, color='grey')
    ax.plot_surface(xx[i4].reshape(2, 2),
                    yy[i4].reshape(2, 2),
                    zz[i4].reshape(2, 2), alpha=alpha, color='grey')


def plot_circular_gate(pg: c.DM, Rg: c.DM, rg: float, dg: float, ax: plt.Axes,
                       alpha: float = 1) -> None:
    '''Plot a circular gate in 3D.
    '''
    exg = Rg[:, 0]
    eyg = Rg[:, 1]
    ezg = Rg[:, 2]
    t = np.linspace(0, dg, 2)
    theta = np.linspace(0, 2 * np.pi, 100)
    rsample = np.linspace(0, rg, 2)

    # use meshgrid to make 2d arrays
    t, theta2 = np.meshgrid(t, theta)

    rsample, theta = np.meshgrid(rsample, theta)

    # generate coordinates for surface
    # "Tube"
    X, Y, Z = [pg[i] + exg[i] * (t-dg/2) +
               rg * np.sin(theta2) * eyg[i] +
               rg * np.cos(theta2) * ezg[i]
               for i in [0, 1, 2]]

    ax.plot_surface(X, Y, Z, color='grey', alpha=alpha)

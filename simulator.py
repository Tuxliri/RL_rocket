# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
from cProfile import label
import numpy as np
from scipy.integrate import RK45, solve_ivp
from math import fmod

from matplotlib import pyplot as plt


class Simulator3DOF():
    def __init__(self, IC, dt=0.5, dynamics='std3DOF') -> None:
        super(Simulator3DOF, self).__init__()

        self.dynamics = dynamics
        self.timestep = dt
        self.t = 0
        self.state = IC                     # state is in GLOBAL COORDINATES

        self.states = [IC]
        self.actions = []
        self.derivatives = []

        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = 45000                      # rocket initial mass
        self.Cdalfa = 2                     # drag coefficient [-]
        self.Cnalfa = 1                     # normal force coefficient [-]
        self.I = 6.04e6                     # inertia moment [kg*m^2]
        self.Isp = 360                      # Specific impulse [s]

        # Geometric properties NB: REDEFINE THEM FROM THE TIP OF THE BOOSTER!! (or change torque equation in RHS)
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 10.5                      # Reference surface [m^2]
        self.x_PVP = 0                      # Thrust gimbal point [m]
        self.x_T = 40

        pass

    def step(self, u):

        if self.dynamics == 'std3DOF':
            def _event(t, y):
                return y[1]

            # RK integration
            _event.terminal = True

            solution = solve_ivp(
                fun=lambda t, y: self.RHS(t, y, u),
                t_span=[self.t, self.t+self.timestep],
                y0=self.state,
                events=_event
            )

            self.state = np.array([var[-1] for var in solution.y])

            self.t += self.timestep

            self.state[2] = self._wrapTo2Pi(self.state[2])

        else:
            raise NotImplementedError()

        # Keep track of all states
        self.states.append(self.state)
        self.actions.append(u)

        return self.state, {'states': self.states, 'derivatives': self.derivatives}, solution.status

    def RHS(self, t, state, u):
        """ 
        Function computing the derivatives of the state vector
        in inertial coordinates
        """
        # extract dynamics variables
        x, y, phi, vx, vz, om, mass = state

        # Get control variables
        delta = u[0]
        T = u[1]

        # Implement getting it from the height (y)
        rho = 1.225  # *exp(-y/H) scaling due to height

        alfa = 0
        #alfa = self._computeAoA(y)

        # Compute aerodynamic coefficients
        Cn = self.Cnalfa*alfa
        Cd = self.Cdalfa*alfa   # ADD Cd0

        # Compute aero forces
        v2 = vx**2 + vz**2
        Q = 0.5*rho*v2

        A = Cd*Q*self.Sref

        N = Cn*Q*self.Sref

        g = self.g0

        # Compute state derivatives
        ax = (T*np.cos(delta+phi) - N*np.sin(phi) - A*np.cos(phi))/mass
        ay = (T*np.sin(delta+phi) + N*np.cos(phi) - A*np.cos(phi))/mass - g
        dom = (N*(self.x_CG - self.x_CP) - T*np.sin(delta)
               * (self.x_T - self.x_CG))/self.I

        dm = -T/(self.Isp*self.g0)

        dstate = np.array([vx, vz, om, ax, ay, dom, dm])

        return dstate

    def _computeAoA(self, state):  # CHECK
        if self.dynamics == 'std3DOF':
            phi = state[2]
            vx = state[3]
            vy = state[4]

            gamma = np.arctan2(vy, vx)

            if not(vx == 0 and vy == 0):
                alfa = phi - gamma
            else:
                alfa = 0

        else:
            raise NotImplementedError

        return self._normalize(alfa)

    def _wrapTo2Pi(self, angle):
        """
        Wrap the angle between 0 and 2 * pi.

        Args:
            angle (float): angle to wrap.

        Returns:
            The wrapped angle.

        """
        pi_2 = 2. * np.pi

        return fmod(fmod(angle, pi_2) + pi_2, pi_2)

    def _wrapToPi(self, angle):
        """
        Wrap the angle between 0 and pi.

        Args:
            angle (float): angle to wrap.

        Returns:
            The wrapped angle.

        """

        return fmod(fmod(angle, np.pi) + np.pi, np.pi)

    def _plotStates(self, VISIBLE: bool = False):
        heights = []
        downranges = []
        ths = []
        vxs = []
        vzs = []
        oms = []
        mass = []
        thrusts = []

        fig, ax = plt.subplots()

        for state in self.states:
            downranges.append(state[0])
            heights.append(state[1])
            ths.append(state[2])
            vxs.append(state[3])
            vzs.append(state[4])
            oms.append(state[5])
            mass.append(state[6])

        for action in self.actions:
            # Improvement: allow logging of both thrust and gimbaling
            thrusts.append(action[1])

        line1, = ax.plot(downranges, label='Downrange (x)')
        line2, = ax.plot(heights, label='Height (y)')
        line3, = ax.plot(ths, label='phi')

        #line4, = ax.plot(vxs, label='Cross velocity (v_x)')
        line5, = ax.plot(vzs, label='Vertical velocity (v_z)')
        #line6, = ax.plot(mass, label='mass')

        ax.legend()

        fig2, ax2 = plt.subplots()
        __, = ax2.plot(thrusts, label='Thrust (N)')
        ax.legend()

        if VISIBLE:
            plt.show(block=False)

        return fig, fig2

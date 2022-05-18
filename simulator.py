# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
import numpy as np
from scipy.integrate import RK45, odeint
from math import fmod

from matplotlib import pyplot as plt


class Simulator():
    def __init__(self, IC, dt=0.5, dynamics='std3DOF') -> None:
        super(Simulator, self).__init__()

        self.dynamics = dynamics
        self.timestep = dt
        self.t = 0
        self.state = IC                     # state is in GLOBAL COORDINATES
        self.localState = self._globalToLocal(self.state)

        self.states = [IC]
        self.derivatives = []

        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = 45000                      # rocket initial mass
        self.maxGimbal = np.deg2rad(20)     # [rad]
        self.maxThrust = 500                # [N]
        self.minThrust = 100                # [N]
        self.Cdalfa = 2                     # drag coefficient [-]
        self.Cnalfa = 1                     # normal force coefficient [-]
        self.I = 6.04e6                     # inertia moment [kg*m^2]

        # Geometric properties
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 10.5                      # Reference surface [m^2]
        self.x_PVP = 0                      # Thrust gimbal point [m]

        pass

    def step(self):

        if self.dynamics == 'std3DOF':
            theta = self.state[2]
            fx = self.RHS(self.t, self._globalToLocal(self.state))
            
            #euler integration
            self.localState = self.localState + self.timestep*fx
            self.derivatives.append(fx)

            # integrate with RK45
            """ timeRange = [self.t, self.t+self.timestep]
            solution = odeint(lambda y,t : self.RHS(t,y),
                              localState,
                              timeRange,rtol=1e-8,atol=1e-8)

            localState = solution[-1] """

            self.t += self.timestep

            self.localState[2] = self._wrapTo2Pi(self.localState[2])
            self.state = self._localToGlobal(self.localState, theta)

        elif self.dynamics == 'linear3DOF':
            raise NotImplementedError()

        elif self.dynamics == '6DOF':
            # Implement the Simulink interface
            # here, with the step() method
            raise NotImplementedError

        else:
            raise NotImplementedError()

        # Keep track of all states
        self.states.append(self.state)
        
        return self.state

    def RHS(self, t, y):
        """ 
        Function computing the derivatives of the state vector
        in body coordinates
        """
        # extract dynamics variables
        x, z, th, dx, dz, dth = y

        # Get control variables
        T = 0  # *u[0]
        beta = 0  # *u[1]

        # Implement getting it from the height (z)
        rho = 1.225

        alfa = 0
        #alfa = self._computeAoA(y)

        # Compute aerodynamic coefficients
        Cn = self.Cnalfa*alfa
        Cd = self.Cdalfa*alfa   # ADD Cd0

        # Compute aero forces
        v2 = dx**2 + dz**2
        Q = 0.5*rho*v2

        D = Cd*Q*self.Sref

        N = Cn*Q*self.Sref

        g = self.g0

        # Torque arms
        l_alfa = self.x_CP - self.x_CG
        l_c = self.x_CG - self.x_PVP

        # Compute state derivatives
        ddx = (T*np.cos(beta) - D)/self.m - g*np.sin(th) - dth*dz
        ddz = g*np.cos(th) + dx*dth + (-N - T*np.sin(beta))/self.m
        ddth = (l_alfa*N - l_c*T*np.sin(beta))/self.I
        # dm = T/(self.Isp*self.g0)

        dy = np.array([dx, dz, dth, ddx, ddz, ddth])
        return dy

    def _computeAoA(self, state):
        if self.dynamics == 'std3DOF':
            vx = state[3]
            vz = state[4]

            alfa = np.arctan2(vz, vx)

        elif self.dynamics == '6DOF':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return alfa

    def _localToGlobal(self, stateLocal, theta):
        '''
        Need to fix this as it needs the angle BEFORE
        the euler integration step
        '''
        if self.dynamics == 'std3DOF':
            #theta = stateLocal[2]
            c = np.cos(theta)
            s = np.sin(theta)
            ROT = np.array([[c, s],
                            [-s, c]])

            stateGlobal = np.copy(stateLocal)
            stateGlobal[0:2] = ROT @ stateLocal[0:2]
            stateGlobal[3:5] = ROT @ stateLocal[3:5]
            
        elif self.dynamics == '6DOF':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return stateGlobal

    def _globalToLocal(self, stateGlobal):
        '''
        Need to fix this as it is working
        only for the attitude, not for the global pose
        '''
        if self.dynamics == 'std3DOF':
            theta = stateGlobal[2]
            ROT = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            stateLocal = np.copy(stateGlobal)
            stateLocal[0:2] = ROT @ stateGlobal[0:2]
            stateLocal[3:5] = ROT @ stateGlobal[3:5]

        elif self.dynamics == '6DOF':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return stateLocal

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

    def _plotStates(self,times):
        height = []
        downrange = []
        ths = []
        vxs = []
        vzs = []
        oms = []
        ddzs = []
        fig, ax = plt.subplots()

        for state in self.states:
            downrange.append(state[0])
            height.append(state[1])
            ths.append(state[2])
            vxs.append(state[3])
            vzs.append(state[4])
            oms.append(state[5])

        for dy in self.derivatives:
            ddzs.append(dy[4])
        
        ts = np.array(times)
        #analytical_velz = 9.81*ts*np.cos(0.1*ts)
        line1, = ax.plot(downrange, label='Downrange (x)')
        line2, = ax.plot(height, label='Height (-z)')
        #line3, = ax.plot(vxs, label='Cross velocity (v_x)')
        #line4, = ax.plot(vzs, label='Cross velocity (v_z)')
        #line5, = ax.plot(analytical_velz, label='Analytical v_bz')
        #line6, = ax.plot(ths, label='theta')
        #line7, = ax.plot(ddzs, label='ddz')
        RHS = 9.81*np.cos(np.array(ths)) + np.array(oms)*np.array(vzs)
        #line8, = ax.plot(RHS, label='RHS')
        ax.legend()
        plt.show()

       
        return height, downrange


if __name__ == "__main__" :
    IC = np.array([0, -10000, np.pi/2, 0, 0, 0.1])
    RKT1 = Simulator(IC, 1)
    states = []
    times = []

    while RKT1.t < 100:
        states.append(RKT1.step())
        times.append(RKT1.t)

    heights = RKT1._plotStates(times)

    print(heights[-1])
    print(RKT1.t)

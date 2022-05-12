# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
from math import radians
import numpy as np
from scipy import integrate
from math import fmod


class Simulator():
    def __init__(self, IC, dt=0.01, dynamics='std3DOF') -> None:
        super(Simulator, self).__init__()

        self.dynamics = dynamics
        self.timestep = dt
        self.t = 0
        self.dt = 0.5
        self.state = IC

        # Define environment properties
        self.g0 = 9.81

        # Define rocket properties
        self.m = 5000                       # rocket initial mass
        self.maxGimbal = np.deg2rad(20)     # [rad]
        self.maxThrust = 500                # [N]
        self.minThrust = 100                # [N]
        self.Cdalfa = 2                     # drag coefficient [-]
        self.Cnalfa = 1                     # normal force coefficient [-]
        self.I = 1000                       # inertia moment [kg*m^2]

        # Geometric properties
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 50                      # Reference surface [m^2]
        self.x_PVP = 0                      # Thrust gimbal point [m]

        pass

    def step(self):

        if self.dynamics == 'std3DOF':
            solution = integrate.solve_ivp(self.RHS, [0, 60],
                self._globalToLocal(self.state)).y[:, -1]
            solution[2] = self._wrapTo2Pi(solution[2])
            return solution

        elif self.dynamics == 'linear3DOF':
            raise NotImplementedError()

        elif self.dynamics == '6DOF':
            # Implement the Simulink interface
            # here, with the step() method
            raise NotImplementedError

        else:
            raise NotImplementedError()

    def RHS(self, t, y):
        # extract dynamics variables
        x, z, th, dx, dz, dth = y

        # Get control variables
        T = 0  # *u[0]
        beta = 0  # *u[1]

        # Implement getting it from the height (z)
        rho = 1.225

        alfa = 0
        alfa = self._computeAoA(y)

        # Compute aerodynamic coefficients
        Cn = self.Cnalfa*alfa
        Cd = self.Cdalfa*alfa

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
        ddx = (T*np.cos(beta) - D)/self.m
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

    def _localToGlobal(self, stateLocal):
        if self.dynamics == 'std3DOF':
            theta = stateLocal[2]
            ROT = np.array([np.cos(theta), np.sin(theta)],
                           [-np.sin(theta), np.cos(theta)])
            stateGlobal = stateLocal
            stateGlobal[0:2] = np.dot(
                ROT, stateLocal[0:2]
            )
            stateGlobal[3:5] = np.dot(
                ROT, stateLocal[3:5]
            )
        elif self.dynamics == '6DOF':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return stateGlobal

    def _globalToLocal(self, stateGlobal):
        if self.dynamics == 'std3DOF':
            theta = stateGlobal[2]
            ROT = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
            stateLocal = stateGlobal
            stateLocal[0:2] = ROT @ stateGlobal[0:2]
            stateLocal[3:5] = ROT @ stateGlobal[3:5]

        elif self.dynamics == '6DOF':
            raise NotImplementedError

        else:
            raise NotImplementedError

        return stateLocal

    def _wrapTo2Pi(self,angle):
        """
        Wrap the angle between 0 and 2 * pi.

        Args:
            angle (float): angle to wrap.

        Returns:
            The wrapped angle.

        """
        pi_2 = 2. * np.pi

        return fmod(fmod(angle, pi_2) + pi_2, pi_2)

if __name__ == "__main__":
    IC = np.array([10, 10, 10, 3, 4, 5])
    RKT1 = Simulator(IC)
    RKT1.step()

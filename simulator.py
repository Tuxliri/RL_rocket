# In this file the dynamics are simulated using
# different kind of simulators. A 3DOF simulator,
# a linearized 3DOF and a 6DOF simulink simulator
import numpy as np
from scipy import integrate

class Simulator():
    def __init__(self, IC, dt = 0.5, dynamics = 'std3DOF') -> None:
        super(Simulator,self).__init__()
        
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
        self.Cd = 2                         # drag coefficient [-]
        self.Cn = 1                         # normal force coefficient [-]
        self.I = 1000                       # inertia moment [kg*m^2]

        # Geometric properties
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 50                      # Reference surface [m^2]

        pass

    def step(self):

        if self.dynamics == 'std3DOF':
            return integrate.solve_ivp(self.RHS, [self.t,60+ self.t + self.dt], self.state).y[:,-1]

        elif self.dynamics == 'linear3DOF':
            raise NotImplementedError()

        else:
            raise NotImplementedError()

    def RHS(self, t, y):
            # extract dynamics variables
            x, z, th, dx, dz, dth = y

            # Get control variables
            T = 0#*u[0]
            beta = 0#*u[1]

            # Implement getting it from the height (z)
            rho = 1.225

            # Compute aero forces
            v2 = dx**2 + dz**2
            Q = 0.5*rho*v2
            D = self.Cd*Q*self.Sref

            N = self.Cn*Q*self.Sref

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


if __name__ == "__main__":
    IC = np.array([10, 10, 10, 3, 4, 5])
    RKT1 = Simulator(IC)
    RKT1.step()

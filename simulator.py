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

        # Geometric properties NB: REDEFINE THEM FROM THE TIP OF THE BOOSTER!! (or change torque equation in RHS)
        self.x_CG = 10                      # Center of gravity [m]
        self.x_CP = 20                      # Center of pressure [m]
        self.Sref = 10.5                      # Reference surface [m^2]
        self.x_PVP = 0                      # Thrust gimbal point [m]
        self.x_T = 40

        pass

    def step(self):

        if self.dynamics == 'std3DOF':
            
            fx = self.RHS(self.t, self.state)
            
            #euler integration
            self.state = self.state + self.timestep*fx
            self.derivatives.append(fx)

            self.t += self.timestep

            self.state[2] = self._wrapTo2Pi(self.state[2])

        elif self.dynamics == '6DOF':
            # Implement the Simulink interface
            # here, with the step() method
            raise NotImplementedError

        else:
            raise NotImplementedError()

        # Keep track of all states
        self.states.append(self.state)
        
        return self.state

    def RHS(self, t, state):
        """ 
        Function computing the derivatives of the state vector
        in inertial coordinates
        """
        # extract dynamics variables
        x, y, phi, vx, vz, om = state

        # Get control variables
        T = 0  # *u[0]
        delta = 0  # *u[1]

        # Implement getting it from the height (y)
        rho = 1.225 #*exp(-y/H) scaling due to height

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
        ax = (T*np.cos(delta+phi) - N*np.sin(phi) - A*np.cos(phi))/self.m
        ay = (T*np.sin(delta+phi) + N*np.cos(phi) - A*np.cos(phi))/self.m - g
        dom = (N*(self.x_CG - self.x_CP) - T*np.sin(delta)*(self.x_T - self.x_CG))/self.I
        
        # dm = T/(self.Isp*self.g0)

        dstate = np.array([vx, vz, om, ax, ay, dom])

        return dstate

    def _computeAoA(self, state): # CHANGE
        if self.dynamics == 'std3DOF':
            phi = state[2]
            vx = state[3]
            vy = state[4]

            gamma = np.arctan2(vy, vx)

            if not( vx == 0 and vy == 0):
                alfa = phi - gamma
            else:
                alfa = 0

        else:
            raise NotImplementedError

        return self._normalize(alfa)

    def _normalize(self, num, lower=-np.pi, upper=np.pi):
    
        from math import floor, ceil
        # abs(num + upper) and abs(num - lower) are needed, instead of
        # abs(num), since the lower and upper limits need not be 0. We need
        # to add half size of the range, so that the final result is lower +
        # <value> or upper - <value>, respectively.
        res = num
        
        if lower >= upper:
            raise ValueError("Invalid lower and upper limits: (%s, %s)" %
                            (lower, upper))

        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if res == upper else num
        
        res = num * 1.0  # Make all numbers float, to be consistent

        return res

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
        line2, = ax.plot(height, label='Height (y)')
        #line3, = ax.plot(vxs, label='Cross velocity (v_x)')
        #line4, = ax.plot(vzs, label='Cross velocity (v_z)')
        #line5, = ax.plot(analytical_velz, label='Analytical v_bz')
        line6, = ax.plot(ths, label='phi')
        #line7, = ax.plot(ddzs, label='ddz')
        #line8, = ax.plot(RHS, label='RHS')
        ax.legend()
        plt.show()

       
        return height, downrange


if __name__ == "__main__" :
    IC = np.array([0, 0, np.pi/2, 0, 0, 0])
    RKT1 = Simulator(IC, 0.5)
    states = []
    times = []

    while RKT1.t < 100:
        states.append(RKT1.step())
        times.append(RKT1.t)

    heights = RKT1._plotStates(times)

    print(heights[-1])
    print(RKT1.t)

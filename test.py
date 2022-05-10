import simulator as sim
import numpy as np

IC = np.array([10,10, 10, 3, 4, 5])
RKT1 = sim.Simulator(IC)

RESULT = RKT1.step()
print(RESULT.y[:,-1])
RESULT = RKT1.step()

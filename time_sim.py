from timeit import timeit

SETUP_CODE='''
import simulator as sim
import numpy as np

IC = np.array([10,10, 10, 3, 4, 5])
RKT1 = sim.Simulator(IC)
'''

RUN_CODE = '''
RKT1.step()
'''

print(timeit(RUN_CODE,SETUP_CODE, number=1000)/1000)
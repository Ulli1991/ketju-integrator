"""
A simple example of using the bindings with two BHs.
Run from this directory or with this directory on the python 
search path for the below import to work.
"""
from ketju_integrator import *

import numpy as np
import matplotlib.pyplot as plt

# Initial state of the system
init_state = State(0., np.array([1., 1.]),
                   np.array([[40.,0,0], [-40.,0,0]]),
                   np.array([[0.,.085, 0], [0.,-0.085,0]]),
                   np.array([[1.,0,0], [0.,1.,0.]])
                   )
# Output times
ts = np.linspace(0,1e5, 1000)

# run the integrator
ret_states = run_integrator(init_state, ts, nproc=4, PN_flags=PNFlags.PN_ALL)

# Plot orbits
pos1 = np.array([st.pos[0] for st in ret_states])
pos2 = np.array([st.pos[1] for st in ret_states])
plt.plot(*pos1.T[:2])
plt.plot(*pos2.T[:2])
plt.show()


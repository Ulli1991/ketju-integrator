"""
Bindings to the BH merger remnant calculation for testing.
"""
import os
import ctypes

import numpy as np

class BHMergerInput(ctypes.Structure):
    _fields_ = [
    ('G', ctypes.c_double),
    ('c', ctypes.c_double),
    ('m1', ctypes.c_double),
    ('m2', ctypes.c_double),
    ('x1', ctypes.c_double*3),
    ('x2', ctypes.c_double*3),
    ('v1', ctypes.c_double*3),
    ('v2', ctypes.c_double*3),
    ('S1', ctypes.c_double*3),
    ('S2', ctypes.c_double*3),
    ]

    def __init__(self, m1, m2,  x1, x2, v1, v2, S1, S2, G=1., c=1.):
        self.m1 = m1
        self.m2 = m2
        self.G = G
        self.c = c
        np.ctypeslib.as_array(self.x1)[:] = np.asarray(x1)
        np.ctypeslib.as_array(self.x2)[:] = np.asarray(x2)
        np.ctypeslib.as_array(self.v1)[:] = np.asarray(v1)
        np.ctypeslib.as_array(self.v2)[:] = np.asarray(v2)
        np.ctypeslib.as_array(self.S1)[:] = np.asarray(S1)
        np.ctypeslib.as_array(self.S2)[:] = np.asarray(S2)


class BHMergerRemnantProperties(ctypes.Structure):
    _fields_ = [
    ('m', ctypes.c_double),
    ('x', ctypes.c_double*3),
    ('v', ctypes.c_double*3),
    ('S', ctypes.c_double*3),
    ]

_this_file = os.path.realpath(__file__)
shim_file = os.path.join(os.path.dirname(_this_file), 
                         "../../lib/libketju_integrator_python_shim.so")
shim = ctypes.cdll.LoadLibrary(shim_file)
shim.ketju_calculate_bh_merger_remnant_properties.argtypes = [
                ctypes.POINTER(BHMergerInput),
                ctypes.POINTER(BHMergerRemnantProperties)]


def normalized_merger_properties(q, R, V, chi1_vec, chi2_vec):
    """
    Merger properties for a system with mass ratio q on a separation R (along x)
    with a relative velocity V (along y) and the given spin magnitudes (chi = S/m**2).
    """
    m1 = 1/(1+q)
    m2 = 1 - m1
    S1 = m1**2 * np.asarray(chi1_vec)
    S2 = m2**2 * np.asarray(chi2_vec)
    bh_in = BHMergerInput(m1, m2, [m2*R, 0, 0], [-m1*R, 0, 0], [0, m2*V, 0],
                          [0, -m1*V, 0], S1, S2)
    bh_out = BHMergerRemnantProperties()
    shim.ketju_calculate_bh_merger_remnant_properties(bh_in, bh_out)

    return dict(mass=bh_out.m, x=np.array(bh_out.x), v=np.array(bh_out.v),
                S=np.array(bh_out.S), chi=np.linalg.norm(bh_out.S)/bh_out.m**2)


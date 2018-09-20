import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import parallel, prange
cimport openmp

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport sin, cos

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def trace(np.ndarray[DTYPE_t, ndim=1] wl, np.ndarray[DTYPE_t, ndim=1] x_vec, np.ndarray[DTYPE_t, ndim=1] y_vec, float sx,
          float sy, float rot, float shear, float offset_x, float offset_y, float disp_x, float disp_y, float wl_start):
    cdef Py_ssize_t i, N = len(wl)
    cdef np.ndarray[DTYPE_t, ndim=1] transformed_x = np.zeros([N], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] transformed_y = np.zeros([N], dtype=DTYPE)
    assert wl.dtype == DTYPE and y_vec.dtype == DTYPE and x_vec.dtype == DTYPE
    cdef double tx, ty, value, xpos, ypos, m0,m1,m2, m3,m4, m5

    for i in cython.parallel.prange(N, nogil=True, num_threads=8, schedule='static'):
        # calculate offsets
        tx = (wl[i] - wl_start) * disp_x + offset_x
        ty = (wl[i] - wl_start) * disp_y + offset_y
        # calculate affine matrix
        m0 = sx * cos(rot)
        m1 = -sy * sin(rot + shear)
        m2 = tx
        m3 = sx * sin(rot)
        m4 = sy * cos(rot + shear)
        m5 = ty
        # do transformation
        xpos = m0*x_vec[i] + m1*y_vec[i] + m2
        ypos = m3*x_vec[i] + m4*y_vec[i] + m5

        transformed_x[i] = xpos
        transformed_y[i] = ypos

    return np.vstack((transformed_x, transformed_y))
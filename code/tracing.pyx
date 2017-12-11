import numpy as np
cimport cython
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport sin, cos


cdef inline lambda_to_x_y(DTYPE_t wl):
    cdef DTYPE_t x,y
    x = (wl-600.)*500. + 50.
    y = 10.
    return x,y

cdef inline compose_matrix(float sx,float sy,float rot,float shear,float tx,float ty):
    return sx * cos(rot), -sy * sin(rot + shear), sx * sin(rot),  sy * cos(rot + shear), tx, ty

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def trace(np.ndarray[DTYPE_t, ndim=1] wl,np.ndarray[DTYPE_t, ndim=1] x_vec, np.ndarray[DTYPE_t, ndim=1] y_vec, float sx, float sy, float rot, float shear):
    cdef int N = len(wl)
    cdef np.ndarray transformed = np.zeros([N,3], dtype=DTYPE)
    assert wl.dtype == DTYPE and y_vec.dtype == DTYPE and x_vec.dtype == DTYPE
    cdef DTYPE_t tx, ty, value, xpos, ypos
    for i in range(N):
        tx, ty = lambda_to_x_y(wl[i])
        m00, m10, m01, m11, m02, m12  = compose_matrix(sx,sy,rot,shear,tx,ty)

        xpos = m00*x_vec[i] + m01*y_vec[i] + m02
        ypos = m10*x_vec[i] + m11*y_vec[i] + m12
        # transformed[i] = np.dot(m, xy_vec[i].T)
        transformed[i][0] = xpos
        transformed[i][1] = ypos

    return transformed[:, :2].T
import ctypes as C
import torch
import numpy as np

# load in external library
aglib = C.cdll.LoadLibrary('./anigauss.so')

def anigauss(inarr, sigv, sigu, phi=0., derv=0, deru=0):
    
    # make sure we have a C-order array of doubles
    if inarr.dtype is not np.double:
        inarr = inarr.astype(np.double, order='C')
    elif not inarr.flags.c_contiguous:
        inarr = np.ascontiguousarray(inarr)

    # create output array
    outarr = np.zeros_like(inarr, dtype=np.double)

    # size parameters for array
    (sizey, sizex) = inarr.shape

    # call external function
    aglib.anigauss(inarr.ctypes.data_as(C.POINTER(C.c_double)),
                   outarr.ctypes.data_as(C.POINTER(C.c_double)),
                   C.c_int(sizex), C.c_int(sizey), C.c_double(sigv), C.c_double(sigu),
                   C.c_double(phi), C.c_int(derv), C.c_int(deru))

    # return filtered image
    return outarr

def get_gaussian_kernel(ksize=17, sigma=1.5, phi=None):

    kd2 = (ksize - 1) // 2

    zz = np.zeros((ksize, ksize))
    zz[kd2][kd2] = 1

    if phi is None:
        kernel = anigauss(zz, sigma, sigma)
        kernel_iso = kernel / kernel.sum()
        return torch.tensor(kernel_iso, dtype=torch.float32)
    
    else:
        kernel = anigauss(zz, sigma, sigma*2, phi=phi)
        kernel_aniso = kernel / kernel.sum()
        return torch.tensor(kernel_aniso, dtype=torch.float32)
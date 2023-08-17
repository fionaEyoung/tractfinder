import sys
import numpy as np
from mrtrix3 import app, ANSI

# Higher than console(), lower than debug()
def info(text):
  if app.VERBOSITY > 1:
    sys.stderr.write(app.EXEC_NAME + ': ' + ANSI.console + text + ANSI.clear + '\n')

## Utility functions for angles and coordinates

# Quick util function for getting angle between two polar points
def ang(azA, polA, azB, polB):
  return np.arccos( np.sin(polA)*np.sin(polB)*np.cos(azA-azB) + np.cos(polA)*np.cos(polB) )

def c2s(*args):
    # Arguments supplied as single Nx3 array
    if len(args)==1:
        C = args[0]
        assert C.ndim==2 and C.shape[1]==3
        # r, el, az
        S = np.zeros(C.shape)

        S[:,0] = np.sqrt(np.sum(C**2, axis=1))
        S[:,1] = np.arccos(C[:,2]/S[:,0])
        S[:,2] = np.arctan2(C[:,1], C[:,0])

        return S
    # Arguments supplied individually as 1D X, Y, Z arrays
    elif len(args)==3:
        X, Y, Z = args
        rho = np.sqrt(X**2 + Y**2 + Z**2)
        el = np.arccos(Z / rho)
        az = np.arctan2(Y, X)
        return rho, el, az
    else:
        raise TypeError("Supply either 1 or 3 inputs")

def s2c(*args):
    # Arguments supplied as N*3 array
    if len(args)==1:
        S = args[0]
        assert S.ndim==2

        if S.shape[1]==3:
            # x, y, z
            C = np.zeros(S.shape)

            C[:,0] = S[:,0] * np.sin(S[:,1]) * np.cos(S[:,2])
            C[:,1] = S[:,0] * np.sin(S[:,1]) * np.sin(S[:,2])
            C[:,2] = S[:,0] * np.cos(S[:,1])
            return C
        elif S.shape[1]==2:
            # x, y, z
            C = np.zeros((S.shape[0],3))

            C[:,0] = np.sin(S[:,0]) * np.cos(S[:,1])
            C[:,1] = np.sin(S[:,0]) * np.sin(S[:,1])
            C[:,2] = np.cos(S[:,0])
            return C
        else:
            raise ValueError("Second dimension must have length 2 (if only supplying angles) or 3")
    # Also support only two argumnets (El, Az), assume R=1
    elif len(args)==2:
        El, Az = args
        X = np.sin(El)* np.cos(Az)
        Y = np.sin(El)* np.sin(Az)
        Z = np.cos(El)

        return X, Y, Z
    # 3 Arguments: R, El, Az
    elif len(args)==3:
        R, El, Az = args
        X = R* np.sin(El)* np.cos(Az)
        Y = R* np.sin(El)* np.sin(Az)
        Z = R* np.cos(El)
        return X, Y, Z

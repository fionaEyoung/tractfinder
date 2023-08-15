import sys
from os import path
from pytest import raises
import numpy as np
from itertools import combinations
from virtue import c2s, s2c

TEST_DIR = path.dirname(path.realpath(__file__))

# Load testing data
dirs_cart = np.loadtxt(path.join(TEST_DIR,'dir60_cart.txt'))
dirs_sph = np.loadtxt(path.join(TEST_DIR,'dir60_sph.txt'))

assert dirs_cart.shape == (60,3)
assert dirs_sph.shape == (60,2)


# Function tests
def test_c2s():

    # Test single input matrix
    S = c2s(dirs_cart)
    assert np.allclose(S[:,1], dirs_sph[:,0])
    assert np.allclose(S[:,2], dirs_sph[:,1])
    assert np.allclose(S[:,0], np.ones(max(S.shape)))

    # Test multiple inputs
    rho, el, az = c2s(dirs_cart[:,0], dirs_cart[:,1], dirs_cart[:,2])
    assert np.allclose(el, dirs_sph[:,0])
    assert np.allclose(az, dirs_sph[:,1])
    assert np.allclose(rho, np.ones(rho.shape))

    # Test unsupported number of inputs
    with raises(TypeError):
        S = c2s(dirs_cart[:,0], dirs_cart[:,1])
    with raises(TypeError):
        rho, el = c2s(dirs_cart[:,0], dirs_cart[:,1], dirs_cart[:,0], dirs_cart[:,1])
    with raises(AssertionError):
        # Cannot input single vector array
        S = c2s(dirs_cart[:,0])

def test_s2c():

    rho = np.ones((max(dirs_sph.shape),1))
    # Test single input matrix with angles only
    C = s2c(dirs_sph)
    for i in range(3):
        assert np.allclose(C[:,i], dirs_cart[:,i])
    # Test single input matrix with rho, el, az
    C = s2c(np.hstack( (rho, dirs_sph)) )
    for i in range(3):
        assert np.allclose(C[:,i], dirs_cart[:,i])
    # Test multiple inputs with angles only
    x, y, z = s2c(dirs_sph[:,0], dirs_sph[:,1])
    assert np.allclose(x, dirs_cart[:,0])
    assert np.allclose(y, dirs_cart[:,1])
    assert np.allclose(z, dirs_cart[:,2])
    # Test multiple inputs with rho, el, az
    x, y, z = s2c(rho, dirs_sph[:,0], dirs_sph[:,1])
    assert np.allclose(x, dirs_cart[:,0])
    assert np.allclose(y, dirs_cart[:,1])
    assert np.allclose(z, dirs_cart[:,2])

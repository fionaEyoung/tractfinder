# VIRtual TUmour Expansion
import argparse
import sys, os
import numpy as np
import nibabel as nib
import skimage
from skimage import measure, morphology
from skimage.filters import gaussian
from mrtrix3.io import load_mrtrix, save_mrtrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv

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


def grow(image, tumour_mask, brain_mask, lookup=None, mode='reverse'):

    """
    Grow tumour using radial growth algorithm
    inputs
    image: input image volume
    tumour_mask: volume of segmented tumour region
    brain_mask: volume of brain region excluding skull
    outputs
    D: deformation field
    out: final image with deformation
    lookup: lookup tables for reuse
    """

    # TODO: Input checks

    # Voxel grids
    X, Y, Z = np.meshgrid(*[np.arange(i) for i in image.shape[:3]],
                          copy=False, indexing='ij')
    # Vector of voxel coordinates
    P = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    # Image dimensions
    w, l, h = img.shape[:3]

    # Largest tumour component
    tumour_modif, S = simplify_vol(tumour_mask)

    # Tumour surface and face centroids
    Ft, Vt, Ct = surf_from_vol(tumour_modif, sigma=2, target_nfaces=1000)
    # Brain surface and face centroids
    Fb, Vb, Cb = surf_from_vol(brain_mask, sigma=2, target_nfaces=5000)

    # Variables
    SP = P-S
    Dp = np.linalg.norm(SP, axis=1)
    e  = SP / Dp[:, np.newaxis] # Damn you np broadcasting
    SCb = Cb - S
    SCt = Ct - S
    Dbc = np.linalg.norm(SCb, axis=1)
    Dtc = np.linalg.norm(SCt, axis=1)

    # Spherical angles of vectors from tumour seed to image grid coordinates
    _, ELp, AZp = c2s(SP).T # Use transpose to unpack columns

    # Regular grid of spherical angles
    # TODO: !! MAGIC NUMBER !!
    n = 400;
    d_theta = 2*np.pi/n
    az, pol = np.meshgrid(np.linspace(-np.pi, np.pi, n),
                          np.linspace(0, np.pi, int(n/2)),
                          indexing="ij")

    ## Lookup tables for Dh: distance along e from seed to brain surface

    # Spherical angles connecting seed to brain hull centroids
    _, ELb, AZb = c2s(SCb).T

    # Compute angles between gridded angles and brain hull centroids
    PHI =  ang( AZb[None,:], ELb[None,:],
                az.reshape((-1,1)), pol.reshape((-1,1))
                ).reshape((400,200,-1))
    # Closest brain hull triangle for each angular interval and lookuptable for
    # Db for evenly spaced angles
    distances = Dbc[PHI.argmin(axis=2)]
    # Now find the closest grid angle for each image gridpoint and look up the
    # value of Db for that gridpoint

    Db = distances[np.floor(AZp/d_theta).astype(int)+200,
                   np.floor(ELp/d_theta).astype(int)].flatten()
    ## Lookup tables for Dt: distance along e from seed to tumour surface

    # Spherical angles connecting seed to tumour surface centroids
    _, ELt, AZt = c2s(SCt).T

    # Compute angles between gridded angles and tumour surface centroids
    PHI =  ang( AZt[None,:], ELt[None,:],
                az.reshape((-1,1)), pol.reshape((-1,1))
                ).reshape((400,200,-1))
    # Closest tumour surface triangle for each angular interval and lookuptable for
    # Dt for evenly spaced angles
    distances = Dtc[PHI.argmin(axis=2)]

    # Now find the closest grid angle for each image gridpoint and look up the
    # value of Dt for that gridpoint

    Dt = distances[np.floor(AZp/d_theta).astype(int)+200,
                   np.floor(ELp/d_theta).astype(int)].flatten()

    # Calculate displacement factor k
    if args.expon: # Exponential deformation decay
        k = np.exp( -args.expon * ((Dp - Dt)/(Db - Dt)) )
    else: # Linear deformation decay
        k = 1 - ((Dp - Dt)/(Db - Dt))

    # Deformation field
    if mode == 'reverse':
        # Return "P_old", or pull-back / reverse deformation warp convention
        return P - e * k[:, None] * Dt[:, None] * args.squish
    elif mode == 'forward':
        # Return "P_new", or forward deformation warp convention
        return P + e * k[:, None] * Dt[:, None] * args.squish
    else:
        raise ValueError(f"Unsupported mode option {mode}")
    #TODO: return Displacement field option

def ang(azA, polA, azB, polB):
    return np.arccos( np.sin(polA)*np.sin(polB)*np.cos(azA-azB) + np.cos(polA)*np.cos(polB) )

def decimate_surf():
    sphereS = vtkSphereSource()
    sphereS.Update()

    inputPoly = vtkPolyData()
    inputPoly.ShallowCopy(sphereS.GetOutput())

    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(inputPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(inputPoly.GetNumberOfPolys()) + "polygons.\n")

    decimate = vtkDecimatePro()
    decimate.SetInputData(inputPoly)
    decimate.SetTargetReduction(.10)
    decimate.Update()

    decimatedPoly = vtkPolyData()
    decimatedPoly.ShallowCopy(decimate.GetOutput())

    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimatedPoly.GetNumberOfPoints()) + "points.\n"
          "There are " + str(decimatedPoly.GetNumberOfPolys()) + "polygons.\n")

def surf_from_vol(vol, sigma=1, target_reduction=0.8, target_nfaces=None):
    """
    Get triangulated surface object from binary volume.
    This is achieved by first smoothing the binary image, finding the 0.5 isosurface and decimating the resulting mesh to a reasonable number of triangles.
    inputs
    ------
        vol:    Binary image volume with masked area for surface extraction
        sigma:  Gaussian smoothing sigma (default 1)
        target_reduction: Fraction of triangles to remove from initial surface (default 0.8)
        target_nfaces:  Number of faces in final triangulated surface. Overrides target_reduction.
    outputs
    -------
        faces:  np array of faces of size N_faces x 3
        vertices:   np array of vertices x,y,z coordinates
        centroids:  np array of centroid coordinates for each face
    """
    # Extract isosurface from smoothed volume using marching cubes algorithm
    V, F, _, _ =  measure.marching_cubes(
        gaussian(vol, sigma=sigma), 0.5, step_size=1)
    # Convert to pyvista mesh object
    mesh = pv.PolyData(V, np.c_[3*np.ones((F.shape[0],1)), F].astype(int))

    # Determine corresponding reduction if target faces given
    if target_nfaces:
        target_reduction = (mesh.n_faces - target_nfaces) / mesh.n_faces

    # Reduce number of triangles in mesh by desired amount
    decimated = mesh.decimate(target_reduction)
    # Get new faces and vertices as np arrays
    V = decimated.points
    F = decimated.faces.reshape(-1, 4)[:,1:]

    # Determine centroids for each triangel
    # (Vb[Fb] -> N_face * 3 vertices * 3 axes)
    centroids = np.squeeze( np.sum(V[F], axis=1) / 3 )

    return F, V, centroids

def simplify_vol(vol, convex_hull=True, largest_object=True):
    # TODO: Docstring
    if not any((convex_hull, largest_object)):
        raise ValueError("At least one of convex_hull and largest_object options must be true")

    ll = measure.label(vol)
    cc = measure.regionprops(ll)
    i = np.argmax([r.area for r in cc])
    x, y, z  = cc[i].centroid

    if largest_object:
        out = (ll== (1+i))

    if convex_hull:
        #convex = morphology.convex_hull_image(largest)
        out[cc[i].bbox[0]:cc[i].bbox[3],
                cc[i].bbox[1]:cc[i].bbox[4],
                cc[i].bbox[2]:cc[i].bbox[5]] = cc[i].convex_image
        return out, (x, y, z)
    else:
        return out, (x, y, z)

def load_generic(fname):
    if fname.endswith('.mif'):
        img = load_mrtrix(fname)
        return img, img.data
    elif fname.endswith('.nii') or fname.endswith('.nii.gz'):
        img = nib.load(fname)
        return img, img.get_fdata()
    else:
        raise ValueError("Invalid file extension: "+fname)

def parse_args(args):

    def valid_ext(opts):
        def type_fun(fname):
            ext = fname.split(os.extsep,1)[1]
            if not ext in opts:
                raise argparse.ArgumentTypeError(f"Only the following file extensions are supported: {opts}")
            return fname
        return type_fun
    extensions = ('mif', 'nii', 'nii.gz')

    P = argparse.ArgumentParser()
    P.add_argument('input', type=valid_ext(extensions),
                    help="input filename to be deformed")
    P.add_argument('output', type=valid_ext(extensions),
                   help="output filename for deformed image")
    P.add_argument('--tumour', type=valid_ext(extensions),
                   help="tumour mask image")
    P.add_argument('--brain', type=valid_ext(extensions),
                   help="brain mask image")
    deform_opts = P.add_mutually_exclusive_group()
    deform_opts.add_argument('--expon', type=int,
                   help="decay constant for exponentional deformation")
    deform_opts.add_argument('--squish', type=float, default=1,
                   help="squishfactor for linear deformation")

    return P.parse_args()

def main():

    args = parse_args(sys.argv[1:])

    # Extract image data arrays
    img, _ = load_generic(args.input)
    _, tumour_vol = load_generic(args.tumour)
    _, brain_vol = load_generic(args.brain)
    # TODO check dimensions on images
    if not brain_vol.shape==img.shape[:3]:
        print(brain_vol.shape)
        print(img.shape[:3])
        raise ValueError("Brain mask and image must have same voxel space")
    if not tumour_vol.shape==img.shape[:3]:
        raise ValueError("Tumour mask and image must have same voxel space")

    D = grow(img, tumour_vol, brain_vol, mode='reverse')

    # Save deformation field to mrtrix file. Convert voxel indices to scanner coordinates
    img.data = (np.hstack((img.vox * D, np.ones((max(D.shape), 1))))
                @ img.transform.T)[:,:3].reshape(*img.shape, 3)
    save_mrtrix(args.output, img)

if __name__ == '__main__':
    main()

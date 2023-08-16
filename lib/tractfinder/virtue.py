# VIRtual TUmour Expansion
# Author: Fiona Young
import os
import numpy as np
import nibabel as nib
from nibabel.orientations import apply_orientation, axcodes2ornt, aff2axcodes
import skimage
from skimage import measure, morphology
from skimage.filters import gaussian
from tractfinder.image import load_mrtrix, save_mrtrix, Image
from tractfinder.utils import c2s, s2c
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from scipy.special import lambertw

# Quick util function for getting angle between angle between two polar points
def ang(azA, polA, azB, polB):
  return np.arccos( np.sin(polA)*np.sin(polB)*np.cos(azA-azB) + np.cos(polA)*np.cos(polB) )


def entry_point(tumour_mask, brain_mask, out_path):
  tumour = load_mrtrix(brain_mask)
  brain = load_mrtrix(tumour_mask)

  imshape = brain.data.shape

  assert tumour.data.shape == imshape, "Dimension mismatch"

  tumour = np.logical_and(tumour.data, brain.data)

  # All defaults, except exponential (adaptive l)
  D = compute_radial_deformation(imshape, tumour, brain.data, expon=-1)

  out = Image.empty_as(brain)

  # Black magic that apparently I wrote
  out.data = (np.hstack((out.vox * D, np.ones((max(D.shape), 1))))
              @ out.transform.T)[:,:3].reshape(*imshape, 3)

  save_mrtrix(out_path, out)


# Get triangulated surface object from binary volume
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
        gaussian(vol.astype(float), sigma=sigma, preserve_range=True), 0.5, step_size=1)
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




# Tage a binary volume and simplify it in someway.
# Two possible simplifications can be made (or combined):
# choosing the largest connected region, or converting the
# volume into it's convex hull.
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



# The mother function! Compute radial deformation field from
# a brain and tumour mask
def compute_radial_deformation(imshape, tumour_mask, brain_mask,
                               save_lookup=None,
                               Dt=None, Db=None, S_override=None,
                               mode='reverse', field='deformation',
                               expon=None, squish=1, expon_const=False,
                               v=0, convex_tumour=False):
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
    X, Y, Z = np.meshgrid(*[np.arange(i) for i in imshape[:3]],
                          copy=False, indexing='ij')
    # Vector of voxel coordinates
    P = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    # Image dimensions
    w, l, h = imshape[:3]

    # Largest tumour component and centre point S
    tumour_modif, S = simplify_vol(tumour_mask, convex_hull=convex_tumour)
    if S_override is not None:
        S = S_override

    if v > 1:
        print(f"Tumour seed voxel: {S}")

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

    if isinstance(Dt, str):
        Dt = np.load(Dt)
    if isinstance(Db, str):
        Db = np.load(Db)
    # Otherwise Dt/Db are either not set (so calculate them) or are already npy arrays

    if (Dt is None) or (Db is None):

        if v > 0:
            print("Computing Dt and Db lookup tables...")

        # Spherical angles of vectors from tumour seed to image grid coordinates
        _, ELp, AZp = c2s(SP).T # Use transpose to unpack columns

        # Regular grid of spherical angles
        # TODO: !! MAGIC NUMBER !!
        n = 400;
        d_theta = 2*np.pi/n
        az, pol = np.meshgrid(np.linspace(-np.pi, np.pi, n),
                              np.linspace(0, np.pi, int(n/2)),
                              indexing="ij")

        if Db is None:
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
            if save_lookup:
                np.save(os.path.join(save_lookup, "Db.npy"), Db)

        if Dt is None:
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
            if save_lookup:
                np.save(os.path.join(save_lookup, "Dt.npy"), Dt)

    # Logical mask for brain region voxels
    m = brain_mask.flatten().astype(bool)
    # Modulate "tumour size" with squishfactor
    Dt *= squish

    if expon:
        # Set maximum value for strictly outside deformation
        if not expon_const:
            # Initialise decay constant
            l = Db/Dt
            # Normalisation constant
            c = np.exp(-l)/(np.exp(-l)-1)
            # Iterate optimum value of l
            niter = 5
            for i in range(niter):
                l = Db/(Dt * (1-c))
                c = np.exp(-l)/(np.exp(-l)-1)

            if not expon == -1:
                print(f"lambda min: {min(l):.1f}, lambda max: {max(l):.1f}. Requested lambda: {expon:.1f}")
                l = np.minimum(l, expon)

        else:
            l = expon


    # Return "P_new", or forward deformation warp convention
    if mode == 'forward':

        if expon: # Exponential deformation decay
            if v > 0:
                print(f"""Computing exponential tissue deformation with decay lambda = {expon} and
                      squishfactor {squish}""")

            k = np.zeros(Dp.shape)
            # Normalisation constant
            c = np.exp(-l)/(np.exp(-l)-1)
            # Displacement factor
            k = (1-c) * np.exp( -l * ((Dp)/(Db)) ) + c
            k[~m] = 0

            return (field=='deformation')*P + e * k[:, None] * Dt[:, None]

        else: # Linear deformation decay
            if v > 0:
                print(f"""Computing linear tissue deformation with squishfactor {squish}""")
            k = 1 - (Dp/Db)
            k[~m] = 0

            return (field=='deformation')*P + e * k[:, None] * Dt[:, None]

    # Return "P_old", or pull-back / reverse deformation warp convention
    elif mode == 'reverse':

        # Calculate displacement factor k
        if expon: # Exponential deformation decay
            if v > 0:
                print(f"""Computing exponential tissue deformation with decay lambda = {expon} and
                      squishfactor {squish}""")

            k = np.zeros(Dp.shape)
            c = (np.exp(-l))/(np.exp(-l)-1)

            k = Dt*c - Db/l * lambertw( (-l * Dt * (1-c) * np.exp(-l/Db * (Dp - Dt*c)))/Db , k=0 ).real
            # Zero outside brain surface
            k[~m] = 0

            return (field=='deformation')*P - e * k[:, None]

        else: # Linear deformation decay
            if v > 0:
                print(f"""Computing linear tissue deformation with squishfactor {squish}""")
            k = 1 - ((Dp - Dt)/(Db - Dt))
            k[~m] = 0

            return (field=='deformation')*P - e * k[:, None] * Dt[:, None]

    elif mode == 'both':

        if expon: # Exponential deformation decay
            if v > 0:
                print(f"""Computing exponential tissue deformation with decay lambda = {expon} and
                      squishfactor {squish}""")

            k = np.zeros(Dp.shape)
            # Normalisation constant
            c = (np.exp(-l))/(np.exp(-l)-1)

            # Forward displacement factor
            k = (1-c) * np.exp( -l * ((Dp)/(Db)) ) + c
            k[~m] = 0
            D_forward = (field=='deformation')*P + e * k[:, None] * Dt[:, None]

            # Reverse displacement factor
            k = Dt*c - Db/l * lambertw( (-l * Dt * (1-c) * np.exp(-l/Db * (Dp - Dt*c)))/Db , k=0 ).real
            k[~m] = 0
            D_reverse = (field=='deformation')*P - e * k[:, None]

            return (D_forward, D_reverse)

        else: # Linear deformation decay
            if v > 0:
                print(f"""Computing linear tissue deformation with squishfactor {squish}""")

            k = 1 - (Dp/Db)
            k[~m] = 0
            D_forward = (field=='deformation')*P + e * k[:, None] * Dt[:, None]

            k = 1 - ((Dp - Dt)/(Db - Dt))
            k[~m] = 0
            D_reverse = (field=='deformation')*P - e * k[:, None] * Dt[:, None]

            return (D_forward, D_reverse)

    else:
        raise ValueError(f"Unsupported mode option {mode}")
    #TODO: return Displacement field option

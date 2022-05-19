# VIRtual TUmour Expansion
import argparse
import sys, os
from pathlib import Path
import numpy as np
import nibabel as nib
import skimage
from skimage import measure, morphology
from skimage.filters import gaussian
from mrtrix3.io.image import load_mrtrix, save_mrtrix, Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
from scipy.special import lambertw

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


def grow(imshape, tumour_mask, brain_mask,
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

def ang(azA, polA, azB, polB):
    return np.arccos( np.sin(polA)*np.sin(polB)*np.cos(azA-azB) + np.cos(polA)*np.cos(polB) )

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


def load_generic(fname, voxel_array_only=False):
    if fname.endswith('.mif'):
        img = load_mrtrix(fname)
        if voxel_array_only:
            return img.data
        else:
            return {'full': img,
                    'data': img.data,
                    'voxdims': img.vox,
                    'transform': img.transform}
    elif fname.endswith('.nii') or fname.endswith('.nii.gz'):
        img = nib.load(fname)
        if voxel_array_only:
            return img.get_fdata()
        else:
            return {'full': img,
                    'data': img.get_fdata(),
                    'voxdims': img.header.get_zooms(),
                    'transform': img.affine}
    else:
        raise ValueError("Invalid file extension: "+fname)

def parse_args(args):

    def valid_ext(opts):
        def type_fun(fname):
            ext = ''.join(Path(fname).suffixes)
            if not ext in opts:
                raise argparse.ArgumentTypeError(f"Filename {fname}: Only the following file extensions are supported: {opts}")
            return fname
        return type_fun
    extensions = ('.mif', '.nii', '.nii.gz')

    def valid_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"Not a valid directory: {string}")


    P = argparse.ArgumentParser()
    P.add_argument('input', type=valid_ext(extensions),
                    help="input filename to be deformed")
    P.add_argument('output', type=valid_ext(('.mif')), # Only supprt write to .mif file
                   help="output filename for deformed image")
    P.add_argument('--tumour', '-t', type=valid_ext(extensions), required=True,
                   help="tumour mask image")
    P.add_argument('--brain', '-b', type=valid_ext(extensions), required=True,
                   help="brain mask image")
    P.add_argument('--expon', type=float, nargs='?', const=-1,
                   help="Decay constant for exponentional deformation. If no value specified, program will dynamically set optimum value.")
    P.add_argument('--squish', type=float, default=1,
                   help="squishfactor for linear deformation")
    P.add_argument('--precomputed', type=valid_path,
                   help="directory containing precomputed arrays for Dt and Db")
    P.add_argument('--save', type=str,
                   help="directory to save arrays Dt and Db for future use")
    P.add_argument('--verbosity', '-v', action='count', default=0,
                   help="Increase output verbosity")
    P.add_argument('--seed_override', type=str,
                   help="Manually specify seed point from which tumour grows, as comma separated list of voxel coordinates")
    P.add_argument('--def_mode', type=str, choices=['forward', 'reverse', 'both'], default='forward',
                   help="Specify forward or reverse deformation field convention")
    P.add_argument('--out_format', type=str, choices=['deformation', 'displacement'], default='displacement',
                   help="Specify whether output should be a deformation or displacement field")
    P.add_argument('--expon_const', action='store_true',
                   help="Override optimum exponential factor, keep constant at set value. (This may result in weak deformation if lambda is too high)")
    P.add_argument('--convex_tumour', action='store_true',
                   help="Use convex hull of tumour segmentation to calculate deformation.")

    a = P.parse_args(args)

    # Handle arguments
    if a.seed_override:
        a.seed_override = np.array(a.seed_override.split(','), dtype=float)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    if a.expon == -1 and a.expon_const:
        P.error("Must specify value for constant decay factor using '--expon VAL'.")

    return a

def main():

    args = parse_args(sys.argv[1:])

    # Extract image data arrays
    img = load_generic(args.input)
    tumour_vol = load_generic(args.tumour, voxel_array_only=True)
    brain_vol = load_generic(args.brain, voxel_array_only=True)
    # TODO check dimensions on images
    if not brain_vol.shape==img['data'].shape[:3]:
        raise ValueError(f"""Dimension mismatch.
                         Brain mask {brain_vol.shape} and
                         image {img['data'].shape[:3]} must have same voxel space.""")
    if not tumour_vol.shape==img['data'].shape[:3]:
        raise ValueError(f"""Dimension mismatch.
                         Tumour mask {tumour_vol.shape} and
                         image {img['data'].shape[:3]} must have same voxel space.""")
    # Crop tumour mask to brain mask
    tumour_mask = np.logical_and(tumour_vol, brain_vol)

    Dt, Db = (None, None)
    if args.precomputed:
        if os.path.exists(os.path.join(args.precomputed, 'Dt.npy')):
            Dt = os.path.join(args.precomputed, 'Dt.npy')
        if os.path.exists(os.path.join(args.precomputed, 'Db.npy')):
            Db = os.path.join(args.precomputed, 'Db.npy')

    # Calculate deformation field from tumour
    D = grow(img['data'].shape, tumour_vol, brain_vol,
             mode=args.def_mode, expon=args.expon, squish=args.squish,
             save_lookup=args.save, Dt=Dt, Db=Db, v=args.verbosity,
             S_override=args.seed_override, expon_const=args.expon_const,
             convex_tumour=args.convex_tumour, field=args.out_format)

    ## Save deformation field to mrtrix file. Convert voxel indices to scanner coordinates
    if args.def_mode == 'both':
        # Returned is tuple of two deformation fields, forward and reverse

        for d, mode in zip(D, ('forward', 'reverse')):

            # Create separate filenames
            outfile, f_ext = os.path.splitext(args.output)
            outfile += '-' + mode + f_ext

            comments = [f'squish = {args.squish}', f'lambda = {args.expon}', f'{mode} deformation field']
            # Initialise new Mrtrix3 image with voxel size, transform, and algorithm comments
            if isinstance(img['full'], Image):
                out = Image.empty_as(img['full'])
                out.comments.extend(comments)
            else:
                out = Image(vox=img['voxdims'], transform=img['transform'], comments=comments)
            # Save deformation field data
            out.data = (np.hstack((out.vox * d, np.ones((max(d.shape), 1))))
                        @ out.transform.T)[:,:3].reshape(*img['data'].shape, 3)
            save_mrtrix(outfile, out)

    else:
        comments = [f'squish = {args.squish}', f'lambda = {args.expon}', f'{args.def_mode} deformation field']
        # Initialise new Mrtrix3 image with voxel size, transform, and algorithm comments
        if isinstance(img['full'], Image):
            out = Image.empty_as(img['full'])
            out.comments.extend(comments)
        else:
            out = Image(vox=img['voxdims'], transform=img['transform'], comments=comments)
        # Save deformation field data

        out.data = (np.hstack((out.vox * D, np.ones((max(D.shape), 1))))
                    @ out.transform.T)[:,:3].reshape(*img['data'].shape, 3)
        save_mrtrix(args.output, out)

if __name__ == '__main__':
    main()

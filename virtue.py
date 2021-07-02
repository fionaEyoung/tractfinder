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

def grow(image, tumour_mask, brain_mask, lookup=None):

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


    # isosurface from tumour mask
    # isosurface from brain mask

    # surface centroids and other parameters

    # Dh lookup table
    # Dt lookup table

    # Compute Dh and Dt for each voxel using lookup tables

    # compute k for each voxel

    # P_new = P + e*k*Dt * const

    # Compute deformed image using interpolation

    return

def ang(azA, polA, azB, polB):
    return np.arccos( sin(polA)*sin(polB)*cos(azA-azB) + cos(polA)*cos(polB) )

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

    return P.parse_args()

def main():

    args = parse_args(sys.argv[1:])

    # Extract image data arrays
    img, dat = load_generic(args.input)
    _, tumour_vol = load_generic(args.tumour)
    _, brain_vol = load_generic(args.brain)
    # TODO check dimensions on images
    if not brain_vol.shape==dat.shape[:3]:
        print(brain_vol.shape)
        print(dat.shape[:3])
        raise ValueError("Brain mask and image must have same voxel space")
    if not tumour_vol.shape==dat.shape[:3]:
        raise ValueError("Tumour mask and image must have same voxel space")

    # Voxel grids
    X, Y, Z = np.meshgrid(*[np.arange(i) for i in dat.shape[:3]],
                          copy=False, indexing='ij')
    # Vector of voxel coordinates
    P = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    # Image dimensions
    w, l, h = dat.shape[:3]

    # Largest tumour component
    tumour_modif, S = simplify_vol(tumour_vol)

    # Brain surface and face centroids
    Fb, Vb, Cb = surf_from_vol(brain_vol, sigma=2, target_nfaces=3000)

    # Tumour surface and face centroids
    Ft, Vt, Ct = surf_from_vol(tumour_modif, sigma=2, target_nfaces=900)

    # cpos = [(0.4, -0.07, -0.31), (0.05, -0.13, -0.06), (-0.1, 1, 0.08)]
    # dargs = dict(show_edges=True, color=True)
    # p = pv.Plotter(shape=(1, 2))
    # p.add_mesh(meshb, **dargs)
    # p.add_text("Original", font_size=24)
    # p.camera_position = cpos
    # p.reset_camera()
    # p.subplot(0, 1)
    # p.add_mesh(decimated, **dargs)
    # p.camera_position = cpos
    # p.reset_camera()
    # p.link_views()
    # p.show()

    if True:
        # Display resulting triangular mesh using Matplotlib. This can also be done
        # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        b_mesh = Poly3DCollection(Vb[Fb], alpha=0, edgecolors='k', linewidths=0.1)
        ax1.add_collection3d(b_mesh)
        t_mesh = Poly3DCollection(Vt[Ft], alpha=0.5, edgecolors='r')
        ax1.add_collection3d(t_mesh)
        ax1.scatter(Cb[:,0], Cb[:,1], Cb[:,2], marker='x', s=0.5)

        ax1.set_xlabel("x LR")
        ax1.set_ylabel("y AP")
        ax1.set_zlabel("z IS")

        ax1.set_xlim(0, w)  # a = 6 (times two for 2nd ellipsoid)
        ax1.set_ylim(0, l)  # b = 10
        ax1.set_zlim(0, h)  # c = 16

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()

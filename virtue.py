# VIRtual TUmour Expansion
import argparse
import sys
import numpy as np
import nibabel as nib
import skimage
from skimage import measure
from mrtrix3.io import load_mrtrix, save_mrtrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def grow(image, tumour_mask, brain_mask,
         lookup=None):

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

def get_surf(vol):
    """
    Get triangulated surface object from binary volume
    """
    verts, faces, normals, values = measure.marching_cubes(vol, 0)


def parse_args(args):

    P = argparse.ArgumentParser()
    P.add_argument('input', help="input filename to be deformed")
    P.add_argument('output', help="output filename for deformed image")
    P.add_argument('--tumour', help="tumour mask image")
    P.add_argument('--brain', help="brain mask image")

    return P.parse_args()


def main():

    args = parse_args(sys.argv[1:])
    if args.input.endswith('.mif'):
        img = load_mrtrix(args.input)
        dat = img.data
    elif args.input.endswith('.nii') or args.input.endswith('.nii.gz'):
        img = nib.load(args.input)
        dat = img.get_fdata()
    else:
        raise IOError("File extension not supported: "+args[0])

    print(dat.shape)

    brain_vol = load_mrtrix(args.brain).data
    verts, faces, normals, values = measure.marching_cubes(brain_vol, 0)
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 96)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 96)  # b = 10
    ax.set_zlim(0, 60)  # c = 16

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

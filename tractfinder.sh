# Copyright (c) 2023 Fiona Young
#
# This shell script is a simple demo of the basic tractfinder pipeline for a single tract, including MNI registration.
# Before running, edit the path and setup variables below.

## !---- Configure your setup variables ----!

# Tract (OR, CST, ...)
tract=OR
# Hemisphere (left or right)
h=left

# Fibre orientation distribution image (FOD) from CSD
FOD=/path/to/WM_CSD_FOD.mif
# Subject T1 structural imgae (only needed for registration)
T1=/path/to/subject/T1.nii.gz
# The tract orientation atlas in MNI space
MNI_ATLAS=/path/to/${tract}_${h}_tod_atlas.mif
# The template MNI image (only needed for registration)
MNI=/path/to/MNI_template.nii.gz
# Where to store the results
OUT_DIR=/path/to/output_directory

# (for binary segmentation only) Set threshold
thresh=0.05

## ---------------------------------------- !

## Shorthand function for computing the inner product of two ODF images
IP() {
  mrcalc -quiet $1 $2 -mult -|mrmath -quiet - sum -axis 3 $3
}

## MNI registration
# skip this if you already have an affine transform from MNI->subject
# This also assumes that your T1 and diffusion data are coregistered!

# NOTE: this command uses 9 degrees of freedom, i.e. linear registration
# (translation, rotation, scaling). To use full affine including shearing, use -dof 12
flirt -in $MNI -ref $T1 -dof 9 -omat $OUT_DIR/t_mni_2_t1_fsl.txt -out $OUT_DIR/OUT_DIR mnireg.nii.gz
transformconvert $OUT_DIR/t_mni_2_t1_fsl.txt $MNI $T1 flirt_import $OUT_DIR/t_mni_2_t1_mr.txt -quiet

## Tractfinder

mrtransform -linear $OUT_DIR/t_mni_2_t1_mr.txt -reorient_fod yes \
    $MNI_ATLAS \
    $OUT_DIR/$(basename $MNI_ATLAS) \
    -template $FOD -quiet

IP $OUT_DIR/$(basename $MNI_ATLAS) $FOD $OUT_DIR/${tract}_${h}_tractmap.mif
# If you prefer to run the individual commands from the command line, it would look like this:
# mrcalc $OUT_DIR/$(basename $MNI_ATLAS) $FOD -mult -quiet - | mrmath - sum -axis 3 $OUT_DIR/${tract}_${h}_tractmap.mif -quiet

# To get a binary mask:
mrthreshold $OUT_DIR/${tract}_${h}_tractmap.mif -abs $thresh $OUT_DIR/${tract}_${h}_segmentation.mif


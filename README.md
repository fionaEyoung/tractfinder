# Tractfinder

A simple tract segmentation technique.

## Getting started

### Tract atlases

To use tractfinder, you'll need to get your hands on a set of TOD atlases! Contact fiona.young.15@ucl.ac.uk to get one, or download from somewhere online when they become publically available.
You can also make your own atlases pretty easily! All you need is a dataset of streamlines, curated to your satisfaction, for at least 15 training subjects. Steps for creating custom atlases will be added soon.

### Pipeline

The basic pipeline is outlined in the script `tractfinder.sh`. Edit the path variables within appropriately and run as `sh tractfinder.sh`.
You can also inspect the file and run the corresponding commands directly from the command line (there are only 5 steps at most).

A more user-friendly interface is coming soon, I'm sure.

## Dependencies

At a minimum, you'll need an up-to-date installation of [MRtrix3](https://github.com/MRtrix3/mrtrix3).

The quick-start script also uses FSL's [`flirt`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to perform MNI registration. If you already have an (affine) registration transform for your data, then this step can be skipped.


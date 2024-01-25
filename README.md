# Tractfinder

A simple tract segmentation technique.

## Installation and setup

### Dependencies

At a minimum, you'll need an up-to-date installation of [MRtrix3](https://github.com/MRtrix3/mrtrix3).

Tractfinder also uses FSL's [`flirt`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to perform MNI registration. If you already have an (affine) registration transform for your data, then this step can be skipped.
Otherwise, an up-to-date installation of [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) is required.

### Installing tractfinder (Unix)

The script `bin/tractfinder` is written as an external module against the MRtrix3 python library.
To install tractfinder, first clone this repository:

```bash
$ git clone https://github.com/fionaEyoung/tractfinder.git
```

Next, link the tractfinder repository with your MRtrix3 installation. See MRtrix's [guide on external modules](https://mrtrix.readthedocs.io/en/latest/tips_and_tricks/external_modules.html) for more information.

You will need the path to the MRtrix3 `build` script, *relative to your tractfinder directory*.

> [!NOTE]
> The following instructions assume an MRtrix3 `build` file exists, which is automatically true for installations of MRtrix3 that have been [built from source](https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html). See [the end of this section](#linking-pre-compiled-mrtrix3) if you downloaded MRtrix3 as a [pre-compiled package](https://www.mrtrix.org/download/).

Assuming your `mrtrix3` and `tractfinder` directories are in the same location, option 1 is to create a symbolic link:

```bash
cd tractfinder
ln -s ../mrtrix3/build
```

alternatively, using a text file:

```bash
cd tractfinder
echo ../mrtrix3/build > build
```

#### Linking pre-compiled MRtrix3

If you downloaded MRtrix3 directly as a pre-compiled package from the website, rather than building from source, then you will need to create a dummy `build` file:

1. Locate your MRtrix3 installation. This will likely be in `/usr/local/mrtrix` for a Unix system.
2. Create an empty file called `build` in this location. The contents of the file is irrelevant! You made need admin privilages to create the file, i.e.:
```
sudo touch /usr/local/mrtrix/build
```
3. Follow the linking instructions above, replacing the relevant path with that of the file you just created, e.g.:
```bash
cd tractfinder
echo /usr/local/mrtrix/build > build
```

### Configuring path

You can always invoke the tractfinder script as `<path/to>/tractfinder/bin/tractfinder`. In order to enable invoking the command using `tractfinder` alone, add it's `bin` directory to your path (replace `<path/to>` with the appropriate location, e.g. `~` if you have installed tractfinder in your home directory):

```bash
export PATH=$PATH:<path/to>/tractfinder/bin
```

## User guide

### Tract atlases

To use tractfinder, you'll need at least one tract orientation atlas! Atlases for the corticospinal tract, arcuate fasciculus, optic radiation and inferior fronto-occipital fasciculus are available here: https://zenodo.org/records/10149873
You can also make your own atlases pretty easily! All you need is a dataset of streamlines, curated to your satisfaction, for at least 15 training subjects. Steps for creating custom atlases will be added soon.

### Basic usage

Tractfinder operates on white matter fibre orientation distribution (FOD) obtained using [constrained spherical deconvolution](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/constrained_spherical_deconvolution.html).
[Multi-shell, multi-tissue CSD](https://mrtrix.readthedocs.io/en/latest/constrained_spherical_deconvolution/multi_shell_multi_tissue_csd.html) is recommended where possible.

There are two basic modes of using tractfinder.
You can either specify pairs of atlas and corresponding output images, each corresponding to a single tract to be mapped, or a directory containing
atlases and an output directory (which doesn't have to already exist).
In the latter case, any valid SH image in atlas directory will be mapped to the input image and the result will be stored in the output directory with a name matching the corresponding atlas, plus a suffix.

Tractfinder involves alignment of an atlas in template space with the target image.
To this end, you **must** either provide an affine transformation ([in MRtrix3 format](https://mrtrix.readthedocs.io/en/latest/reference/commands/transformconvert.html)) using the `-transform` option or a pair of structural images (in template and subject space respectively) using the `-struct` option.
> [!NOTE]
> the second argument provided to `-struct` is assumed to be co-registered with the corresponding diffusion space (i.e. the input FOD image).

#### Examples

Map a single tract, represented in the atlas file `CST_left.mif` to the FOD image `wm_fod.mif`. Affine transform from template space to subject space is provided in `t_mni_2_subject.txt`

```bash
tractfinder wm_fod.mif CST_left.mif CST_left_tractmap.nii.gz -transform t_mni_2_subject.txt
```

As above, but registration has not been pre-computed. In this case, supply a structural image "`T1w.nii.gz`" in subject space (i.e., already aligned with `wm_fod.mif`)

```bash
tractfinder wm_fod.mif CST_left.mif CST_left_tractmap.nii.gz -struct MNI152.nii.gz T1w.nii.gz
```

To map multiple tracts, you can supply each atlas and output file individually:

```bash
tractfinder wm_fod.mif CST_left.mif CST_left_tractmap.nii.gz AF_right.mif AF_right_tractmap.nii.gz -transform t_mni_2_subject.txt
```

Or specify a folder containing all atlases, and an output path to store the results:

```bash
tractfinder wm_fod.mif tract_atlases tractfinder_output -transform t_mni_2_subject.txt
```

Assuming the following contents for `tract_atlases`
```
tract_atlases/
|-- CST_left.mif
|-- CST_right.mif
|-- AF_right.mif
```

the result will be
```
tractfinder_output/
|-- CST_left_tractmap.mif
|-- CST_right_tractmap.mif
|-- AF_right_tractmap.mif
```

(Note: the `_tractmap` is the default suffix. It can be changed using the `-suffix` option.)

### Tumour deformation

The tumour deformation functionality is currently under development, and may not be stable.
To use it, checkout the development (`dev`) branch:

```bash
cd tractfinder
git checkout dev
```

Full user guide [here](https://github.com/fionaEyoung/tractfinder/tree/dev#tumour-deformation-modelling).

### Shell script [legacy]

The basic pipeline is also outlined in the script `tractfinder.sh`. Edit the path variables within appropriately and run as `sh tractfinder.sh`.
You can also inspect the file and run the corresponding commands directly from the command line (there are only 5 steps at most).

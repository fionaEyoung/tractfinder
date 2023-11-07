# Tractfinder

A simple tract segmentation technique.

## Installation and setup

### Dependencies

At a minimum, you'll need an up-to-date installation of [MRtrix3](https://github.com/MRtrix3/mrtrix3).

Tractfinder also uses FSL's [`flirt`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) to perform MNI registration. If you already have an (affine) registration transform for your data, then this step can be skipped.
Otherwise, an up-to-date installation of [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) is required.

### Installing tractfinder

The script `bin/tractfinder` is written as an external module against the MRtrix3 python library.
To install tractfinder, first clone this repository:

```bash
$ git clone https://github.com/fionaEyoung/tractfinder.git
```

Next, link the tractfinder repository with your MRtrix3 installation. See MRtrix's [guide on external modules](https://mrtrix.readthedocs.io/en/latest/tips_and_tricks/external_modules.html) for more information.

You will need the path to the MRtrix3 `build` script, *relative to your tractfinder directory*. Assuming your `mrtrix3` and `tractfinder` directories are in the same location, option 1 is to create a symbolic link:

```bash
cd tractfinder
ln -s ../mrtrix3/build
```

alternatively, using a text file:

```bash
cd tractfinder
echo ../mrtrix3/build > build
```

### Configuring path

You can always invoke the tractfinder script as `<path/to>/tractfinder/bin/tractfinder`. In order to enable invoking the command using `tractfinder` alone, add it's `bin` directory to your path (replace `<path/to>` with the appropriate location, e.g. `~` if you have installed tractfinder in your home directory):

```bash
export PATH=$PATH:<path/to>/tractfinder/bin
```

## Getting started

### Tract atlases

To use tractfinder, you'll need to get your hands on a set of TOD atlases! Contact fiona.young.15@ucl.ac.uk to get one, or download from somewhere online when they become publicly available.
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
Note: the second argument provided to `-struct` is assumed to be co-registered with the corresponding diffusion space (i.e. the input FOD image).

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

## Tumour deformation modelling

> [!NOTE]
> This functionality is still under development, and may not run smoothly!
> It has not been extensively edge-tested, so please report any issues :)

The deformation algorithm is described in [Young _et al._ (2022)](https://doi.org/10.1007%2Fs11548-022-02617-z) and is an extension of the one described by [Nowinski & Belov (2005)](https://doi.org/10.1016/j.acra.2005.04.018).

Tumour deformation modelling can be run in addition to the normal usage described above.
The additional options required are

* `-tumour image`: A segmentation mask of the lesions
* `-brain_mask image` OR `-struct image image`: A brain mask is needed to compute the deformation field. You can provide one manually, or allow the program to generate a brain mask based on the structural image and input FOD image.

Tumour model computational time should be around <1min, with peak RAM usage at around ~9GB.
After that, the deformation field will be used to transform and reorient the TOD atlas before mapping to the subject FOD.
This step can take an additional 1-2 minutes per atlas.

### Advanced usage

Tractfinder's radial tumour deformation model maps a point $P$ in the original image to new position $P'$ according to:

$$ P' = f(P) = P + \mathbf{\hat{e}}k(P)D_ts $$

The program will by default use an exponential deformation factor

$$ k(P) = (1-c)e^{-\lambda \frac{D_p}{D_b}} +c $$

with $\lambda$ set dynamically to the maximum possible value (see [Young _et al._ (2022)](https://doi.org/10.1007%2Fs11548-022-02617-z) for details).
To choose an alternative use the `-k` option:

* `-k exponential_constant`: use a single value for $\lambda$ throughout the entire brain (requires `-l value`)
* `-k linear`: use a linear deformation model. This is equivalent to the algorithm in [Nowinski & Belov (2005)](https://doi.org/10.1016/j.acra.2005.04.018).

The `-scale value` option controls the scale factor $s$. Setting $0 < s < 1$ is useful for modelling a partially shrunken or resected tumour using a prior segmentation, or for tumours with a partially infiltrating boundary, as it effectively scales the tumour radius.

The most intensive part of the program is the calculation of lookup matrices for $D_t$ and $D_b$.
However, these depend only on the brain and tumour segmentations, not on the deformation model parameters.
If computing multiple deformation fields (e.g. trying out different parameters), then these matrices can be stored and reused using `-distance_lookup directory`.

Finally, it's possible to only compute the deformation field, without subsequent atlas registration, deformation and mapping.
Use `-deformation_only` and `-deformation_field image` to store the output field.
Unfortunately, for now the positional arguments are still necessary to appease Argparse, but you can be lazy and use

```
tractfinder _ _ _ -deformation_only -tumour tumour.nii.gz -brain_mask brain.nii.gz -deformation_field D.nii.gz
```

Just be aware that if relying on the program to automatically generate a brain mask, you will still need to provide the first positional argument:

```
tractfinder fod.mif _ _ -deformation_only -tumour tumour.nii.gz -struct mni.nii.gz t1w.nii.gz -deformation_field D.nii.gz
```

Internally, the functionality to output both forward and reverse deformation fields is available, but this still needs to be exposed to the user interface if there is demand.
(The forward deformation field would be required e.g. for transforming streamlines.)

## Shell script [legacy]

The basic pipeline is also outlined in the script `tractfinder.sh`. Edit the path variables within appropriately and run as `sh tractfinder.sh`.
You can also inspect the file and run the corresponding commands directly from the command line (there are only 5 steps at most).

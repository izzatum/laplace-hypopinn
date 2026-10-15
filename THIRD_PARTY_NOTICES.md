# License scope and third-party notices

The [MIT License](LICENSE) covers the maintained implementation in `src/`, tests,
scripts, configurations, new examples and tutorials, and project documentation.
It does **not** relicense the historical material in `archive/`, `otway2.csv`,
or third-party dependencies. The MIT metadata in the Python distribution refers
to the maintained package; the archive and velocity dataset are not packaged.

## Archived curvature library

`archive/legacy/curvature/` derives from **DLR-RM/curvature**, by Matthias Humt and
Jongseok Lee, under the **GNU General Public License, version 3**. Its license is
preserved in [DLR-Curvature-GPL-3.0.txt](LICENSES/DLR-Curvature-GPL-3.0.txt).
Reference source and license:
[revision 164d3ac782a34e2cb1b563698843d33b8b272ce0](https://github.com/DLR-RM/curvature/tree/164d3ac782a34e2cb1b563698843d33b8b272ce0),
[upstream LICENCE](https://github.com/DLR-RM/curvature/blob/164d3ac782a34e2cb1b563698843d33b8b272ce0/LICENCE).

The archived `__init__.py`, `datasets.py`, `imagenet.py`, `lenet5.py`, `resnet.py`
and `lenet5_mnist.pth` match that upstream revision byte-for-byte. Two files
contain historical changes already present in this repository's September 2022
upload: `curvatures.py` replaces `register_backward_hook` with
`register_full_backward_hook`; `utils.py` replaces four `torch.symeig` calls with
`torch.linalg.eigvalsh`/`torch.linalg.eigh`. Archiving did not change these files;
[the manifest](archive/manifest.json) records their preserved hashes.

The maintained package does not import or distribute this archived library in its
Python wheel or source distribution. The archived library's GPL terms continue
to apply when it is used or redistributed.

The following notices cover components adapted by, or explicitly credited in,
the archived library. References identify inspected upstream versions, without
claiming that each version was the exact original copy source.

| Component and attribution | Archived use | Retained license |
| --- | --- | --- |
| EKFAC-pytorch — Copyright 2018 Thomas George, César Laurent and Université de Montréal | `curvatures.py`: KFAC adaptation | [MIT](LICENSES/EKFAC-MIT.txt) |
| SWAG — Copyright 2019 Wesley Maddox, Timur Garipov, Pavel Izmailov, Dmitry Vetrov and Andrew Gordon Wilson | `curvatures.py`: credited inspiration for Diagonal | [BSD-2-Clause](LICENSES/SWAG-BSD-2-Clause.txt) |
| Detectron2 — Copyright Facebook, Inc. and its affiliates | `imagenet.py`: `MapDataset` and `DatasetFromList` | [Apache-2.0](LICENSES/Detectron2-Apache-2.0.txt) |
| torchvision — Copyright Soumith Chintala 2016 | `resnet.py`: ResNet helpers and classes | [BSD-3-Clause](LICENSES/Torchvision-BSD-3-Clause.txt) |
| joblib — Copyright 2008–2016 The joblib developers | `imagenet.py`: credited wrapper source | [BSD-3-Clause](LICENSES/Joblib-BSD-3-Clause.txt) |
| loky — Copyright 2017 Olivier Grisel and Thomas Moreau | `imagenet.py`: simplified `PicklableWrapper` | [BSD-3-Clause](LICENSES/Loky-BSD-3-Clause.txt) |
| PyTorch tutorials — Copyright 2017 Pytorch contributors | `datasets.py`: FGSM adaptation | [BSD-3-Clause](LICENSES/PyTorch-Tutorials-BSD-3-Clause.txt) |

Immutable upstream license references:

- [EKFAC-pytorch](https://github.com/Thrandis/EKFAC-pytorch/blob/8d3f0590358e1888f65cea98836b487f6719dbf6/LICENSE)
- [SWAG](https://github.com/wjmaddox/swa_gaussian/blob/ed5fd56e34083b42630239e59076952dee44daf4/LICENSE)
- [Detectron2](https://github.com/facebookresearch/detectron2/blob/4841e70ee48da72c32304f9ebf98138c2a70048d/LICENSE)
- [torchvision](https://github.com/pytorch/vision/blob/78ed10cc51067f1a6bac9352831ef37a3f842784/LICENSE)
- [joblib](https://github.com/joblib/joblib/blob/27ccbb82a7025e5d149a235582b861db1987dbb5/LICENSE.txt)
- [loky](https://github.com/joblib/loky/blob/dff10d0199db65d06e745af76d3db7272bf739d8/LICENSE.txt)
- [PyTorch tutorials](https://github.com/pytorch/tutorials/blob/8d0b5cc8dcfe450b6d54556e8ac4c1d8a87cce21/LICENSE)

## Historical notebooks and data

`archive/notebooks/` retains the original source, outputs and attribution. Those
notebooks use the archived GPL curvature library and, for Otway examples, the
bundled velocity data. This MIT addition does not grant new rights to the
historical notebooks, third-party portions or embedded data.

`otway2.csv` has no explicit data-reuse license in the original repository. The
paper describes the Otway velocity model in the context of the CO2CRC Stage 2C
project and cites Glubokovskikh et al. (2016), *Seismic monitoring of CO2
geosequestration: CO2CRC Otway case study using full 4d FDTD approach*. Availability
in this repository does not establish a reuse license; this file is excluded
from the MIT grant. The analytic synthetic examples and maintained tutorials
do not require this dataset.

The article's CC BY 4.0 terms apply separately to the publication and do not
establish a license for archived code or the CSV. Installed dependencies retain
their own licenses. Research citations are listed in [CITATION.cff](CITATION.cff)
and the [method reference](docs/methods.md).

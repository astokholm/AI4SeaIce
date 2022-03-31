# AI4SeaIce

<p align="center">
<a href="https://github.com/astokholm/AI4SeaIce/graphs/contributors">
        <img src="https://img.shields.io/badge/contributors-2-green" /></a>
<a href="https://github.com/astokholm/AI4SeaIce/">
        <img src="https://img.shields.io/badge/version-0.1.0-blue" /></a>
<a href="https://github.com/astokholm/AI4SeaIce/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-green" /></a>
</p>

<details>
<summary>Table of Contents</summary>
<ol>
<li><a href="#about-this-project">About This Project</a></li>
<li><a href="#requireents">Requirements</a></li>
<li><a href="#installation">Installation</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#license">License</a></li>
<li><a href="#contact">Contact</a></li>
<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>
</details>

## About this project

This project contains the code for a U-Net model architecture used for sea ice charting segmentation. It also 
contains the names of files used for training and validation (testing).
New: Loss functions added for ICLR conference paper. (31/04/2022)

<p align="right">(<a href="#top">back to top</a>)</p>

## Requirements
This project uses [Python](https://www.python.org/) 3.9, and requires [PyTorch](https://www.pytorch.org/) 1.10.0 
library and its dependencies.

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation

To install this project either download it as a `.zip` file and extract it into
a desired directory or clone it via the terminal or console command:

* using the HTTPS

```shell
git clone https://github.com/astokholm/AI4SeaIce.git
```

* or SSH

```shell
git clone git@github.com:astokholm/AI4SeaIce.git
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Usage
At its current development stage, the scripts in the repository are not meant to be run as executable code, but to 
provided the details about the architecture and parameters of the models used in sea ice charting experiments (see 
[`models`](./models)). 

The training and validation split used in the experiments is available in [`datalists`](./datalists).

<p align="right">(<a href="#top">back to top</a>)</p>

## License
Distributed
under MIT licence. See [`LICENSE`](./LICENSE) for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## Contributors

* [Andreas Stokholm](https://github.com/astokholm/)
* [Andrzej Kucik](https://github.com/AndrzejKucik/)

<p align="right">(<a href="#top">back to top</a>)</p>

## Contact

E-mail: [astokholm@space.dtu.dk](mailto:astokholm@space.dtu.dk),  [andrzej.kucik@esa.int](mailto:andrzej.kucik@esa.int)

Project Link: <https://github.com/astokholm/AI4SeaIce>

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgements

* [DTU Space](https://www.space.dtu.dk/)
* [ESA &Phi;-Lab](https://philab.phi.esa.int/) <a href="https://github.com/ESA-PhiLab">
        <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" /></a>

<p align="right">(<a href="#top">back to top</a>)</p>

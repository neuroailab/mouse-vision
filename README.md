# Models of Mouse Vision
This repo contains self-supervised, ImageNet pre-trained convolutional network models of mouse visual cortex; the preprocessed neural responses; and training code for these models across a range of self-supervised objective functions, so that others may possibly use them as a basis for modeling other small animal visual systems.

This repository is based on our paper:

**Aran Nayebi\*, Nathan C. L. Kong\*, Chengxu Zhuang, Justin L. Gardner, Anthony M. Norcia, Daniel L. K. Yamins**

["Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation"](https://www.biorxiv.org/content/10.1101/2021.06.16.448730)

*PLOS Computational Biology 2023 (in press)*

## Getting started
It is recommended that you install this repo within a virtual environment (Python 3.6 recommended), and run inferences there.
An example command for doing this with `anaconda` would be:
```
conda create -y -n your_env python=3.6.10 anaconda
```
To install this package and all of its dependecies, clone this repo on your machine and then install it via pip:
1. `git clone https://github.com/neuroailab/mouse-vision.git` to clone the repository.
2. `cd mouse-vision/`
3. `conda activate your_env`
4. Run `pip install -e .` to install the current version.

## Available Pre-trained Models
To get the saved checkpoints of the models, simply run this script:
```
./get_checkpoints.sh
```
This will save them to the current directory in the folder `./model_ckpts/`.
If you want a subset of the models, feel free to modify the for loop in the above bash script.

## Training Code

## Neural Responses
To download the preprocessed Allen Institute Neuropixels and Calcium Imaging datasets, simply run this script:
```
./get_neural_data.sh
```
This will save the data to the current directory in the folder `./neural_data/`.
If you want a subset of the neural datasets, feel free to modify the for loop in the above bash script.
Despite it being the larger of the two datasets in terms of size, we strongly recommend working with the newer Neuropixels dataset (`mouse_neuropixels_visual_data_with_reliabilities.pkl`).
You can load the corresponding neural dataset with the command:
```
from mouse_vision.core.utils import open_dataset
d = open_dataset(/path/to/file.pkl)
```
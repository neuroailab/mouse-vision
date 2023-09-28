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
To get the saved checkpoints of the models, simply run this bash script:
```
./get_checkpoints.sh
```
This will save them to the current directory in the folder `./model_ckpts/`.
If you want a subset of the models, feel free to modify the for loop in the above bash script.

Models are named according to the convention of `[architecture]_[lossfunction]`, all of which are trained on the ImageNet dataset and described in [our paper](https://www.biorxiv.org/content/10.1101/2021.06.16.448730).
You can see [this notebook](https://github.com/neuroailab/mouse-vision/blob/main/Loading%20model%20weights.ipynb) for an example of loading pre-trained models.

Some models may be better suited than others based on your needs, but we recommend: 
- `alexnet_bn_ir`, which is overall our best predictive model of mouse visual cortical responses (specifically, the first four layers: `features.3`, `features.7`, `features.10`, and `features.13`, as can be seen in Figure 2B).
*We highly recommend this model for general purpose use.*
It is an AlexNet architecture trained with the Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `dual_stream_ir`, which is our second best predictive model and models the skip connections of mouse visual cortex, consisting of two streams.
It is trained with the Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `six_stream_simclr`, which is our best predictive six stream architecture, and is trained with the SimCLR objective on `64x64`-pixel ImageNet inputs.
- `shi_mousenet_ir`, which is the Shi *et al.* 2020 MouseNet architecture that attempts to map the details of the mouse connectome onto a CNN architecture, and is trained with an Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `shi_mousenet_vispor5_ir`, which is the same as `shi_mousenet_ir`, but where the final loss layer reads off of the penultimate layer (`VISpor5`) of the model, rather than the concatenation of the earlier layers as originally proposed.

## Training Code
Download ImageNet and then run under `mouse_vision/model_training/`:
```
CUDA_VISIBLE_DEVICES=[gpu_id] python run_trainer.py --config=[]
```
The loss functions available are implemented in the `mouse_vision/loss_functions/` [directory](https://github.com/neuroailab/mouse-vision/tree/main/mouse_vision/loss_functions), and include self-supervised loss functions such as: Instance Recognition, SimCLR, SimSiam, VICReg, BarlowTwins, MoCov2, RotNet, RelativeLocation, and AutoEncoding; along with supervised loss functions such as: Depth Prediction and CrossEntropy (for categorization).

For example, to train our best model overall (`alexnet_bn_ir`), you can run this command:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --config=configs/ir/alexnet_bn_ir.json
```
Note that you will have to modify the `save_prefix` key in the json file to the directory that you want to save your checkpoints.


## Neural Responses
To download the preprocessed Allen Institute Neuropixels and Calcium Imaging datasets, simply run this bash script:
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
You can see [this notebook](https://github.com/neuroailab/mouse-vision/blob/main/Loading%20neural%20data.ipynb) for an example of loading and interacting with the neural data.
If you prefer to load the the neural responses directly from the Allen SDK, you can refer to the `mouse_vision/neural_data` [directory](https://github.com/neuroailab/mouse-vision/tree/main/mouse_vision/neural_data).

## Cite
If you used this codebase for your research, please consider citing our paper:
```
@article{NayebiKong2023mouse,
  title={Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation},
  author={Nayebi, Aran* and Kong, NC* and Zhuang, Chengxu and Gardner, Justin L and Norcia, Anthony M and Yamins, DL},
  journal={PLOS Computational Biology},
  year={2023}
}
```

## Contact
If you have any questions or encounter issues, either submit a Github issue here or email `anayebi@mit.edu` and `nclkong@mit.edu`.

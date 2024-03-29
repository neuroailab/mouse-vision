# Models of Mouse Vision
This repo contains self-supervised, ImageNet pre-trained convolutional network PyTorch models of mouse visual cortex; the preprocessed neural responses; and training code for these models across a range of self-supervised objective functions, so that others may possibly use them as a basis for modeling other small animal visual systems.

Since our models are stimulus-computable, they can be applied to new visual stimuli without modification and generate predictions for further neural and behavioral experiments of your design.

This repository is based on our paper:

**Aran Nayebi\*, Nathan C. L. Kong\*, Chengxu Zhuang, Justin L. Gardner, Anthony M. Norcia, Daniel L. K. Yamins**

["Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation"](https://www.biorxiv.org/content/10.1101/2021.06.16.448730)

*PLOS Computational Biology* 19(10): e1011506. https://doi.org/10.1371/journal.pcbi.1011506

Here's a [video recording](https://www.youtube.com/watch?v=9h_3bHVDMhA&t=650s) that explains our work a bit.

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
*You can see [this notebook](https://github.com/neuroailab/mouse-vision/blob/main/Loading%20model%20weights.ipynb) for an example of loading pre-trained models.*

Some models may be better suited than others based on your needs, but we recommend: 
- `alexnet_bn_ir`, which is overall our best predictive model of mouse visual cortical responses. *We highly recommend this model for general purpose use.* Specifically, we recommend the [first four layers](https://github.com/neuroailab/mouse-vision/blob/main/mouse_vision/models/model_layers.py#L6-L10): `features.3`, `features.7`, `features.10`, and `features.13`, as can be seen in Figure 2B. It is an AlexNet architecture trained with the Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `dual_stream_ir`, which is our second best predictive model and models the skip connections of mouse visual cortex, consisting of two streams.
It is trained with the Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `six_stream_simclr`, which is our best predictive six stream architecture, and is trained with the SimCLR objective on `64x64`-pixel ImageNet inputs.
- `shi_mousenet_ir`, which is the [Shi *et al.* 2020 MouseNet](https://doi.org/10.1371/journal.pcbi.1010427) architecture that attempts to map the details of the mouse connectome onto a CNN architecture, and is trained with an Instance Recognition objective on `64x64`-pixel ImageNet inputs.
- `shi_mousenet_vispor5_ir`, which is the same as `shi_mousenet_ir`, but where the final loss layer reads off of the penultimate layer (`VISpor5`) of the model, rather than the concatenation of the earlier layers as originally proposed.
We thought this might aid the original [Shi *et al.* 2020 MouseNet](https://doi.org/10.1371/journal.pcbi.1010427)’s task performance and neural predictivity, since it can be difficult to train linear layers when the input dimensionality is very large.
We found that this model better predicted the mouse visual cortex Neuropixels responses than `shi_mousenet_ir`.

## Training Code
Download ImageNet (or your image dataset of choice) and then run under `mouse_vision/model_training/`:
```
python run_trainer.py --config=[]
```
Specify the `gpu_id` in the config. The loss functions available are implemented in the `mouse_vision/loss_functions/` [directory](https://github.com/neuroailab/mouse-vision/tree/main/mouse_vision/loss_functions), and include self-supervised loss functions such as: Instance Recognition, SimCLR, SimSiam, VICReg, BarlowTwins, MoCov2, RotNet, RelativeLocation, and AutoEncoding; along with supervised loss functions such as: Depth Prediction and CrossEntropy (for categorization).

Model architectures are implemented in the `mouse_vision/models/` [directory](https://github.com/neuroailab/mouse-vision/tree/main/mouse_vision/models), and range from multi-stream models (our custom parallel stream models and the [Shi *et al.* 2020 MouseNet](https://doi.org/10.1371/journal.pcbi.1010427)), multi-stream sparse autoencoders, to single stream feedforward networks such as AlexNet, ResNets, etc.

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
*You can see [this notebook](https://github.com/neuroailab/mouse-vision/blob/main/Loading%20neural%20data.ipynb) for an example of loading and interacting with the neural data.*
If you prefer to load the the neural responses directly from the Allen SDK, you can refer to the `mouse_vision/neural_data` [directory](https://github.com/neuroailab/mouse-vision/tree/main/mouse_vision/neural_data).

## Cite
If you used this codebase for your research, please consider citing our paper:
```
@article{NayebiKong2023mouse,
    doi = {10.1371/journal.pcbi.1011506},
    author = {Nayebi*, Aran AND Kong*, Nathan C. L. AND Zhuang, Chengxu AND Gardner, Justin L. AND Norcia, Anthony M. AND Yamins, Daniel L. K.},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {Mouse visual cortex as a limited resource system that self-learns an ecologically-general representation},
    year = {2023},
    month = {10},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pcbi.1011506},
    pages = {1-36},
}
```

## Contact
If you have any questions or encounter issues, either submit a Github issue here or email `anayebi@mit.edu` and `nclkong@mit.edu`.

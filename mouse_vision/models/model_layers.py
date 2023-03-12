"""
PyTorch modules for each model architecture. These are the layers that we will
extract activations from for downstream analyses.
"""

MODEL_LAYERS = {
    # AlexNet
    "alexnet":
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"] +
        ["features.12"] +
        ["classifier.2"] +
        ["classifier.5"],

    "alexnet_64x64_rl_scratch_truncated": 
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"],

    "alexnet_bn":
        ["features.3"] +
        ["features.7"] +
        ["features.10"] +
        ["features.13"] +
        ["features.17"] +
        ["classifier.3"] +
        ["classifier.7"],

    "alexnet_two_64x64":
        ["features.2"] +
        ["features.5"],

    "alexnet_three_64x64":
        ["features.2"] +
        ["features.5"] +
        ["features.7"],

    "alexnet_four_64x64":
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"],

    "alexnet_five_64x64":
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"] +
        ["features.12"],

    "alexnet_six_64x64":
        ["features.2"] +
        ["features.5"] +
        ["features.7"] +
        ["features.9"] +
        ["features.12"] +
        ["classifier.2"],

    # VGGs
    "vgg11":
        ["features.2"] +
        ["features.5"] +
        ["features.10"] +
        ["features.15"] +
        ["features.20"] +
        ["classifier.1"] +
        ["classifier.4"],

    "vgg13":
        ["features.4"] +
        ["features.9"] +
        ["features.14"] +
        ["features.19"] +
        ["features.24"] +
        ["classifier.1"] +
        ["classifier.4"],

    "vgg16":
        ["features.4"] +
        ["features.9"] +
        ["features.16"] +
        ["features.23"] +
        ["features.30"] +
        ["classifier.1"] +
        ["classifier.4"],

    "vgg19":
        ["features.4"] +
        ["features.9"] +
        ["features.18"] +
        ["features.27"] +
        ["features.36"] +
        ["classifier.1"] +
        ["classifier.4"],

    # ResNets
    "resnet18":
        ["relu", "maxpool"] +
        ["layer1.0", "layer1.1"] +
        ["layer2.0", "layer2.1"] +
        ["layer3.0", "layer3.1"] +
        ["layer4.0", "layer4.1"] +
        ["avgpool"],

    "resnet34":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet50":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet101":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(23)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "resnet152":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(8)] +
        [f"layer3.{i}" for i in range(36)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "wide_resnet50_2":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(6)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    "wide_resnet101_2":
        ["relu", "maxpool"] +
        [f"layer1.{i}" for i in range(3)] +
        [f"layer2.{i}" for i in range(4)] +
        [f"layer3.{i}" for i in range(23)] +
        [f"layer4.{i}" for i in range(3)] +
        ["avgpool"],

    # SqueezeNets
    "squeezenet1_0": [
        "features." + layer for layer in
        ["2"] + [f"{i}.expand3x3_activation" for i in [3, 4, 5, 7, 8, 9, 10, 12]]
    ],
    "squeezenet1_1": [
        "features." + layer for layer in
        ["2"] + [f"{i}.expand3x3_activation" for i in [3, 4, 6, 7, 9, 10, 11, 12]]
    ],

    # DenseNets
    "densenet121":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(24)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(16)],

    "densenet161":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(36)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(24)],

    "densenet169":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(32)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(32)],

    "densenet201":
        ["features.pool0"] +
        [f"features.denseblock1.denselayer{i+1}" for i in range(6)] + ["features.transition1.pool"] +
        [f"features.denseblock2.denselayer{i+1}" for i in range(12)] + ["features.transition2.pool"] +
        [f"features.denseblock3.denselayer{i+1}" for i in range(48)] + ["features.transition3.pool"] +
        [f"features.denseblock4.denselayer{i+1}" for i in range(32)],

    # Inceptions
    "googlenet": # Inception v1
        ["maxpool1", "maxpool2"] +
        [f"inception3{i}" for i in ['a', 'b']] +
        ["maxpool3"] +
        [f"inception4{i}" for i in ['a', 'b', 'c', 'd', 'e']] +
        ["maxpool4"] +
        [f"inception5{i}" for i in ['a', 'b']] +
        ["avgpool"],

    "inception_v3":
        #["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "Conv2d_3b_1x1", "Conv2d_4a_3x3"] +
        ["maxpool1", "maxpool2"] +
        [f"Mixed_5{i}" for i in ['b', 'c', 'd']] +
        [f"Mixed_6{i}" for i in ['a', 'b', 'c', 'd', 'e']] +
        [f"Mixed_7{i}" for i in ['a', 'b', 'c']],

    # ShuffleNet
    "shufflenet_v2_x0_5":
        ["maxpool"] +
        [f"stage2.{i}" for i in range(4)] +
        [f"stage3.{i}" for i in range(8)] +
        [f"stage4.{i}" for i in range(4)] +
        ["conv5.2"],

    "shufflenet_v2_x1_0":
        ["maxpool"] +
        [f"stage2.{i}" for i in range(4)] +
        [f"stage3.{i}" for i in range(8)] +
        [f"stage4.{i}" for i in range(4)] +
        ["conv5.2"],

    # MobileNet
    "mobilenet_v2":
        ["features.0.2"] +
        ["features.1.conv.1"] +
        [f"features.{i}.conv.2" for i in range(2,18)] + # output conv of each inverted res block
        ["features.18.2"],

    # MNASNets
    "mnasnet0_5":
        ["layers.2", "layers.5"] +
        [f"layers.8.{i}.layers.6" for i in range(3)] +
        [f"layers.9.{i}.layers.6" for i in range(3)] +
        [f"layers.10.{i}.layers.6" for i in range(3)] +
        [f"layers.11.{i}.layers.6" for i in range(2)] +
        [f"layers.12.{i}.layers.6" for i in range(4)] +
        [f"layers.13.{i}.layers.6" for i in range(1)] +
        ["layers.16"],

    "mnasnet1_0":
        ["layers.2", "layers.5"] +
        [f"layers.8.{i}.layers.6" for i in range(3)] +
        [f"layers.9.{i}.layers.6" for i in range(3)] +
        [f"layers.10.{i}.layers.6" for i in range(3)] +
        [f"layers.11.{i}.layers.6" for i in range(2)] +
        [f"layers.12.{i}.layers.6" for i in range(4)] +
        [f"layers.13.{i}.layers.6" for i in range(1)] +
        ["layers.16"],

    "xception":
        [],

    "nasnetamobile":
        [],

    "shi_mousenet":
        ["LGNv",
         "VISp4",
         "VISp2/3",
         "VISp5",
         "VISal4",
         "VISal2/3",
         "VISal5",
         "VISpl4",
         "VISpl2/3",
         "VISpl5",
         "VISli4",
         "VISli2/3",
         "VISli5",
         "VISrl4",
         "VISrl2/3",
         "VISrl5",
         "VISl4",
         "VISl2/3",
         "VISl5",
         "VISpor4",
         "VISpor2/3",
         "VISpor5"],

    "simplified_mousenet_six_stream": [
        "VISp", "VISal", "VISli", "VISl", "VISrl", "VISpl", "VISpm", "VISpor", "VISam"
    ],

    "simplified_mousenet_dual_stream": [
        "VISp", "ventral", "dorsal", "VISpor", "VISam"
    ],

    "simplified_mousenet_single_stream": [
        "VISp", "ventral", "VISpor"
    ],

}

# Bakhtiari et al. (2021): CPC models
CPC_2P_MODEL_LAYERS = list()
CPC_2P_MODEL_LAYERS.append("backbone.s1.pool_layer")
for path in [1, 2]:
    for block_idx in range(0, 10):
        CPC_2P_MODEL_LAYERS.append(f"backbone.path{path}.res_blocks.res{block_idx}.branch2.a_relu")
        CPC_2P_MODEL_LAYERS.append(f"backbone.path{path}.res_blocks.res{block_idx}.branch2.b_relu")
        CPC_2P_MODEL_LAYERS.append(f"backbone.path{path}.res_blocks.res{block_idx}.relu")
CPC_2P_MODEL_LAYERS.append("relu1")  # after features from both paths are concatenated, avg pooled and relu
MODEL_LAYERS["monkeynet_2p_cpc"] = CPC_2P_MODEL_LAYERS

# AlexNet variants
MODEL_LAYERS["alexnet_64x64_input_pool_6"] = MODEL_LAYERS["alexnet"]
MODEL_LAYERS["alexnet_64x64_input_pool_6_with_ir_transforms"] = MODEL_LAYERS["alexnet"]
MODEL_LAYERS["alexnet_bn_64x64_input_pool_6_with_ir_transforms"] = MODEL_LAYERS["alexnet"]
MODEL_LAYERS["alexnet_64x64_input_pool_1"] = MODEL_LAYERS["alexnet"]
MODEL_LAYERS["alexnet_64x64_input_dict"] = ["pool1", "pool2", "conv3", "conv4", "pool5", "fc1", "fc2"]

MODEL_LAYERS["alexnet_64x64_input_pool_6_cifar10"] = MODEL_LAYERS["alexnet"]

# AlexNet BN variants
MODEL_LAYERS["alexnet_bn_mocov2_64x64"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_bn_simclr_64x64"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_bn_simsiam_64x64"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_dmlocomotion"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_bn_ir_64x64_input_pool_6"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_224x224"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_84x84"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_104x104"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_124x124"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_144x144"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_164x164"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_184x184"] = MODEL_LAYERS["alexnet_bn"]
MODEL_LAYERS["alexnet_ir_204x204"] = MODEL_LAYERS["alexnet_bn"]

# VGG16 variants
MODEL_LAYERS["vgg16_64x64_input"] = MODEL_LAYERS["vgg16"]
MODEL_LAYERS["vgg16_64x64_input_cifar10"] = MODEL_LAYERS["vgg16"]
MODEL_LAYERS["vgg16_ir_64x64"] = MODEL_LAYERS["vgg16"]

# ResNet18 variants
MODEL_LAYERS["resnet18_64x64_input"] = MODEL_LAYERS["resnet18"]

MODEL_LAYERS["resnet18_64x64_input_cifar10"] = MODEL_LAYERS["resnet18"]

MODEL_LAYERS["resnet18_ir_64x64"] = MODEL_LAYERS["resnet18"]
MODEL_LAYERS["resnet18_ir_preavgpool_64x64"] = MODEL_LAYERS["resnet18"]

# Other ResNet variants
MODEL_LAYERS["resnet34_64x64_input"] = MODEL_LAYERS["resnet34"]
MODEL_LAYERS["resnet34_ir_64x64"] = MODEL_LAYERS["resnet34"]
MODEL_LAYERS["resnet34_ir_preavgpool_64x64"] = MODEL_LAYERS["resnet34_ir_64x64"]
MODEL_LAYERS["resnet50_64x64_input"] = MODEL_LAYERS["resnet50"]
MODEL_LAYERS["resnet50_ir_64x64"] = MODEL_LAYERS["resnet50"]
MODEL_LAYERS["resnet50_ir_preavgpool_64x64"] = MODEL_LAYERS["resnet50_ir_64x64"]
MODEL_LAYERS["resnet101_64x64_input"] = MODEL_LAYERS["resnet101"]
MODEL_LAYERS["resnet101_ir_64x64"] = MODEL_LAYERS["resnet101"]
MODEL_LAYERS["resnet101_ir_preavgpool_64x64"] = MODEL_LAYERS["resnet101_ir_64x64"]
MODEL_LAYERS["resnet152_64x64_input"] = MODEL_LAYERS["resnet152"]
MODEL_LAYERS["resnet152_ir_64x64"] = MODEL_LAYERS["resnet152"]
MODEL_LAYERS["resnet152_ir_preavgpool_64x64"] = MODEL_LAYERS["resnet152_ir_64x64"]

# Shi MouseNet variants
MODEL_LAYERS["shi_mousenet_vispor5"] = MODEL_LAYERS["shi_mousenet"]
MODEL_LAYERS["shi_mousenet_vispor5_pool_4"] = MODEL_LAYERS["shi_mousenet"]

MODEL_LAYERS["shi_mousenet_cifar10"] = MODEL_LAYERS["shi_mousenet"]
MODEL_LAYERS["shi_mousenet_vispor5_cifar10"] = MODEL_LAYERS["shi_mousenet"]

## Parallel Stream MouseNet variants
# intermediate image resolution models
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_32x32"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_44x44"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_84x84"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_104x104"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_124x124"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_144x144"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_164x164"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_184x184"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_204x204"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]

# 224 px models
MODEL_LAYERS["simplified_mousenet_single_stream_ir_224x224"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir_224x224"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_ir_224x224"] = MODEL_LAYERS["simplified_mousenet_six_stream"]

# 64 px models
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_bn"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_vispor_only"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_vispor_only_visp_3x3"] = MODEL_LAYERS["simplified_mousenet_six_stream"]

MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_bn"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_vispor_only"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_vispor_only_visp_3x3"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]

MODEL_LAYERS["simplified_mousenet_six_stream_cifar10"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_cifar10"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_vispor_only_cifar10"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_vispor_only_visp_3x3_cifar10"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_bn_cifar10"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_ir"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_rotnet"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_simclr"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_mocov2"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_six_stream_visp_3x3_simsiam"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_ae_six_stream"] = MODEL_LAYERS["simplified_mousenet_six_stream"]
MODEL_LAYERS["simplified_mousenet_depth_hour_glass_six_stream"] = MODEL_LAYERS["simplified_mousenet_six_stream"]

MODEL_LAYERS["simplified_mousenet_dual_stream_cifar10"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_cifar10"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_vispor_only_cifar10"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_vispor_only_visp_3x3_cifar10"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_bn_cifar10"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_ir"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_rotnet"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_simclr"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_mocov2"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_dual_stream_visp_3x3_simsiam"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_ae_dual_stream"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]
MODEL_LAYERS["simplified_mousenet_depth_hour_glass_dual_stream"] = MODEL_LAYERS["simplified_mousenet_dual_stream"]

MODEL_LAYERS["simplified_mousenet_single_stream_cifar10"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_single_stream_ir"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_single_stream_rotnet"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_single_stream_simclr"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_single_stream_mocov2"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_single_stream_simsiam"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_ae_single_stream"] = MODEL_LAYERS["simplified_mousenet_single_stream"]
MODEL_LAYERS["simplified_mousenet_depth_hour_glass_single_stream"] = MODEL_LAYERS["simplified_mousenet_single_stream"]

### Concatenation combinations
MODEL_LAYER_CONCATS = {}

# Shi MouseNet
MODEL_LAYER_CONCATS["shi_mousenet"] = {}
MODEL_LAYER_CONCATS["shi_mousenet"]["1"] = {"LGNv": ["LGNv"]}
for area in ["VISp", "VISal", "VISpl", "VISli", "VISrl", "VISl", "VISpor"]:
    MODEL_LAYER_CONCATS["shi_mousenet"]["1"][area] = [area+cortical for cortical in ["4", "2/3", "5"]]

# Shi MouseNet variants
MODEL_LAYER_CONCATS["shi_mousenet_vispor5"] = MODEL_LAYER_CONCATS["shi_mousenet"]
MODEL_LAYER_CONCATS["shi_mousenet_vispor5_pool_4"] = MODEL_LAYER_CONCATS["shi_mousenet"]


### Extended model layers
MODEL_LAYERS_EXTENDED = {

    # VGGs
    "vgg16":
        ["features.1"] +
        ["features.4"] +
        ["features.6"] +
        ["features.9"] +
        ["features.11"] +
        ["features.13"] +
        ["features.16"] +
        ["features.18"] +
        ["features.20"] +
        ["features.23"] +
        ["features.25"] +
        ["features.27"] +
        ["features.30"] +
        ["classifier.1"] +
        ["classifier.4"],

    # ResNets
    "resnet18":
        ["maxpool"] +
        [f"layer1.{i}.relu{j}" for i in range(2) for j in range(1, 3)] +
        [f"layer2.{i}.relu{j}" for i in range(2) for j in range(1, 3)] +
        [f"layer3.{i}.relu{j}" for i in range(2) for j in range(1, 3)] +
        [f"layer4.{i}.relu{j}" for i in range(2) for j in range(1, 3)],

    "resnet34":
        ["maxpool"] +
        [f"layer1.{i}.relu{j}" for i in range(3) for j in range(1, 3)] +
        [f"layer2.{i}.relu{j}" for i in range(4) for j in range(1, 3)] +
        [f"layer3.{i}.relu{j}" for i in range(6) for j in range(1, 3)] +
        [f"layer4.{i}.relu{j}" for i in range(3) for j in range(1, 3)],

    "resnet50":
        ["maxpool"] +
        [f"layer1.{i}.relu{j}" for i in range(3) for j in range(1, 4)] +
        [f"layer2.{i}.relu{j}" for i in range(4) for j in range(1, 4)] +
        [f"layer3.{i}.relu{j}" for i in range(6) for j in range(1, 4)] +
        [f"layer4.{i}.relu{j}" for i in range(3) for j in range(1, 4)],

    "resnet101":
        ["maxpool"] +
        [f"layer1.{i}.relu{j}" for i in range(3) for j in range(1, 4)] +
        [f"layer2.{i}.relu{j}" for i in range(4) for j in range(1, 4)] +
        [f"layer3.{i}.relu{j}" for i in range(23) for j in range(1, 4)] +
        [f"layer4.{i}.relu{j}" for i in range(3) for j in range(1, 4)],

    "resnet152":
        ["maxpool"] +
        [f"layer1.{i}.relu{j}" for i in range(3) for j in range(1, 4)] +
        [f"layer2.{i}.relu{j}" for i in range(8) for j in range(1, 4)] +
        [f"layer3.{i}.relu{j}" for i in range(36) for j in range(1, 4)] +
        [f"layer4.{i}.relu{j}" for i in range(3) for j in range(1, 4)],

}

# VGG16 variants
MODEL_LAYERS_EXTENDED["vgg16_64x64_input"] = MODEL_LAYERS_EXTENDED["vgg16"]
MODEL_LAYERS_EXTENDED["vgg16_ir_64x64"] = MODEL_LAYERS_EXTENDED["vgg16"]

# ResNet18 variants
MODEL_LAYERS_EXTENDED["resnet18_64x64_input"] = MODEL_LAYERS_EXTENDED["resnet18"]
MODEL_LAYERS_EXTENDED["resnet18_ir_64x64"] = MODEL_LAYERS_EXTENDED["resnet18"]

# Other ResNet variants
MODEL_LAYERS_EXTENDED["resnet34_64x64_input"] = MODEL_LAYERS_EXTENDED["resnet34"]
MODEL_LAYERS_EXTENDED["resnet34_ir_64x64"] = MODEL_LAYERS_EXTENDED["resnet34"]
MODEL_LAYERS_EXTENDED["resnet50_64x64_input"] = MODEL_LAYERS_EXTENDED["resnet50"]
MODEL_LAYERS_EXTENDED["resnet50_ir_64x64"] = MODEL_LAYERS_EXTENDED["resnet50"]
MODEL_LAYERS_EXTENDED["resnet101_64x64_input"] = MODEL_LAYERS_EXTENDED["resnet101"]
MODEL_LAYERS_EXTENDED["resnet101_ir_64x64"] = MODEL_LAYERS_EXTENDED["resnet101"]
MODEL_LAYERS_EXTENDED["resnet152_64x64_input"] = MODEL_LAYERS_EXTENDED["resnet152"]
MODEL_LAYERS_EXTENDED["resnet152_ir_64x64"] = MODEL_LAYERS_EXTENDED["resnet152"]


if __name__ == "__main__":
    import mouse_vision.models.imagenet_models as im

    def assert_module_exists(model, layer_name):
        module = model
        for p in layer_name.split('.'):
            module = module._modules.get(p)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {p}."

    for model_name in MODEL_LAYERS.keys():
        if "mousenet" in model_name or model_name == "alexnet_64x64_input_dict":
            continue

        print(model_name)
        model = im.__dict__[model_name](pretrained=False)
        layer_names = MODEL_LAYERS[model_name]
        if layer_names == []:
            continue
        else:
            for layer_name in layer_names:
                assert_module_exists(model, layer_name)



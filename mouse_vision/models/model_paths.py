import os

from mouse_vision.core.default_dirs import MODEL_SAVE_DIR
from mouse_vision.models.model_layers import MODEL_LAYERS
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS

MODEL_PATHS = {

    ## intermediate image resolution models
    "simplified_mousenet_dual_stream_visp_3x3_ir_32x32":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_32x32/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_44x44":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_44x44/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_84x84":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_84x84/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_104x104":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_104x104/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_124x124":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_124x124/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_144x144":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_144x144/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_164x164":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_164x164/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_184x184":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_184x184/exp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_204x204":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_dual_204x204/exp1/checkpoint_epoch_199.pt"),

    "alexnet_ir_dmlocomotion":
        os.path.join(MODEL_SAVE_DIR, "dmlocomotion/irloss_alexnet_64x64_input_pool_6/run1/checkpoint_epoch_199.pt"),

    "alexnet_ir_84x84":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_84x84/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_104x104":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_104x104/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_124x124":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_124x124/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_144x144":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_144x144/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_164x164":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_164x164/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_184x184":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_184x184/exp0/checkpoint_epoch_199.pt"),

    "alexnet_ir_204x204":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_204x204/exp1/checkpoint_epoch_199.pt"),

    ## 224 px models

    "alexnet_ir_224x224":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_224x224/exp0/checkpoint_epoch_199.pt"),

    "simplified_mousenet_single_stream_ir_224x224":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_single_stream_224x224_input/gpuexp0/checkpoint_epoch_199.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_ir_224x224":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_dual_stream_visp3x3_224x224_input/gpuexp1/checkpoint_epoch_199.pt"),

    "simplified_mousenet_six_stream_visp_3x3_ir_224x224":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_six_stream_visp3x3_224x224_input/gpuexplowlr0/checkpoint_epoch_199.pt"),

    ## 64 px models

    "alexnet_64x64_input_dict":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_64x64_input_dict/smallim0/model_best.pt"),

    "alexnet_64x64_input_pool_1":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet/ps1smallim/model_best.pt"),

    "alexnet_64x64_input_pool_6":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet/smallim2/model_best.pt"),

    "alexnet_64x64_input_pool_6_with_ir_transforms":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet/sup_alexnet_ir_transforms/model_best.pt"),

    "alexnet_bn_64x64_input_pool_6_with_ir_transforms":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet/sup_alexnet_bn_ir_transforms/model_best.pt"),

    "alexnet_two_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_two_64x64/gpuexp0/model_best.pt"),

    "alexnet_three_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_three_64x64/gpuexp0/model_best.pt"),

    "alexnet_four_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_four_64x64/gpuexp0/model_best.pt"),

    "alexnet_five_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_five_64x64/gpuexp0/model_best.pt"),

    "alexnet_six_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_alexnet_six_64x64/gpuexp0/model_best.pt"),

    "alexnet_64x64_input_pool_6_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_alexnet_64x64_input_pool_6/exp0/model_best.pt"),

    "vgg16_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_vgg16_64x64_input/exp0/model_best.pt"),

    "vgg16_64x64_input_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_vgg16_64x64_input/exp0/model_best.pt"),

    "resnet18_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_resnet18/smallim0/model_best.pt"),

    "resnet18_64x64_input_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_resnet18_64x64_input/exp0/model_best.pt"),

    "resnet34_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_resnet34_64x64_input/gpuexp0/model_best.pt"),

    "resnet50_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_resnet50_64x64_input/gpuexp0/model_best.pt"),

    "resnet101_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_resnet101_64x64_input/gpuexp0/model_best.pt"),

    "resnet152_64x64_input":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_resnet152_64x64_input/gpuexp0/model_best.pt"),

    "shi_mousenet":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_shi_mousenet/van01/model_best.pt"),

    "shi_mousenet_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_shi_mousenet_64x64_input/exp0/model_best.pt"),

    "shi_mousenet_vispor5":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_shi_mousenet/vispor5ps0_01/model_best.pt"),

    "shi_mousenet_vispor5_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_shi_mousenet_vispor5_64x64_input/exp0/model_best.pt"),

    "shi_mousenet_vispor5_pool_4":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_shi_mousenet/vispor5_01/model_best.pt"),

    "simplified_mousenet_six_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_six_stream/alexlr0/model_best.pt"),

    "simplified_mousenet_six_stream_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_six_stream_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_six_stream_visp_3x3":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_six_stream_visp_3x3/alexlr0/model_best.pt"),

    "simplified_mousenet_six_stream_visp_3x3_bn":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_six_stream_visp_3x3_bn_64x64_input/gpulowlr0/model_best.pt"),

    "simplified_mousenet_six_stream_visp_3x3_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_six_stream_visp_3x3_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_six_stream_visp_3x3_bn_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_six_stream_visp_3x3_bn_64x64_input/highlr0/model_best.pt"),

    "simplified_mousenet_six_stream_vispor_only":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_six_stream_vispor_only/alexlr0/model_best.pt"),

    "simplified_mousenet_six_stream_vispor_only_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_six_stream_vispor_only_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_six_stream_vispor_only_visp_3x3":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_six_stream_vispor_only_visp_3x3/alexlr0/model_best.pt"),

    "simplified_mousenet_six_stream_vispor_only_visp_3x3_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_six_stream_vispor_only_visp_3x3_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_dual_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_dual_stream/alexlr0/model_best.pt"),

    "simplified_mousenet_dual_stream_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_dual_stream_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_dual_stream_visp_3x3":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_dual_stream_visp_3x3/alexlr0/model_best.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_bn":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_dual_stream_visp_3x3_bn_64x64_input/gpuexp0/model_best.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_dual_stream_visp_3x3_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_dual_stream_visp_3x3_bn_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_dual_stream_visp_3x3_bn_64x64_input/lowlr0/model_best.pt"),

    "simplified_mousenet_dual_stream_vispor_only":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_dual_stream_vispor_only/alexlr0/model_best.pt"),

    "simplified_mousenet_dual_stream_vispor_only_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_dual_stream_vispor_only_64x64_input/exp0/model_best.pt"),

    "simplified_mousenet_dual_stream_vispor_only_visp_3x3":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_dual_stream_vispor_only_visp_3x3/alexlr0/model_best.pt"),

    "simplified_mousenet_dual_stream_vispor_only_visp_3x3_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_dual_stream_vispor_only_visp_3x3_64x64_input/exp1/model_best.pt"),

    "simplified_mousenet_single_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/xentloss_simplified_mousenet_single_stream_64x64_input/exp1/model_best.pt"),

    "simplified_mousenet_single_stream_cifar10":
        os.path.join(MODEL_SAVE_DIR, "cifar10/xentloss_simplified_mousenet_single_stream_64x64_input/exp1/model_best.pt"),

    # For unsupervised methods we get the last epoch since the metrics that get "best" do not always track final classification accuracy

    # RL end-to-end, truncated AlexNet
    "alexnet_64x64_rl_scratch_truncated":
        os.path.join(MODEL_SAVE_DIR, "dmlocomotion/alexnet_truncated_rl_scratch.pt"),

    # Instance recognition, alexnet
    "alexnet_bn_ir_64x64_input_pool_6":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_alexnet_bn_64x64_input_pool_6/gpuexplowlr0/checkpoint_epoch_199.pt"),

    # SimSiam, AlexNet
    "alexnet_bn_simsiam_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simsiamloss_alexnet_64x64/exp1/checkpoint_epoch_99.pt"),

    # SimCLR, AlexNet
    "alexnet_bn_simclr_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simclrloss_alexnet_64x64/exp1/checkpoint_epoch_199.pt"),

    # MoCov2, AlexNet
    "alexnet_bn_mocov2_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/mocov2loss_alexnet_64x64/exp1/checkpoint_epoch_199.pt"),

    # Instance recognition, vgg16
    "vgg16_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_vgg16_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    # Instance recognition, resnets
    "resnet18_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_resnet18_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    "resnet34_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_resnet34_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    "resnet50_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_resnet50_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    "resnet101_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_resnet101_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    "resnet152_ir_64x64":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_resnet152_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    # Instance recognition, single stream
    "simplified_mousenet_single_stream_ir":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_single_stream_64x64_input/exp1/checkpoint_epoch_199.pt"),

    # RotNet, single stream
    "simplified_mousenet_single_stream_rotnet":
        os.path.join(MODEL_SAVE_DIR, "imagenet/rotnetloss_simplified_mousenet_single_stream_64x64_input/exp1/checkpoint_epoch_49.pt"),

    # SimSiam, single stream
    "simplified_mousenet_single_stream_simsiam":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simsiamloss_simplified_mousenet_single_stream_64x64_input/exp1/checkpoint_epoch_99.pt"),

    # SimCLR, single stream
    "simplified_mousenet_single_stream_simclr":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simclrloss_simplified_mousenet_single_stream_64x64_input/exp1/checkpoint_epoch_199.pt"),

    # MoCov2, single stream
    "simplified_mousenet_single_stream_mocov2":
        os.path.join(MODEL_SAVE_DIR, "imagenet/mocov2loss_simplified_mousenet_single_stream_64x64_input/exp1/checkpoint_epoch_199.pt"),

    # Autoencoder, single stream
    "simplified_mousenet_ae_single_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/aeloss_simplified_mousenet_single_stream/gpuexp2/checkpoint_epoch_99.pt"),

    # Depth prediction, single stream
    "simplified_mousenet_depth_hour_glass_single_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/depthpredloss_simplified_mousenet_single_stream/exp2_hour_glass/checkpoint_epoch_49.pt"),

    # Instance recognition, dual stream
    "simplified_mousenet_dual_stream_visp_3x3_ir":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_dual_stream_visp3x3_64x64_input/gpuexp0/checkpoint_epoch_199.pt"),

    # RotNet, dual stream
    "simplified_mousenet_dual_stream_visp_3x3_rotnet":
        os.path.join(MODEL_SAVE_DIR, "imagenet/rotnetloss_simplified_mousenet_dual_stream_visp3x3_64x64_input/exp0/checkpoint_epoch_49.pt"),

    # SimSiam, dual stream
    "simplified_mousenet_dual_stream_visp_3x3_simsiam":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simsiamloss_simplified_mousenet_dual_stream_visp3x3_64x64_input/exp1/checkpoint_epoch_99.pt"),

    # SimCLR, dual stream
    "simplified_mousenet_dual_stream_visp_3x3_simclr":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simclrloss_simplified_mousenet_dual_stream_visp3x3_64x64_input/exp0/checkpoint_epoch_199.pt"),

    # MoCov2, dual stream
    "simplified_mousenet_dual_stream_visp_3x3_mocov2":
        os.path.join(MODEL_SAVE_DIR, "imagenet/mocov2loss_simplified_mousenet_dual_stream_visp3x3_64x64_input/exp0/checkpoint_epoch_199.pt"),

    # Autoencoder, dual stream
    "simplified_mousenet_ae_dual_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/aeloss_simplified_mousenet_dual_stream/gpuexp2/checkpoint_epoch_99.pt"),

    # Depth prediction, dual stream
    "simplified_mousenet_depth_hour_glass_dual_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/depthpredloss_simplified_mousenet_dual_stream/exp1_hour_glass/checkpoint_epoch_49.pt"),

    # Instance recognition, six stream
    "simplified_mousenet_six_stream_visp_3x3_ir":
        os.path.join(MODEL_SAVE_DIR, "imagenet/irloss_simplified_mousenet_six_stream_visp3x3_64x64_input/gpuexplowlr0/checkpoint_epoch_199.pt"),

    # RotNet, six stream
    "simplified_mousenet_six_stream_visp_3x3_rotnet":
        os.path.join(MODEL_SAVE_DIR, "imagenet/rotnetloss_simplified_mousenet_six_stream_visp3x3_64x64_input/gpuexp0/checkpoint_epoch_49.pt"),

    # SimSiam, six stream
    "simplified_mousenet_six_stream_visp_3x3_simsiam":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simsiamloss_simplified_mousenet_six_stream_visp3x3_64x64_input/exp0/checkpoint_epoch_99.pt"),

    # SimCLR, six stream
    "simplified_mousenet_six_stream_visp_3x3_simclr":
        os.path.join(MODEL_SAVE_DIR, "imagenet/simclrloss_simplified_mousenet_six_stream_visp3x3_64x64_input/exp1/checkpoint_epoch_199.pt"),

    # MoCov2, six stream
    "simplified_mousenet_six_stream_visp_3x3_mocov2":
        os.path.join(MODEL_SAVE_DIR, "imagenet/mocov2loss_simplified_mousenet_six_stream_visp3x3_64x64_input/exp0/checkpoint_epoch_199.pt"),

    # Autoencoder, six stream
    "simplified_mousenet_ae_six_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/aeloss_simplified_mousenet_six_stream/gpuexp2/checkpoint_epoch_99.pt"),

    # Depth prediction, six stream
    "simplified_mousenet_depth_hour_glass_six_stream":
        os.path.join(MODEL_SAVE_DIR, "imagenet/depthpredloss_simplified_mousenet_six_stream/exp1_hour_glass/checkpoint_epoch_49.pt"),

}

for model in MODEL_PATHS.keys():
    assert model in MODEL_LAYERS.keys(), f"{model} not in model_layers.py"
    assert model in MODEL_TRANSFORMS.keys(), f"{model} not in model_transforms.py"



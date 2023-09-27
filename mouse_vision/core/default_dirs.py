import os
import socket

CCN_COND = socket.gethostname().endswith("-ccncluster")
SH_COND = socket.gethostname().startswith("sh")
OM_COND = (
    (not CCN_COND)
    and (not SH_COND)
    and (
        (socket.gethostname() == "openmind7")
        or (socket.gethostname().startswith("node"))
        or (socket.gethostname().startswith("dgx"))
    )
)
# ML datasets
IMAGENET_DATA_DIR = "/data5/chengxuz/Dataset/imagenet_raw"
if CCN_COND and (socket.gethostname() == "node12-neuroaicluster"):
    IMAGENET_DATA_DIR = "/data2/chengxuz/imagenet_raw"
elif OM_COND:
    IMAGENET_DATA_DIR = "/om2/group/yanglab/anayebi/imagenet_raw"
CIFAR10_DATA_DIR = "/mnt/fs5/nclkong/datasets/cifar10/"
PBRNET_DATA_DIR = "/mnt/fs5/nclkong/datasets/pbrnet_new/"
DMLOCOMOTION_DATA_DIR = "/data7/anayebi/jpegs_seq40/"

# Sherlock directories
SHERLOCK_CACHE_DIR = os.path.join(
    os.environ.get("GROUP_SCRATCH", ""), "nclkong/cached_cheese/"
)

# Cluster specific directories
NEURAL_DATA_DIR = ""
HVM_DATA_DIR = ""
NEURAL_RESULTS_DIR = ""
TORCH_HOME = ""
MODEL_FEATURES_SAVE_DIR = ""
if CCN_COND:  # cluster
    NEURAL_DATA_DIR = "/mnt/fs5/nclkong/allen_inst/"
    NEURAL_RESULTS_DIR = os.path.join(NEURAL_DATA_DIR, "results/")
    NEURAL_FIT_RESULTS_DIR = os.path.join(
        NEURAL_DATA_DIR, "results/neural_fit_results/"
    )
    NEURAL_FIT_RESULTS_DIR_NEW = os.path.join(
        NEURAL_DATA_DIR, "results/neural_fit_results_new/"
    )
    NEURAL_FIT_RESULTS_DIR_EXTENDED_NEW = os.path.join(
        NEURAL_DATA_DIR, "results/neural_fit_results_extended_new/"
    )
    NEURAL_FIT_MODEL_TO_MODEL_DIR = os.path.join(
        NEURAL_DATA_DIR, "results/neural_fit_model_to_model/"
    )
    TORCH_HOME = "/mnt/fs5/nclkong/torch_models/"
    MODEL_SAVE_DIR = "/mnt/fs5/nclkong/trained_models/mouse_vision/"
    MODEL_FEATURES_SAVE_DIR = "/mnt/fs5/nclkong/mouse_vision/model_features/"
    MODEL_FEATURES_EXTENDED_SAVE_DIR = (
        "/mnt/fs5/nclkong/mouse_vision/model_features_extended/"
    )
    REP_DIFF_DIR = os.path.join(NEURAL_DATA_DIR, "rep_diff_results/")
    INTERAN_TR_DIR = os.path.join(NEURAL_DATA_DIR, "interan_train_results/")
    INTERAN_IM_SS_DIR = os.path.join(NEURAL_DATA_DIR, "interan_im_ss_results/")
    INTERAN_NEURON_SS_DIR = os.path.join(NEURAL_DATA_DIR, "interan_neuron_ss_results/")
    HVM_DATA_DIR = os.path.join(NEURAL_DATA_DIR, "hvm_data")
    DT_DATA_DIR = os.path.join(NEURAL_DATA_DIR, "dtd/splits/")
    SKLEARN_TRANSFER_RESULTS_DIR = os.path.join(
        NEURAL_DATA_DIR, "sklearn_transfer_results/"
    )

elif SH_COND:  # sherlock
    NEURAL_DATA_DIR = os.path.join(SHERLOCK_CACHE_DIR, "datasets/")
    NEURAL_RESULTS_DIR = os.path.join(SHERLOCK_CACHE_DIR, "results/")
    NEURAL_FIT_RESULTS_DIR = os.path.join(
        SHERLOCK_CACHE_DIR, "results/neural_fit_results/"
    )
    NEURAL_FIT_RESULTS_DIR_NEW = os.path.join(
        SHERLOCK_CACHE_DIR, "results/neural_fit_results_new/"
    )
    NEURAL_FIT_RESULTS_DIR_EXTENDED_NEW = os.path.join(
        SHERLOCK_CACHE_DIR, "results/neural_fit_results_extended_new/"
    )
    NEURAL_FIT_MODEL_TO_MODEL_DIR = os.path.join(
        SHERLOCK_CACHE_DIR, "results/neural_fit_model_to_model/"
    )
    TORCH_HOME = os.path.join(os.environ["GROUP_HOME"], "nclkong/")
    MODEL_SAVE_DIR = os.path.join(
        os.environ["GROUP_SCRATCH"], "nclkong/trained_models/mouse_vision/"
    )
    MODEL_FEATURES_SAVE_DIR = os.path.join(SHERLOCK_CACHE_DIR, "model_features/")
    MODEL_FEATURES_EXTENDED_SAVE_DIR = os.path.join(
        SHERLOCK_CACHE_DIR, "model_features_extended/"
    )
    REP_DIFF_DIR = os.path.join(SHERLOCK_CACHE_DIR, "rep_diff_results/")
    INTERAN_TR_DIR = os.path.join(SHERLOCK_CACHE_DIR, "interan_train_results/")
    INTERAN_IM_SS_DIR = os.path.join(SHERLOCK_CACHE_DIR, "interan_im_ss_results/")
    INTERAN_NEURON_SS_DIR = os.path.join(
        SHERLOCK_CACHE_DIR, "interan_neuron_ss_results/"
    )
    HVM_DATA_DIR = os.path.join(NEURAL_DATA_DIR, "hvm_data")
    DT_DATA_DIR = os.path.join(NEURAL_DATA_DIR, "dtd/splits/")
    SKLEARN_TRANSFER_RESULTS_DIR = os.path.join(
        SHERLOCK_CACHE_DIR, "sklearn_transfer_results/"
    )

elif OM_COND:
    OM_CACHE_DIR = "/om4/group/jazlab/anayebi/mouse_vision/"
    OM_CACHE_DIR_2_BASE = "/om2/group/yanglab/anayebi/"
    OM_CACHE_DIR_2 = os.path.join(OM_CACHE_DIR_2_BASE, "mouse_vision/")
    NEURAL_DATA_DIR_BASE = os.path.join(OM_CACHE_DIR_2_BASE, "neural_dataset_backup/")
    NEURAL_DATA_DIR = os.path.join(NEURAL_DATA_DIR_BASE, "allen_inst/")
    NEURAL_RESULTS_DIR = os.path.join(OM_CACHE_DIR_2, "results/")
    NEURAL_FIT_RESULTS_DIR = os.path.join(OM_CACHE_DIR_2, "results/neural_fit_results/")
    NEURAL_FIT_RESULTS_DIR_NEW = os.path.join(
        OM_CACHE_DIR_2, "results/neural_fit_results_new/"
    )
    NEURAL_FIT_RESULTS_DIR_EXTENDED_NEW = os.path.join(
        OM_CACHE_DIR_2, "results/neural_fit_results_extended_new/"
    )
    NEURAL_FIT_MODEL_TO_MODEL_DIR = os.path.join(
        OM_CACHE_DIR, "results/neural_fit_model_to_model/"
    )
    TORCH_HOME = os.path.join(OM_CACHE_DIR, "torch_home/")
    MODEL_SAVE_DIR = os.path.join(OM_CACHE_DIR, "trained_models/")
    MODEL_FEATURES_SAVE_DIR = os.path.join(OM_CACHE_DIR_2, "model_features/")
    MODEL_FEATURES_EXTENDED_SAVE_DIR = os.path.join(
        OM_CACHE_DIR_2, "model_features_extended/"
    )
    REP_DIFF_DIR = os.path.join(OM_CACHE_DIR, "rep_diff_results/")
    INTERAN_TR_DIR = os.path.join(OM_CACHE_DIR, "interan_train_results/")
    INTERAN_IM_SS_DIR = os.path.join(OM_CACHE_DIR, "interan_im_ss_results/")
    INTERAN_NEURON_SS_DIR = os.path.join(OM_CACHE_DIR, "interan_neuron_ss_results/")
    HVM_DATA_DIR = os.path.join(NEURAL_DATA_DIR_BASE, "hvm_data/")
    DT_DATA_DIR = os.path.join(NEURAL_DATA_DIR, "dtd/splits/")
    SKLEARN_TRANSFER_RESULTS_DIR = os.path.join(
        OM_CACHE_DIR_2, "sklearn_transfer_results/"
    )

else:  # the socket name will be the name of the gcloud instance
    # We only use gcloud for model training not anything neural data related
    IMAGENET_DATA_DIR = "/home/anayebi/mouse_vision_data/imagenet/"
    if "nk" in socket.gethostname():
        # Instance name: mv-pt-vm-nk
        IMAGENET_DATA_DIR = "/home/nclkong/datasets/imagenet_raw/"
    CIFAR10_DATA_DIR = "/home/anayebi/mouse_vision_data/cifar10/"
    MODEL_SAVE_DIR = "gs://mouse_vision_models/"

# HVM data path
HVM_DATA_PATH = os.path.join(HVM_DATA_DIR, "ventral_neural_data.hdf5")
HVM_SPLIT_FILE = os.path.join(HVM_DATA_DIR, "test_hvm_splits.npz")
HVM_V3V6_SPLIT_FILE = os.path.join(HVM_DATA_DIR, "test_hvm_splits_v3v6.npz")

# Neuropixel mouse data directories
NEUROPIX_DATA_PATH = os.path.join(NEURAL_DATA_DIR, "mouse_neuropixels_visual_data.pkl")
NEUROPIX_DATA_PATH_WITH_RELS = os.path.join(
    NEURAL_DATA_DIR, "mouse_neuropixels_visual_data_with_reliabilities.pkl"
)
MANIFEST_PATH = os.path.join(NEURAL_DATA_DIR, "manifest.json")

# Calcium imaging mouse data directories
CALCIUM_DATA_PATH = os.path.join(
    NEURAL_DATA_DIR, "ophys_data/mouse_calcium_visual_data.pkl"
)
CALCIUM_DATA_PATH_WITH_RELS = os.path.join(
    NEURAL_DATA_DIR, "ophys_data/mouse_calcium_visual_data_with_reliabilities.pkl"
)
CALCIUM_MANIFEST_PATH = os.path.join(NEURAL_DATA_DIR, "ophys_data/manifest.json")

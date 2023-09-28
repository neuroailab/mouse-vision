import os

# Model directories
TORCH_HOME = "/home/nclkong/plos_mouse_vision/mouse-vision"
BASE_DIR = "/home/nclkong/plos_mouse_vision/mouse-vision"
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model_ckpts")

# Data directories
NEUROPIX_DATA_PATH_WITH_RELS = os.path.join(
    BASE_DIR, "neural_data/mouse_neuropixels_visual_data_with_reliabilities.pkl"
)
CALCIUM_DATA_PATH_WITH_RELS = os.path.join(
    BASE_DIR, "neural_data/mouse_calcium_visual_data_with_reliabilities.pkl"
)

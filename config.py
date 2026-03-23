import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 8
INPUT_CHANNELS = 12

BATCH_SIZE = 32
NUM_WORKERS = 6
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20

TRAIN_SPLIT = 0.85

# Best deep supervision setting from experiments
AUX_BRANCH_SCALE = 4          # branch generated at H/4, W/4
AUX_LOSS_WEIGHT = 0.10        # best weight
MAIN_LOSS_WEIGHT = 0.90

SEEDS = [0, 1, 2, 3, 4, 5]

S1_DIR = "256x256_Dfc2020/s1_0"
S2_DIR = "256x256_Dfc2020/s2_0"
LABEL_DIR = "256x256_Dfc2020/dfc_0"

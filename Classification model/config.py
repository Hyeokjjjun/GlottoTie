import torch

class Config:
    # Data paths
    COMMERCIAL_DATA_PATH = '/train_data'
    OUR_DEVICE_DATA_PATH = '/val_data'

    # Class settings
    NUM_CLASSES = 3

    # Few-shot settings (3-way 2-shot)
    N_WAY = 3
    K_SHOT = 2
    N_QUERY = 15

    # Data shape
    INPUT_CHANNELS = 60

    # Training settings
    NUM_EPISODES = 15000
    VAL_EPISODES = 100  # Reduced number of validation episodes (500 -> 100)

    # Model settings
    EMBEDDING_DIM = 512
    DROPOUT_RATE = 0.1

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4

    # Data Augmentation
    USE_AUGMENTATION = False
    AUGMENTATION_PROB = 0.8

    # Early Stopping
    PATIENCE = 5

    # Logging frequency
    PRINT_EVERY = 100
    VAL_EVERY = 100

    # Misc
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PATH = './checkpoints'
    SEED = 42
    
config = Config()
import os

data_main_path_directory = os.path.normpath('/train_data_base/')

# image dimensions
n_frames = 1  # number of distorted image for feeding the model (D)
batch_size = 1  # (N)
image_channels = 3  # (C)

# training parameters
n_down_sampling = 4
epochs = 1000
initial_lr = 1e-3
weight_decay = 0

# recovery and results
exp_name = 'LPR_1'
save_freq = 10
storage_path = os.path.normpath('./weights/{}/ckpt_epoch_{}.pt')
display_freq = 4
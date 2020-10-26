import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import datetime
import os
from torch.nn import L1Loss
from unet_3d import UNet
import config
from torch_loader import TurbulenceDataset, PowerPad, RandomHorizontalFlip, ToTensor
from torchvision import utils
from tensorboardX import SummaryWriter
from cuda_utils import set_cuda


def str_pt_to_epoch(storage_path):
    """
    Extract the epoch number from weight path
    :param storage_path: weight path (including the epoch number)
    :return: epoch number
    """
    return int(os.path.splitext(storage_path)[0].split('epoch_')[1])


def recover_model():
    model = UNet().to(device)
    recovery_epoch = 0

    # get experiment weight paths
    weights_dir = './weights/{}'.format(config.exp_name)

    os.system('mkdir ./weights')
    os.system(f'mkdir {weights_dir}')

    # start from latest epoch and load weights from last checkpoint of this experiment
    exp_weights = os.listdir(weights_dir)

    if len(exp_weights) > 0:
        latest_epoch = max([str_pt_to_epoch(weight) for weight in exp_weights if weight.endswith('pt')])
        latest_weight_path = config.storage_path.format(config.exp_name, latest_epoch)

        model_dict = model.state_dict()
        loaded_state_dict = torch.load(latest_weight_path, map_location='cuda:0')
        for model_key, loaded_value in zip(model_dict.keys(), loaded_state_dict.values()):
            model_dict[model_key] = loaded_value
        model.load_state_dict(model_dict)

        recovery_epoch = latest_epoch + 1  # we saved latest_epoch, now start latest_epoch + 1

        print(f'Recovered weight from {latest_weight_path}')
        print(f'Starting from epoch {latest_epoch}...')

    return model, recovery_epoch


data_transforms = {
    'train': transforms.Compose([
        PowerPad(),
        RandomHorizontalFlip(),
        ToTensor()
    ]),
    'val': transforms.Compose([
        PowerPad(),
        ToTensor()
    ]),
}

dtype, device = set_cuda()

writer = SummaryWriter(logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

criterion = L1Loss()

model, recovery_epoch = recover_model()

optimizer = optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)

train_dataset = TurbulenceDataset(root_dir='/home/dsteam/PycharmProjects/turbulence/train_data_base',
                                  transform=data_transforms['train'], writer=writer)

validation_dataset = TurbulenceDataset(root_dir='/home/dsteam/PycharmProjects/turbulence/val_data_base',
                                       transform=data_transforms['val'], writer=writer)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=0)

val_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=0)

for epoch in range(recovery_epoch, config.epochs):
    print(f'Epoch {epoch}/{config.epochs}...')
    epoch_train_loss = 0
    epoch_val_loss = 0
    start_epoch = time.time()
    for batch_idx, batch in enumerate(train_dataloader):
        print(f'Train batch {batch_idx}')
        # Transfer to GPU
        batch_in, batch_gt = batch['distorted_tensor'], batch['gt_image']

        batch_in = batch_in.type(dtype).to(device).permute(0, 2, 1, 3, 4)
        batch_gt = batch_gt.type(dtype).to(device).unsqueeze(0).permute(0, 2, 1, 3, 4)

        model.train()
        optimizer.zero_grad()

        batch_pred = model(batch_in)
        batch_loss = criterion(batch_pred, batch_gt)
        batch_loss.backward()
        optimizer.step()

        epoch_train_loss += batch_loss
        writer.add_scalars(f'{config.exp_name}-Train', {'L1': batch_loss.item()}, epoch)

        del batch_in, batch_gt

    print(f'Epoch {epoch} - Train Loss: {epoch_train_loss}')

    # Validation
    for batch_idx, batch in enumerate(train_dataloader):
        print(f'Val batch {batch_idx}')

        # Transfer to GPU
        batch_in, batch_gt = batch['distorted_tensor'], batch['gt_image']

        batch_in = batch_in.type(dtype).to(device).permute(0, 2, 1, 3, 4)
        batch_gt = batch_gt.type(dtype).to(device).unsqueeze(0).permute(0, 2, 1, 3, 4)

        model.eval()
        with torch.no_grad():
            batch_pred = model(batch_in)
            batch_loss = criterion(batch_pred, batch_gt)

            # Model computations
            writer.add_scalars(f'{config.exp_name}-Val', {'L1': batch_loss.item()}, epoch)

            if batch_idx % config.display_freq == 0:
                stack_img = torch.cat([batch_in[0, :, 0, :, :],
                                       batch_pred[0, :, 0, :, :],
                                       batch_gt[0, :, 0, :, :]], dim=2)

                stack_grid = utils.make_grid(stack_img)

                writer.add_image('Results', stack_grid, epoch)

            epoch_val_loss += batch_loss

        del batch_in, batch_gt

    print(f'Epoch {epoch} - Val Loss: {epoch_val_loss}')

    writer.add_scalars(f'{config.exp_name}-Profiling-Time', {'Epoch Time': time.time() - start_epoch}, epoch)
    if epoch % config.save_freq == 0:
        with open(config.storage_path.format(config.exp_name, epoch), 'wb') as f:
            torch.save(model.state_dict(), f)

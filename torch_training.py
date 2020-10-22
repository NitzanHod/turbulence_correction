import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import datetime
from torch.nn import L1Loss
from unet_3d import UNet
import config
from torch_loader import TurbulenceDataset, NewPad

from tensorboardX import SummaryWriter
from cuda_utils import set_cuda

data_transforms = {
    'train': transforms.Compose([
        NewPad(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        NewPad(),
        transforms.ToTensor()
    ]),
}

dtype, device = set_cuda()

writer = SummaryWriter(logdir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

criterion = L1Loss()
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.initial_lr, weight_decay=config.weight_decay)

train_dataset = TurbulenceDataset(root_dir='/home/dsteam/PycharmProjects/turbulence/train_data_base',
                                  transform=data_transforms['train'])

validation_dataset = TurbulenceDataset(root_dir='/home/dsteam/PycharmProjects/turbulence/val_data_base',
                                       transform=data_transforms['val'])

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=0)

val_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=True,
                            num_workers=0)

for epoch in range(config.epochs):
    print(f'Epoch {epoch}/{config.epochs}...')
    epoch_train_loss = 0
    epoch_val_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        print(f'Train batch {batch_idx}')
        # Transfer to GPU
        batch_in, batch_gt = batch['distorted_tensor'], batch['gt_image']
        # TODO: allow > 1 batches by editing unsqueeze
        batch_in = batch_in.type(dtype).to(device).permute(0, 2, 1, 3, 4)
        batch_gt = batch_gt.type(dtype).to(device).unsqueeze(0).permute(0, 2, 1, 3, 4)

        model.train()
        optimizer.zero_grad()

        batch_pred = model(batch_in)
        batch_loss = criterion(batch_pred, batch_gt)
        batch_loss.backward()
        optimizer.step()

        epoch_train_loss += batch_loss
        writer.add_scalars('Train', {'loss': batch_loss.item()}, epoch)

        del batch_in, batch_gt

    print(f'Epoch {epoch} - Train Loss: {epoch_train_loss}')

    # Validation
    for batch_idx, batch in enumerate(train_dataloader):
        print(f'Val batch {batch_idx}')

        # Transfer to GPU
        batch_in, batch_gt = batch['distorted_tensor'], batch['gt_image']

        # TODO: allow >1 batches by editing unsqueeze
        batch_in = batch_in.type(dtype).to(device).permute(0, 2, 1, 3, 4)
        batch_gt = batch_gt.type(dtype).to(device).unsqueeze(0).permute(0, 2, 1, 3, 4)

        model.eval()
        with torch.no_grad():
            batch_pred = model(batch_in)
            batch_loss = criterion(batch_pred, batch_gt)

            # Model computations
            writer.add_scalars('Val', {'loss': batch_loss.item()}, epoch)

            epoch_val_loss += batch_loss

        del batch_in, batch_gt

    print(f'Epoch {epoch} - Val Loss: {epoch_val_loss}')

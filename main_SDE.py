import argparse
import os
import time
import wandb
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from Toolbox.data_SDE import Dataset
from Toolbox.model_SDE import Net_ms2pan
from Toolbox.losses import SDE_Losses
import numpy as np
from Toolbox.wald_utilities import wald_protocol_v2

# ===================== Utility Functions ====================== #
def save_checkpoint(model, name, satellite):
    """Save the model checkpoint."""
    model_dir = os.path.join('model', satellite, str(name))
    model_out_path = os.path.join(model_dir, 'model_SDE.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

def process_batch(batch, model, criterion, optimizer, device):
    """Process a single batch of data."""
    ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    pan = wald_protocol_v2(ms, pan, 4, 'WV3')

    optimizer.zero_grad()
    out = model(ms)
    loss = criterion(out, pan)
    loss.backward()
    optimizer.step()
    return loss.item()

###################################################################
# ------------------- Train Function ---------------------------- #
###################################################################
def train_sde(training_data_loader, name, satellite, model, criterion, optimizer, device, epochs, enable_wandb):
    """Train the model on the given dataset."""
    min_loss = float('inf')
    t_start = time.time()

    for epoch in tqdm(range(epochs), desc="Epochs", colour='blue', leave=False):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(training_data_loader, desc="Batches", leave=False):
            loss = process_batch(batch, model, criterion, optimizer, device)
            epoch_losses.append(loss)

        mean_loss = np.nanmean(epoch_losses)
        
        if mean_loss and enable_wandb:
            wandb.log({"SDE/loss": mean_loss})

        if mean_loss < min_loss:
            save_checkpoint(model, name, satellite)
            min_loss = mean_loss

    t_end = time.time()
    tqdm.write(f"Sample {name} SDE training completed in {t_end - t_start:.2f}s")

###################################################################
# ------------------- Main Function ----------------------------- #
###################################################################
def main_run(lr_sde=None, epochs=None, batch_size=None, device=None, satellite=None, file_path=None, sample=None, enable_wandb=None):
    # ================== Constants =================== #
    SEED = 10
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.deterministic = True

    # ========== Hyperparameters (Command Line) ========= #
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_sde", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
    parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
    parser.add_argument("--sample", type=int, required=True, help="Sample index to process")
    parser.add_argument("--enable_wandb", type=bool, default=False, help="Enable W&B logging")

    if any(arg is None for arg in [lr_sde, epochs, batch_size, device, satellite, file_path, sample, enable_wandb]):
        cmd_args = parser.parse_args()
        lr_sde = cmd_args.lr_sde
        epochs = cmd_args.epochs
        batch_size = cmd_args.batch_size
        device = cmd_args.device
        satellite = cmd_args.satellite
        file_path = cmd_args.file_path
        sample = cmd_args.sample
        enable_wandb = cmd_args.enable_wandb

    device = torch.device(device)
    model = Net_ms2pan().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_sde, betas=(0.9, 0.999))
    criterion = SDE_Losses(device)

    train_set = Dataset(file_path, sample)
    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )
    train_sde(training_data_loader, sample, satellite, model, criterion, optimizer, device, epochs, enable_wandb)

if __name__ == "__main__":
    main_run()
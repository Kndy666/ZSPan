import argparse
import os
import time
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

# ================== Constants =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ========== Hyperparameters (Command Line) ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
parser.add_argument("--sample", type=int, required=True, help="Sample index to process")
args = parser.parse_args()

# Parse command-line arguments
device = torch.device(args.device)

# Initialize model, optimizer, and loss function
model = Net_ms2pan().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
criterion = SDE_Losses(device)

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
def train_sde(training_data_loader, name, satellite):
    """Train the model on the given dataset."""
    min_loss = float('inf')
    t_start = time.time()

    for epoch in tqdm(range(args.epochs), desc="Epochs", colour='blue', leave=False):
        model.train()
        epoch_losses = []
        
        for batch in tqdm(training_data_loader, desc="Batches", leave=False):
            loss = process_batch(batch, model, criterion, optimizer, device)
            epoch_losses.append(loss)

        mean_loss = np.nanmean(epoch_losses)
        
        if mean_loss < min_loss:
            save_checkpoint(model, name, satellite)
            min_loss = mean_loss

    t_end = time.time()
    tqdm.write(f"Sample {name} training completed in {t_end - t_start:.2f}s")

###################################################################
# ------------------- Main Function ----------------------------- #
###################################################################
if __name__ == "__main__":
    tqdm.write("Starting SDE Training...")
    train_set = Dataset(args.file_path, args.sample)
    training_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0
    )
    train_sde(training_data_loader, args.sample, args.satellite)
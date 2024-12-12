import os
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from Toolbox.losses import FUG_Losses, RSP_Losses
from Toolbox.model_FUG import FusionNet
from Toolbox.model_SDE import Net_ms2pan
from Toolbox.data_RND import Dataset, dynamic_batch_size_collate_fn
from Toolbox.wald_utilities import wald_protocol_v1, wald_protocol_v2

# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ========== HYPER PARAMETERS (Command Line) ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr_fug", type=float, default=0.0005, help="Learning rate for FUG")
parser.add_argument("--lr_rsp", type=float, default=0.001, help="Learning rate for RSP")
parser.add_argument("--epochs", type=int, default=140, help="Number of epochs for both FUG and RSP")
parser.add_argument("--device", type=str, default='cuda', help="Device to use (e.g., 'cuda' or 'cpu')")
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
args = parser.parse_args()

lr_fug = args.lr_fug
lr_rsp = args.lr_rsp
epochs = args.epochs
device = torch.device(args.device)
satellite = args.satellite
file_path = args.file_path

# ===================== Functions ====================== #
def save_checkpoint(model, name, satellite, stage):
    model_dir = os.path.join('model', satellite, str(name))
    model_out_path = os.path.join(model_dir, f'model_{stage}.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Combined Train Function ------------------- #
###################################################################
def train_combined(training_data_loader, name, satellite):
    # Initialize Models
    model = FusionNet().to(device)
    F_ms2pan = Net_ms2pan().to(device)
    F_ms2pan.load_state_dict(torch.load(os.path.join('model', satellite, str(name), 'model_SDE.pth')))

    # Initialize Loss Functions
    criterion_FUG = FUG_Losses(device, os.path.join('model', satellite, str(name), 'model_SDE.pth'))
    criterion_RSP = RSP_Losses(device)

    # Optimizers
    optimizer_FUG = optim.Adam(model.parameters(), lr=lr_fug, betas=(0.9, 0.999))
    optimizer_RSP = optim.Adam(model.parameters(), lr=lr_rsp, betas=(0.9, 0.999))
    
    # Betas for FUG Losses
    betas = [8, 1]

    # Training Loop
    t1 = time.time()
    min_loss_FUG, min_loss_RSP = float('inf'), float('inf')
    cnt1, cnt2 = 0, 0

    with tqdm(range(epochs), desc="Epochs", colour='blue', leave=False) as epoch_progress:
        for epoch in epoch_progress:
            epoch_progress.set_description(f"Epochs {epoch} (cnt1: {cnt1}, cnt2: {cnt2})")

            model.train()
            epoch_train_loss_FUG, epoch_train_loss_RSP = [], []

            for iteration, batch in enumerate(tqdm(training_data_loader, desc="Batches", leave=False), 1):
                if batch.get("full") is not None:
                    ms, lms, pan, shapes = batch["full"]
                    ms, lms, pan = ms.to(device), lms.to(device), pan.to(device)

                    optimizer_FUG.zero_grad()
                    res = model(lms, pan)
                    out = res + lms
                    dsr = wald_protocol_v1(out, pan, 4, 'WV3')
                    dpan = F_ms2pan(out)
                    loss1, loss2 = criterion_FUG(out, pan, ms, dsr, dpan)
                    total_loss_FUG = betas[0] * loss1 + betas[1] * loss2
                    epoch_train_loss_FUG.append(total_loss_FUG.item())
                    total_loss_FUG.backward()
                    optimizer_FUG.step()
                    cnt1 += 1
                if batch.get("reduced") is not None:
                    ms, lms, pan, shapes = batch["reduced"]
                    ms, lms, pan = ms.to(device), lms.to(device), pan.to(device)

                    optimizer_RSP.zero_grad()
                    lms_rr = wald_protocol_v1(lms, pan, 4, 'WV3')
                    pan_rr = wald_protocol_v2(ms, pan, 4, 'WV3')
                    res = model(lms_rr, pan_rr)
                    out2 = lms_rr + res
                    loss_RSP = criterion_RSP(out2, ms)
                    epoch_train_loss_RSP.append(loss_RSP.item())
                    loss_RSP.backward()
                    optimizer_RSP.step()
                    cnt2 += 1

            # Compute Epoch Loss
            if epoch_train_loss_FUG:
                t_loss_FUG = np.nanmean(np.array(epoch_train_loss_FUG))
                if t_loss_FUG < min_loss_FUG:
                    min_loss_FUG = t_loss_FUG
                    save_checkpoint(model, name, satellite, "FUG")

            if epoch_train_loss_RSP:
                t_loss_RSP = np.nanmean(np.array(epoch_train_loss_RSP))
                if t_loss_RSP < min_loss_RSP:
                    min_loss_RSP = t_loss_RSP
                    save_checkpoint(model, name, satellite, "RSP")

            if epoch % 10 == 0:
                epoch_progress.write(f"Sample {name} FUG Loss: {min_loss_FUG:.6f} RSP Loss: {min_loss_RSP:.6f}")
            epoch_progress.set_postfix(FUG_loss=f"{t_loss_FUG:.6f}" if epoch_train_loss_FUG else "N/A", 
                                       RSP_loss=f"{t_loss_RSP:.6f}" if epoch_train_loss_RSP else "N/A",)

    t2 = time.time()
    tqdm.write(f'Sample {name} RND training time: {t2 - t1:.2f}s')


###################################################################
# ------------------- Main Function ----------------------------- #
###################################################################
if __name__ == "__main__":
    tqdm.write('Run Combined Training...')
    with tqdm(range(20), desc="Samples", colour='green') as sample_progress:
        for name in sample_progress:
            sample_progress.set_description(f"Samples {name}")
            train_set = Dataset(file_path, name)
            
            training_data_loader = DataLoader(
                dataset=train_set, 
                num_workers=0, 
                batch_size=8, 
                shuffle=True, 
                drop_last=True, 
                collate_fn=dynamic_batch_size_collate_fn,
                pin_memory=True
            )
        
            train_combined(training_data_loader, name, satellite)
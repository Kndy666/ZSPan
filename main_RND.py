import os
import argparse
import time
import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from Toolbox.losses import FUG_Losses, RSP_Losses
from Toolbox.model_RND import FusionNet
from Toolbox.model_SDE import Net_ms2pan
from Toolbox.data_RND import Dataset_RSP, Dataset_FUG
from Toolbox.wald_utilities import wald_protocol_v1, wald_protocol_v2

# ===================== Utility Functions ====================== #
def save_checkpoint(model, name, satellite, stage):
    """Save the model checkpoint."""
    model_dir = os.path.join('model', satellite, str(name))
    model_out_path = os.path.join(model_dir, f'model_{stage}.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

def train_single_batch(batch, model, optimizer, criterion, device, mode, aux_model=None, betas=None):
    """Handle training for a single batch."""
    ms, lms, pan = batch
    ms, lms, pan = ms.to(device), lms.to(device), pan.to(device)
    
    optimizer.zero_grad()
    
    if mode == "FUG":
        res = model(lms, pan)
        out = res + lms
        dsr = wald_protocol_v1(out, pan, 4, 'WV3')
        dpan = aux_model(out)
        loss1, loss2 = criterion(out, pan, ms, dsr, dpan)
        loss = betas[0] * loss1 + betas[1] * loss2
    elif mode == "RSP":
        lms_rr = wald_protocol_v1(lms, pan, 4, 'WV3')
        pan_rr = wald_protocol_v2(ms, pan, 4, 'WV3')
        res = model(lms_rr, pan_rr)
        out = lms_rr + res
        loss = criterion(out, ms)
    else:
        raise ValueError("Invalid mode specified")
    
    loss.backward()
    optimizer.step()
    return loss.item()

###################################################################
# ------------------- Combined Train Function ------------------- #
###################################################################
def train_combined(file_path, batch_size, ratio, sample, satellite, device, lr_fug, lr_rsp, warmup, epochs, enable_wandb):
    rsp_data_loader = DataLoader(
        dataset=Dataset_RSP(file_path, sample),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )
    fug_data_loader = DataLoader(
        dataset=Dataset_FUG(file_path, sample),
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    # Initialize Models and Losses
    model = FusionNet().to(device)
    aux_model = Net_ms2pan().to(device)
    aux_model.load_state_dict(torch.load(os.path.join('model', satellite, str(sample), 'model_SDE.pth')))
    
    criterion_FUG = FUG_Losses(device, os.path.join('model', satellite, str(sample), 'model_SDE.pth'))
    criterion_RSP = RSP_Losses(device)
    optimizer_FUG = optim.Adam(model.parameters(), lr=lr_fug, betas=(0.9, 0.999))
    optimizer_RSP = optim.Adam(model.parameters(), lr=lr_rsp, betas=(0.9, 0.999))
    
    betas = [8, 1]
    min_loss_FUG, min_loss_RSP = float('inf'), float('inf')
    FUG_cnt, RSP_cnt = 0, 0

    t_start = time.time()
    for epoch in tqdm(range(epochs), desc="Epochs", colour='blue', leave=False):
        model.train()
        epoch_losses_FUG, epoch_losses_RSP = [], []

        if epoch < warmup:
            selction = "FUG"
        else:
            selction = np.random.choice(["FUG", "RSP"], p=[ratio, 1 - ratio])

        if selction == "RSP":
            for batch in tqdm(rsp_data_loader, desc="Batches", leave=False):
                loss_RSP = train_single_batch(
                    batch, model, optimizer_RSP, criterion_RSP, device, mode="RSP"
                )
                epoch_losses_RSP.append(loss_RSP)
                RSP_cnt += 1
        else:
            for batch in tqdm(fug_data_loader, desc="Batches", leave=False):
                loss_FUG = train_single_batch(
                    batch, model, optimizer_FUG, criterion_FUG, device, mode="FUG", aux_model=aux_model, betas=betas
                )
                epoch_losses_FUG.append(loss_FUG)
                FUG_cnt += 1
        
        # Log epoch losses to W&B
        if enable_wandb:
            if epoch_losses_FUG:
                wandb.log({"RND/FUG_loss": np.nanmean(epoch_losses_FUG)}, commit=False)
            if epoch_losses_RSP:
                wandb.log({"RND/RSP_loss": np.nanmean(epoch_losses_RSP)}, commit=False)
            wandb.log({"RND/FUG_cnt": FUG_cnt, "RND/RSP_cnt": RSP_cnt})

        # Save best models
        if epoch_losses_FUG and (mean_loss := np.nanmean(epoch_losses_FUG)) < min_loss_FUG:
            min_loss_FUG = mean_loss
            save_checkpoint(model, sample, satellite, "FUG")
        if epoch_losses_RSP and (mean_loss := np.nanmean(epoch_losses_RSP)) < min_loss_RSP:
            min_loss_RSP = mean_loss
            save_checkpoint(model, sample, satellite, "RSP")
        
        if epoch % 10 == 0:
            tqdm.write(f"Sample {sample}: FUG Loss: {min_loss_FUG:.6f} RSP Loss: {min_loss_RSP:.6f} FUG_cnt: {FUG_cnt}, RSP_cnt: {RSP_cnt}")

    t_end = time.time()
    tqdm.write(f'Sample {sample} RND training time: {t_end - t_start:.2f}s')

###################################################################
# ------------------- Main Function ----------------------------- #
###################################################################
def main_run(lr_fug=None, lr_rsp=None, warmup=None, ratio=None, epochs=None, batch_size=None, device=None, satellite=None, file_path=None, sample=None, enable_wandb=None):
    # ================== Constants =================== #
    SEED = 10
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    cudnn.deterministic = True

    # ========== Hyperparameters (Command Line) ========= #
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_fug", type=float, default=0.0005, help="Learning rate for FUG")
    parser.add_argument("--lr_rsp", type=float, default=0.001, help="Learning rate for RSP")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup epochs for RSP")
    parser.add_argument("--ratio", type=float, default=0.7, help="Ratio of FUG to RSP")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
    parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
    parser.add_argument("--sample", type=int, required=True, help="Sample index to process")
    parser.add_argument("--enable_wandb", type=bool, default=False, help="Enable W&B logging")

    if any(arg is None for arg in [lr_fug, lr_rsp, warmup, ratio, epochs, batch_size, device, satellite, file_path, sample, enable_wandb]):
        cmd_args = parser.parse_args()
        lr_fug = cmd_args.lr_fug
        lr_rsp = cmd_args.lr_rsp
        ratio = cmd_args.ratio
        warmup = cmd_args.warmup
        epochs = cmd_args.epochs
        batch_size = cmd_args.batch_size
        device = cmd_args.device
        satellite = cmd_args.satellite
        file_path = cmd_args.file_path
        sample = cmd_args.sample
        enable_wandb = cmd_args.enable_wandb

    device = torch.device(device)
    train_combined(file_path, batch_size, ratio, sample, satellite, device, lr_fug, lr_rsp, warmup, epochs, enable_wandb)

if __name__ == "__main__":
    main_run()
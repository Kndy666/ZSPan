import torch
import os
import wandb
import h5py
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from Toolbox.model_RND import FusionNet
from Assessment.indexes_evaluation_FS import indexes_evaluation_FS

def calc_indices(out, ms, pan, lms, satellite='wv3'):
    """Calculate the indices for the given sample."""
    HQNR, D_lambda, D_S = indexes_evaluation_FS(out.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), ms.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), pan.squeeze(0).squeeze(0).cpu().detach().numpy(), 11, 0, lms.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), satellite.upper(), 4, 32)
    return HQNR, D_lambda, D_S

def load_checkpoint(model, sample, satellite, stage):
    """Load the model checkpoint."""
    model_dir = os.path.join('model', satellite, str(sample))
    model_out_path = os.path.join(model_dir, f'model_{stage}.pth')
    model.load_state_dict(torch.load(model_out_path))
    return model

def preprocess_data(file_path, sample, device):
    """Load and preprocess data from the dataset."""
    with h5py.File(file_path, 'r') as dataset:
        ms = np.array(dataset['ms'][sample], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][sample], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][sample], dtype=np.float32) / 2047.0

    ms = torch.unsqueeze(torch.from_numpy(ms).float(), dim=0).to(device)
    lms = torch.unsqueeze(torch.from_numpy(lms).float(), dim=0).to(device)
    pan = torch.unsqueeze(torch.from_numpy(pan).float(), dim=0).to(device)
    return ms, lms, pan

def evaluate_model(satellite, file_path, sample, device):
    """Test the FusionNet model on a given sample."""
    device = torch.device(device)
    model = FusionNet().to(device)

    load_checkpoint(model, sample, satellite, 'FUG')

    ms, lms, pan = preprocess_data(file_path, sample, device)

    model.eval()
    with torch.no_grad():
        res = model(lms, pan)
        out = res + lms
        HQNR, D_lambda, D_S = calc_indices(out, ms, pan, lms, satellite)

    return HQNR, D_lambda, D_S

def main_run(satellite=None, file_path=None, sample=None, device=None, enable_wandb=None):
    # ================== Constants =================== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
    parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
    parser.add_argument("--sample", type=int, nargs='+', help="Sample index to process")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    parser.add_argument("--enable_wandb", type=bool, default=False, help="Enable W&B logging")

    if any(arg is None for arg in [satellite, file_path, sample, device, enable_wandb]):
        cmd_args = parser.parse_args()
        satellite = cmd_args.satellite
        file_path = cmd_args.file_path
        sample = cmd_args.sample
        device = cmd_args.device
        enable_wandb = cmd_args.enable_wandb

    df = pd.DataFrame(columns=["HQNR", "D_lambda", "D_S"])
    for i in tqdm(sample, desc="Samples", colour='green'):
        try:
            HQNR, D_lambda, D_S = evaluate_model(satellite, file_path, i, device)
            df.loc[len(df)] = [HQNR, D_lambda, D_S]
            tqdm.write(f"Sample {i}: HQNR: {HQNR:.4f} D_lambda: {D_lambda:.4f} D_S: {D_S:.4f}")
        except FileNotFoundError as e:
            tqdm.write(f"Sample {i} skipped: {e}")
    
    if enable_wandb:
        wandb.define_metric("indices/avg_HQNR", step_metric="epoch")
        wandb.define_metric("indices/avg_D_lambda", step_metric="epoch")
        wandb.define_metric("indices/avg_D_S", step_metric="epoch")
        wandb.define_metric("indices/var_HQNR", step_metric="epoch")
        wandb.define_metric("indices/var_D_lambda", step_metric="epoch")
        wandb.define_metric("indices/var_D_S", step_metric="epoch")

        table = wandb.Table(dataframe=df)
        wandb.log({"indices" : table, 
                   "indices/avg_HQNR" : df["HQNR"].mean(), "indices/avg_D_lambda" : df["D_lambda"].mean(), "indices/avg_D_S" : df["D_S"].mean(),
                   "indices/var_HQNR" : df["HQNR"].var(), "indices/var_D_lambda" : df["D_lambda"].var(), "indices/var_D_S" : df["D_S"].var(),
                   "epoch" : 0})
    
    tqdm.write(f"Avg HQNR: {df['HQNR'].mean():.4f} D_lambda: {df['D_lambda'].mean():.4f} D_S: {df['D_S'].mean():.4f}\nVar HQNR: {df['HQNR'].var():.4f} D_lambda: {df['D_lambda'].var():.4f} D_S: {df['D_S'].var():.4f}")

if __name__ == "__main__":
    main_run()
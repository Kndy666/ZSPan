import argparse
import h5py
import torch
from Toolbox.model_RND import FusionNet
import scipy.io as sio
import numpy as np
import os
import wandb
from tqdm import tqdm

# ===================== Utility Functions ====================== #
def load_model_weights(model, satellite, name, enable_wandb):
    """Load model weights from a checkpoint."""
    ckpt_path = os.path.join('model', satellite, str(name), 'model_FUG.pth')
    base_path = os.path.join("model", satellite, str(name))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if enable_wandb:
        wandb.save(ckpt_path, base_path=base_path)
    weight = torch.load(ckpt_path)
    model.load_state_dict(weight)

def preprocess_data(file_path, name, device):
    """Load and preprocess data from the dataset."""
    with h5py.File(file_path, 'r') as dataset:
        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

    ms = torch.unsqueeze(torch.from_numpy(ms).float(), dim=0).to(device)
    lms = torch.unsqueeze(torch.from_numpy(lms).float(), dim=0).to(device)
    pan = torch.unsqueeze(torch.from_numpy(pan).float(), dim=0).to(device)
    return ms, lms, pan

def save_output(output, satellite, name):
    """Save the output to a .mat file."""
    save_path = os.path.join('result', satellite)
    os.makedirs(save_path, exist_ok=True)
    sio.savemat(os.path.join(save_path, f"output_mulExm_{str(name)}.mat"), {'sr': output})

def test_fug(model, satellite, file_path, name, device, enable_wandb):
    """Test the FusionNet model on a given sample."""
    load_model_weights(model, satellite, name, enable_wandb)

    ms, lms, pan = preprocess_data(file_path, name, device)

    model.eval()
    with torch.no_grad():
        res = model(lms, pan)
        out = res + lms
        I_SR = torch.squeeze(out * 2047).permute(1, 2, 0).cpu().numpy()  # HxWxC

    save_output(I_SR, satellite, name)

def main_run(satellite=None, file_path=None, sample=None, device=None, enable_wandb=None):
    # ================== Constants =================== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
    parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
    parser.add_argument("--sample", type=int, required=True, help="Sample index to process")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    parser.add_argument("--enable_wandb", type=bool, default=False, help="Enable W&B logging")

    if any(arg is None for arg in [satellite, file_path, sample, device, enable_wandb]):
        cmd_args = parser.parse_args()
        satellite = cmd_args.satellite
        file_path = cmd_args.file_path
        sample = cmd_args.sample
        device = cmd_args.device
        enable_wandb = cmd_args.enable_wandb

    device = torch.device(device)
    model = FusionNet().to(device)
    try:
        test_fug(model, satellite, file_path, sample, device, enable_wandb)
    except FileNotFoundError as e:
        tqdm.write(f"Sample {sample} skipped: {e}")

if __name__ == "__main__":
    main_run()
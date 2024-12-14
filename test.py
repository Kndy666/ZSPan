import argparse
import h5py
import torch
from Toolbox.model_RND import FusionNet
import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

# ================== Constants =================== #
parser = argparse.ArgumentParser()
parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
parser.add_argument("--sample", type=int, required=True, help="Sample index to process")
args = parser.parse_args()

# Parse command-line arguments
satellite = args.satellite
file_path = args.file_path

# Initialize model
model = FusionNet()

# ===================== Utility Functions ====================== #
def load_model_weights(model, satellite, name):
    """Load model weights from a checkpoint."""
    ckpt_path = os.path.join('model', satellite, str(name), 'model_FUG.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    weight = torch.load(ckpt_path)
    model.load_state_dict(weight)

def preprocess_data(file_path, name):
    """Load and preprocess data from the dataset."""
    with h5py.File(file_path, 'r') as dataset:
        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

    ms = torch.unsqueeze(torch.from_numpy(ms).float(), dim=0)
    lms = torch.unsqueeze(torch.from_numpy(lms).float(), dim=0)
    pan = torch.unsqueeze(torch.from_numpy(pan).float(), dim=0)
    return ms, lms, pan

def save_output(output, satellite, name):
    """Save the output to a .mat file."""
    save_path = os.path.join('result', satellite)
    os.makedirs(save_path, exist_ok=True)
    sio.savemat(os.path.join(save_path, f"output_mulExm_{str(name)}.mat"), {'sr': output})

###################################################################
# ------------------- Main Test Function ------------------------ #
###################################################################
def test_fug(name):
    """Test the FusionNet model on a given sample."""
    # Load model weights
    load_model_weights(model, satellite, name)

    # Preprocess input data
    ms, lms, pan = preprocess_data(file_path, name)

    # Forward pass
    model.eval()
    with torch.no_grad():
        res = model(lms, pan)
        out = res + lms
        I_SR = torch.squeeze(out * 2047).permute(1, 2, 0).cpu().numpy()  # HxWxC

    # Save output
    save_output(I_SR, satellite, name)

###################################################################
# ------------------- Main Function ----------------------------- #
###################################################################
if __name__ == "__main__":
    tqdm.write("Starting RND Testing...")
    try:
        test_fug(args.sample)
    except FileNotFoundError as e:
        tqdm.write(f"Sample {args.sample} skipped: {e}")
import argparse
import h5py
import torch
from Toolbox.model_RSP import FusionNet
import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm

# ================== Pre-Define =================== #
parser = argparse.ArgumentParser()
parser.add_argument("--satellite", type=str, default='wv3', help="Satellite type")
parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
args = parser.parse_args()

satellite = args.satellite
file_path = args.file_path

model = FusionNet()

###################################################################
# ------------------- Main Test (Run second)---------------------- #
###################################################################

def test(name):
    ckpt = os.path.join('model', satellite, str(name), 'model_FUG.pth')
    weight = torch.load(ckpt)
    model.load_state_dict(weight)

    dataset = h5py.File(file_path, 'r')
    ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
    lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
    pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

    ms = torch.from_numpy(ms).float()
    lms = torch.from_numpy(lms).float()
    pan = torch.from_numpy(pan).float()

    ms = torch.unsqueeze(ms.float(), dim=0)
    lms = torch.unsqueeze(lms.float(), dim=0)
    pan = torch.unsqueeze(pan.float(), dim=0)

    res = model(lms, pan)
    out = res + lms

    I_SR = torch.squeeze(out * 2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

    save_path = os.path.join('result', satellite)
    os.makedirs(save_path, exist_ok=True)
    sio.savemat(os.path.join(save_path, f"output_mulExm_{str(name)}.mat"), {'sr': I_SR})

###################################################################
# ------------------- Main Function (Run first) ------------------- #
###################################################################

if __name__ == "__main__":
    tqdm.write("Run test...")
    with tqdm(range(20), desc="Samples", colour="blue") as sample_progress:
        for name in sample_progress:
            sample_progress.set_description(f"Sample {name}")
            test(name)
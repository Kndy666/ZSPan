# GPL License
# Copyright (C) 2023 , UESTC
# All Rights Reserved
#
# @Time    : 2023/10/1 20:29
# @Author  : Qi Cao
import argparse
import os
import time
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Toolbox.data_SDE import Dataset
from Toolbox.model_SDE import Net_ms2pan
from Toolbox.losses import SDE_Losses
import numpy as np
from Toolbox.wald_utilities import wald_protocol_v2

# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=250, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--device", type=str, default='cuda', help="Device to use")
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
parser.add_argument("--file_path", type=str, default=r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", help="Path to the dataset file")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device)
satellite = args.satellite
file_path = args.file_path

model = Net_ms2pan().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))  # optimizer 1
criterion = SDE_Losses(device)

def save_checkpoint(model, name):  # save model_SDE function
    model_dir = os.path.join('model', satellite, str(name))
    model_out_path = os.path.join(model_dir, 'model_SDE.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################

def train(training_data_loader, name):
    t1 = time.time() # training time
    min_loss = float('inf')
    with tqdm(range(epochs), desc="Epochs", leave=False, colour='blue') as epoch_progress:
        for epoch in epoch_progress:
            epoch_progress.set_description(f"Epochs {epoch}")
            epoch_train_loss = []

            # ============Epoch Train=============== #
            model.train()
            for batch in tqdm(training_data_loader, desc="Batches", leave=False):
                ms, lms, pan = Variable(batch[0]).to(device), \
                            Variable(batch[1]).to(device), \
                            Variable(batch[2], requires_grad=False).to(device)

                pan = wald_protocol_v2(ms, pan, 4, 'WV3')
                optimizer.zero_grad()  # fixed
                out = model(ms)

                loss = criterion(out, pan)  # compute loss
                epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

                loss.backward()  # fixed
                optimizer.step()  # fixed

            t_loss = np.nanmean(np.array(epoch_train_loss))
            epoch_progress.set_postfix(loss=f"{t_loss:.6f}")
            if t_loss < min_loss:
                save_checkpoint(model, name)
                min_loss = t_loss
    t2 = time.time()  # training time
    tqdm.write(f"Sample {name} SDE training time: {t2 - t1:.2f}s")  # training time
###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    tqdm.write("Run SDE...")
    with tqdm(range(20), desc="Samples", colour='green') as sample_progress:
        for name in sample_progress:
            sample_progress.set_description(f"Samples {name}")
            train_set = Dataset(file_path, name)
            training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                              pin_memory=True,
                                              drop_last=True)
            train(training_data_loader, name)
import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision

class Dataset_RSP(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset_RSP, self).__init__()
        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        self.ms_crops = torchvision.transforms.TenCrop(ms.shape[1] // 2)(ms)
        self.lms_crops = torchvision.transforms.TenCrop(lms.shape[1] // 2)(lms)
        self.pan_crops = torchvision.transforms.TenCrop(pan.shape[1] // 2)(pan)

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item]

    def __len__(self):
        return len(self.ms_crops)

class Dataset_FUG(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset_FUG, self).__init__()
        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float().squeeze(0)
        lms = torch.from_numpy(lms).float().squeeze(0)
        pan = torch.from_numpy(pan).float().squeeze(0)

        self.ms_crops = [ms]
        self.lms_crops = [lms]
        self.pan_crops = [pan]

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item]

    def __len__(self):
        return len(self.ms_crops)

if __name__ == "__main__":
    train_set = Dataset_RSP(r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", 19)
    print(len(train_set))
import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision

def dynamic_batch_size_collate_fn(batch):
    result = {}

    large_image_batch = [item for item in batch if item[3] == 128]
    if large_image_batch:
        ms = torch.stack([item[0] for item in large_image_batch])
        lms = torch.stack([item[1] for item in large_image_batch])
        pan = torch.stack([item[2] for item in large_image_batch])
        ms_shapes = torch.tensor([item[3] for item in large_image_batch])
        result["full"] = (ms, lms, pan, ms_shapes)

    small_image_batch = [item for item in batch if item[3] == 64]
    if small_image_batch:
        ms_crops = torch.stack([item[0] for item in small_image_batch])
        lms_crops = torch.stack([item[1] for item in small_image_batch])
        pan_crops = torch.stack([item[2] for item in small_image_batch])
        ms_shapes_crops = torch.tensor([item[3] for item in small_image_batch])
        result["reduced"] = (ms_crops, lms_crops, pan_crops, ms_shapes_crops)

    return result

class Dataset(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset, self).__init__()
        dataset = h5py.File(file_path, 'r')

        ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
        lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
        pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] // 2)
        self.ms_crops = list(MS_crop(ms))
        self.ms_crops.append(ms)
        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] // 2)
        self.lms_crops = list(LMS_crop(lms))
        self.lms_crops.append(lms)
        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] // 2)
        self.pan_crops = list(PAN_crop(pan))
        self.pan_crops.append(pan)

        self.ms_shapes = [crop.shape[1] for crop in self.ms_crops]

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item], self.ms_shapes[item]

    def __len__(self):
        return len(self.ms_crops)

if __name__ == "__main__":
    train_set = Dataset(r"../02-Test-toolbox-for-traditional-and-DL(Matlab)-1/1_TestData/PanCollection/test_wv3_OrigScale_multiExm1.h5", 19)
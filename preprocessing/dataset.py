import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, label_map, num_frames=20, split="train", transform=None):
        """
        root_dir:Frame image root path. -- data/frames/train
        label_map: A mapping dictionary from category to index
        num_frames: The number of frames used in each video segment
        split: Dataset partitioning (train/val/test)
        transform: image augmentation (albumentations)
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.label_map = label_map

        self.samples = [] 
         # Each sample is (frame path list, label)
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
        for pt_file in os.listdir(class_path):
            if pt_file.endswith(".pt"):
                pt_path = os.path.join(class_path, pt_file)
                self.samples.append((pt_path, self.label_map[class_name]))



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, label = self.samples[idx]
        video = torch.load(pt_path)            # [T, C, H, W]
        #video = video.permute(1, 0, 2, 3)      # [C, T, H, W] for CNN input (frame-sequence)

        if self.transform:
            video = self.transform(video)

        return video, label

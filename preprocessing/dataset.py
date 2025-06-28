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
    def __init__(self, root_dir, label_map, num_frames=30, split="train", transform=None):
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
            for video_id in os.listdir(class_path):
                frame_dir = os.path.join(class_path, video_id)
                frame_paths = sorted(glob(os.path.join(frame_dir, "*.jpg")))
                if len(frame_paths) >= 1:
                    self.samples.append((frame_paths, self.label_map[class_name]))
        
        # Print the paths and labels of the first 20 samples
        print("\n[DEBUG] First 20 samples loaded:")
        for i in range(min(20, len(self.samples))):
            frame_paths, label = self.samples[i]
            class_name = list(self.label_map.keys())[list(self.label_map.values()).index(label)]
            print(f"Sample {i}: class={class_name}, label={label}")
            print(f"    Frame count: {len(frame_paths)} | First frame: {frame_paths[0]}")



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        # truncate or fill the frame length
        if len(frame_paths) >= self.num_frames:
            frame_paths = frame_paths[:self.num_frames]
        else:
            frame_paths += [frame_paths[-1]] * (self.num_frames - len(frame_paths))

        images = []
        for path in frame_paths:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)["image"]
            else:
                image = T.ToTensor()(Image.fromarray(image))
            images.append(image)

        # return shape: [T, C, H, W], with the label as int
        #video_tensor = torch.stack(images)
        # video_tensor = video_tensor.permute(1, 0, 2, 3) #[C, T, H, W]
        video_tensor = torch.stack(images).permute(1, 0, 2, 3)
        return video_tensor, label

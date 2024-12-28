import io
import os
import zipfile
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import CenterCrop, Resize, ToTensor, Normalize


class ZipImageDataset(Dataset):
    def __init__(self, zip_path, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        # Open the zip file and list image files
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            self.file_list = [
                f for f in zf.namelist()
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))  # Filter for image files
            ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Open the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            # Read the file at the given index
            file_name = self.file_list[idx]
            with zf.open(file_name) as file:
                # Load the image using PIL
                image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Apply any transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, file_name


def image_data_loader(zip_path, args, transform=None) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor() if transform is None else transform
    dataset = ZipImageDataset(zip_path, transform=transform)

    total_size = len(dataset)
    train_size = int(total_size * args.train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

class DepthDataset(Dataset):
    def __init__(self, Depth_dir, Depth_transform=None, frame_interval=3):
        self.Depth_dir = Depth_dir
        self.frame_interval = frame_interval

        if Depth_transform is not None:
            self.Depth_transform = Depth_transform
        else:
            self.Depth_transform = transforms.Compose([
                CenterCrop((224 * 3, 224 * 2)),
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ])

        self.depth_file_list = defaultdict(list)
        for root, _, files in os.walk(os.path.expanduser(self.Depth_dir)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    depth_key = os.path.basename(root)
                    if depth_key != '.ipynb_checkpoints':
                        self.depth_file_list[depth_key].append(os.path.join(root, file))

        for key, value in self.depth_file_list.items():
            self.depth_file_list[key] = sorted(value)

        self.depth_file_list_keys = list(self.depth_file_list.keys())

        # self._assertion()

    def __len__(self):
        return len(self.depth_file_list)

    def __getitem__(self, idx):
        depth_frames = []
        depth_key = self.depth_file_list_keys[idx]
        for i, image_path in enumerate(self.depth_file_list.get(depth_key, [])):
            if i % self.frame_interval == 0:
                image = Image.open(image_path).convert('L')
                depth_frames.append(self.Depth_transform(image))

        if len(depth_frames) > 0:
            depth_tensor = torch.stack(depth_frames, dim=0)
        else:
            depth_tensor = torch.empty(0, 1, 224, 224)

        depth_tensor = depth_tensor.repeat(1, 3, 1, 1)  # Duplicate the single channel to 3 channels

        label = int(depth_key[-3:]) - 1

        return depth_tensor, label, depth_key


def pad_collate_fn(batch, sequence_length):
    # Extract videos and file names
    depths, label, file_names = zip(*batch)

    # Determine the maximum number of frames in this batch
    max_frames = max([depth.shape[0] for depth in depths])
    max_frames = max_frames if sequence_length == -1 else max(max_frames, sequence_length)

    # Pad all videos to the max_frames length
    padded_depth_frames = []

    for depth in depths:
        padding_size = (0, 0, 0, 0, 0, 0, 0, max_frames - depth.shape[0])
        padded_depth = torch.nn.functional.pad(depth, padding_size)
        padded_depth_frames.append(padded_depth)

    padded_depth_frames = torch.stack(padded_depth_frames)

    label = torch.tensor(label, dtype=torch.int64)

    return padded_depth_frames, label, file_names

def data_loader(Depth_path, args, Depth_transform=None):
    assert args.batch_size is not None
    assert args.train_split is not None
    assert args.sequence_length is not None

    # dataset = RGBDDataset(RGB_path, Depth_path, RGB_transform=RGB_transform, Depth_transform=Depth_transform, frame_interval=args.frame_interval)
    dataset = DepthDataset(Depth_path, Depth_transform=Depth_transform)

    total_size = len(dataset)
    train_size = int(total_size * args.train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))

    return train_loader, test_loader


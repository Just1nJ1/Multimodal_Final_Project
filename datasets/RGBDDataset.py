import io
import os
import time
import zipfile
from collections import defaultdict

import av
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize, ToTensor


class RGBDZipedDataset(Dataset):
    def __init__(self, RGB_zip_path, Depth_zip_path, RGB_transform=None, Depth_transform=None, frame_interval=0.5):
        self.RGB_zip_path = RGB_zip_path
        self.Depth_zip_path = Depth_zip_path

        if RGB_transform is not None:
            self.RGB_transform = RGB_transform
        else:
            self.RGB_transform = transforms.Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ])

        if Depth_transform is not None:
            self.Depth_transform = Depth_transform
        else:
            self.Depth_transform = transforms.Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
            ])

        self.frame_interval = frame_interval

        # Open the zip file and list video files
        with zipfile.ZipFile(self.RGB_zip_path, 'r') as zf:
            self.video_file_list = [
                f for f in zf.namelist()
                if f.lower().endswith('.avi')  # Filter for video files
            ]

        self.depth_file_list = defaultdict(list)

        with zipfile.ZipFile(Depth_zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.depth_file_list[name.split('/')[1]].append(name)

        for key, value in self.depth_file_list.items():
            self.depth_file_list[key] = sorted(value)

        self._assertion()

    def __len__(self):
        return len(self.video_file_list)

    def __getitem__(self, idx):
        file_name = self.video_file_list[idx]

        # Open the zip file
        with zipfile.ZipFile(self.RGB_zip_path, 'r') as zf:
            with zf.open(file_name) as file:
                video_bytes = file.read()

        # Load video frames as a list of tensors
        video_frames = self._load_video_frames(video_bytes)

        # Stack all frames into a single tensor (T, C, H, W)
        if len(video_frames) > 0:
            video_tensor = torch.stack(video_frames, dim=0)  # Stack frames along the temporal dimension
        else:
            # Handle the case where a video has no frames
            video_tensor = torch.empty(0, 3, 224, 224)

        depth_frames = []
        with zipfile.ZipFile(self.Depth_zip_path, 'r') as zf:
            for name in self.depth_file_list[file_name[13:-8]]:
                with zf.open(name) as file:
                    image = Image.open(io.BytesIO(file.read())).convert('L')
                    depth_frames.append(self.Depth_transform(image))
        if len(depth_frames) > 0:
            depth_tensor = torch.stack(depth_frames, dim=0)
        else:
            depth_tensor = torch.empty(0, 1, 224, 224)

        depth_tensor = depth_tensor.repeat(1, 3, 1, 1)  # Duplicate the single channel to 3 channels

        assert video_tensor.shape[0] == depth_tensor.shape[0]
        assert file_name.find('A') == 29

        label = int(file_name[30:33]) - 1

        return video_tensor, depth_tensor, label, file_name

    def _load_video_frames(self, video_bytes):
        # Open the video using PyAV from the byte stream
        video_stream = io.BytesIO(video_bytes)
        container = av.open(video_stream, format='avi')

        # Get the frame rate from the video stream
        video_stream = container.streams.video[0]  # Get the first video stream
        fps = video_stream.average_rate  # Get the average frame rate as a Fraction
        fps = float(fps)  # Convert to float for calculations

        interval_frames = max(int(self.frame_interval * fps), 1)  # Calculate frame interval

        frames = []
        for frame_idx, frame in enumerate(container.decode(video=0)):  # Decode video frames
            # Sample frames at the specified interval
            if frame_idx % interval_frames == 0:
                frame_rgb = frame.to_rgb().to_ndarray()  # Convert frame to RGB ndarray
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.RGB_transform(frame_pil)
                frames.append(frame_tensor)

        return frames

    def _assertion(self):
        print('####### Dataset Assertion ########')
        for key, value in self.depth_file_list.items():
            assert f'nturgb+d_rgb/{key}_rgb.avi' in self.video_file_list, f'nturgb+d_rgb/{key}_rgb.avi not in self.video_file_list'
        for name in sorted(self.video_file_list):
            assert name[13:-8] in self.depth_file_list.keys(), f'{name[13:-8]} not in self.depth_file_list.keys()'
        print('####### Assertion Complete #######')


class RGBDDataset(Dataset):
    def __init__(self, RGB_dir, Depth_dir, RGB_transform=None, Depth_transform=None, frame_interval=0.5):
        self.RGB_dir = RGB_dir
        self.Depth_dir = Depth_dir

        if RGB_transform is not None:
            self.RGB_transform = RGB_transform
        else:
            self.RGB_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

        if Depth_transform is not None:
            self.Depth_transform = Depth_transform
        else:
            self.Depth_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

        self.frame_interval = frame_interval

        self.video_file_list = []
        for root, _, files in os.walk(os.path.expanduser(os.path.join(self.RGB_dir, "nturgb+d_rgb"))):
            for file in files:
                if file.lower().endswith('.avi'):
                    self.video_file_list.append(os.path.join(root, file))

        # Create depth file mapping
        self.depth_file_list = defaultdict(list)
        for root, _, files in os.walk(os.path.expanduser(self.Depth_dir)):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    depth_key = os.path.basename(root)  # Use folder name as the key
                    self.depth_file_list[depth_key].append(os.path.join(root, file))

        for key, value in self.depth_file_list.items():
            self.depth_file_list[key] = sorted(value)

        # self._assertion()

    def __len__(self):
        return len(self.video_file_list)

    def __getitem__(self, idx):
        file_path = self.video_file_list[idx]
        file_name = os.path.basename(file_path)

        # Load video frames as a list of tensors
        video_frames = self._load_video_frames(file_path)

        # Stack all frames into a single tensor (T, C, H, W)
        if len(video_frames) > 0:
            video_tensor = torch.stack(video_frames, dim=0)  # Stack frames along the temporal dimension
        else:
            # Handle the case where a video has no frames
            video_tensor = torch.empty(0, 3, 224, 224)

        depth_frames = []
        depth_key = file_name[:-8]
        for image_path in self.depth_file_list.get(depth_key, []):
            image = Image.open(image_path).convert('L')
            depth_frames.append(self.Depth_transform(image))

        if len(depth_frames) > 0:
            depth_tensor = torch.stack(depth_frames, dim=0)
        else:
            depth_tensor = torch.empty(0, 1, 224, 224)

        depth_tensor = depth_tensor.repeat(1, 3, 1, 1)  # Duplicate the single channel to 3 channels

        assert video_tensor.shape[0] == depth_tensor.shape[0]

        assert file_name.find('A') == 16
        label = int(file_name[17:20]) - 1

        return video_tensor, depth_tensor, label, file_name

    def _load_video_frames(self, video_path):
        # Open the video using PyAV
        container = av.open(video_path)

        # Get the frame rate from the video stream
        video_stream = container.streams.video[0]  # Get the first video stream
        fps = video_stream.average_rate  # Get the average frame rate as a Fraction
        fps = float(fps)  # Convert to float for calculations

        interval_frames = max(int(self.frame_interval * fps), 1)  # Calculate frame interval

        frames = []
        for frame_idx, frame in enumerate(container.decode(video=0)):  # Decode video frames
            # Sample frames at the specified interval
            if frame_idx % interval_frames == 0:
                frame_rgb = frame.to_rgb().to_ndarray()  # Convert frame to RGB ndarray
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.RGB_transform(frame_pil)
                frames.append(frame_tensor)

        return frames

    # def _assertion(self):
    #     print('####### Dataset Assertion ########')
    #     for key, value in self.depth_file_list.items():
    #         assert f'nturgb+d_rgb/{key}_rgb.avi' in self.video_file_list, f'nturgb+d_rgb/{key}_rgb.avi not in self.video_file_list'
    #     for name in sorted(self.video_file_list):
    #         assert name[13:-8] in self.depth_file_list.keys(), f'{name[13:-8]} not in self.depth_file_list.keys()'
    #     print('####### Assertion Complete #######')


def pad_collate_fn(batch, sequence_length):
    # Extract videos and file names
    videos, depths, label, file_names = zip(*batch)

    # Determine the maximum number of frames in this batch
    max_frames = max([video.shape[0] for video in videos])
    max_frames = max_frames if sequence_length == -1 else max(max_frames, sequence_length)

    # Pad all videos to the max_frames length
    padded_video_frames = []

    for video in videos:
        padding_size = (0, 0, 0, 0, 0, 0, 0, max_frames - video.shape[0])  # (width, height, channels, frames)
        padded_video = torch.nn.functional.pad(video, padding_size)  # Pad with zeros
        padded_video_frames.append(padded_video)

    # Stack videos into a single tensor
    padded_video_frames = torch.stack(padded_video_frames)

    padded_depth_frames = []

    for depth in depths:
        padding_size = (0, 0, 0, 0, 0, 0, 0, max_frames - depth.shape[0])
        padded_depth = torch.nn.functional.pad(depth, padding_size)
        padded_depth_frames.append(padded_depth)

    padded_depth_frames = torch.stack(padded_depth_frames)

    label = torch.tensor(label, dtype=torch.int64)

    return padded_video_frames, padded_depth_frames, label, file_names

def data_loader(RGB_path, Depth_path, args, RGB_transform=None, Depth_transform=None):
    assert args.frame_interval is not None
    assert args.batch_size is not None
    assert args.train_split is not None
    assert args.sequence_length is not None

    # dataset = RGBDDataset(RGB_path, Depth_path, RGB_transform=RGB_transform, Depth_transform=Depth_transform, frame_interval=args.frame_interval)
    datasets = []
    for r in RGB_path:
        datasets.append(RGBDDataset(r, Depth_path, RGB_transform=RGB_transform, Depth_transform=Depth_transform, frame_interval=args.frame_interval))

    dataset = ConcatDataset(datasets)

    total_size = len(dataset)
    train_size = int(total_size * args.train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))

    return train_loader, test_loader


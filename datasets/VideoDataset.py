import io
import zipfile
import os

import av
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import CenterCrop, Resize, ToTensor, Normalize


class ZipVideoDataset(Dataset):
    def __init__(self, zip_path, transform=None, frame_interval=0.5):
        self.zip_path = zip_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                CenterCrop((224 * 3, 224 * 2)),
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.5], std=[0.5]),
            ])
        self.frame_interval = frame_interval

        # Open the zip file and list video files
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            self.file_list = [
                f for f in zf.namelist()
                if f.lower().endswith('.avi')  # Filter for video files
            ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Open the zip file
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            file_name = self.file_list[idx]
            with zf.open(file_name) as file:
                video_bytes = file.read()

        # Load video frames as a list of tensors
        video_frames = self._load_video_frames(video_bytes)

        # Stack all frames into a single tensor (T, C, H, W)
        if len(video_frames) > 0:
            video_tensor = torch.stack(video_frames, dim=0)  # Stack frames along the temporal dimension
        else:
            # Handle the case where a video has no frames
            video_tensor = torch.empty(0, 3, 1920, 1080)

        return video_tensor, file_name

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
                frame_tensor = self.transform(frame_pil)
                # frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1)  # Convert to tensor (C, H, W)
                frames.append(frame_tensor)

        return frames


def pad_collate_fn(batch, sequence_length):
    """
    Pads videos in a batch to the same number of frames.
    Args:
        batch: List of tuples (video_frames, file_name)
            - video_frames: Tensor of shape (T, C, H, W)
            - file_name: String
    Returns:
        padded_videos: Tensor of shape (batch_size, max_frames, C, H, W)
        file_names: List of strings
    """
    # Extract videos and file names
    videos, file_names = zip(*batch)

    # Determine the maximum number of frames in this batch
    max_frames = max([video.shape[0] for video in videos])
    max_frames = max_frames if sequence_length == -1 else max(max_frames, sequence_length)

    # Pad all videos to the max_frames length
    padded_videos = []
    for video in videos:
        padding_size = (0, 0, 0, 0, 0, 0, 0, max_frames - video.shape[0])  # (width, height, channels, frames)
        print(padding_size)
        padded_video = torch.nn.functional.pad(video, padding_size)  # Pad with zeros
        print(padded_video.shape)
        padded_videos.append(padded_video)

    # Stack videos into a single tensor
    padded_videos = torch.stack(padded_videos)

    return padded_videos, file_names

def video_data_loader(zip_path, args, transform=None) -> tuple[DataLoader, DataLoader]:
    assert args.frame_interval is not None
    assert args.batch_size is not None
    assert args.train_split is not None
    assert args.sequence_length is not None

    dataset = ZipVideoDataset(zip_path, transform=transform, frame_interval=args.frame_interval)

    total_size = len(dataset)
    train_size = int(total_size * args.train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))

    return train_loader, test_loader


class RGBDataset(Dataset):
    def __init__(self, RGB_dir, RGB_transform=None, frame_interval=4):
        self.RGB_dir = RGB_dir

        if RGB_transform is not None:
            self.RGB_transform = RGB_transform
        else:
            self.RGB_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        self.frame_interval = frame_interval

        self.video_file_list = []
        for root, _, files in os.walk(os.path.expanduser(os.path.join(self.RGB_dir, "nturgb+d_rgb"))):
            for file in files:
                if file.lower().endswith('.avi'):
                    self.video_file_list.append(os.path.join(root, file))

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

        assert file_name.find('A') == 16
        label = int(file_name[17:20]) - 1

        return video_tensor, label, file_name

    def _load_video_frames(self, video_path):
        container = av.open(video_path)

        frames = []
        for frame_idx, frame in enumerate(container.decode(video=0)):  # Decode video frames
            # Sample frames at the specified interval
            if frame_idx % self.frame_interval == 0:
                frame_rgb = frame.to_rgb().to_ndarray()  # Convert frame to RGB ndarray
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.RGB_transform(frame_pil)
                frames.append(frame_tensor)

        return frames


def pad_collate_fn(batch, sequence_length):
    # Extract videos and file names
    videos, label, file_names = zip(*batch)

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

    label = torch.tensor(label, dtype=torch.int64)

    return padded_video_frames, label, file_names

def data_loader(RGB_path, args, RGB_transform=None):
    assert args.frame_interval is not None
    assert args.batch_size is not None
    assert args.train_split is not None
    assert args.sequence_length is not None

    datasets = []
    for r in RGB_path:
        datasets.append(RGBDataset(r, RGB_transform=RGB_transform, frame_interval=args.frame_interval))

    dataset = ConcatDataset(datasets)

    total_size = len(dataset)
    train_size = int(total_size * args.train_split)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=lambda batch: pad_collate_fn(batch, args.sequence_length))

    return train_loader, test_loader

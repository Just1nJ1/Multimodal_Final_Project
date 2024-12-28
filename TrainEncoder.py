import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from datasets import ImageDataset, VideoDataset
from embeddings.VisionEmbedding import VideoEncoderWithPositionalEncoding
from run import store_csv


def parseargs():
    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument('--name', type=str, required=True, help="Model name")
    # parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    # parser.add_argument('--lr', type=float, required=False, default=0.001, help="Learning rate")
    # parser.add_argument('--warmup_steps', type=int, required=False, default=1000, help="Warmup steps")

    # Mamba Parameters
    # parser.add_argument('--mamba_dim', type=int, default=2, help='dimension of mamba hidden state')
    # parser.add_argument('--expand_factor', type=int, default=2, help='mamba hidden state expansion factor')
    # parser.add_argument('--n_layers', type=int, default=2, help='number of mamba layers')
    # parser.add_argument('--num_classes', type=int, default=60, help='number of classes')

    # Dataset Parameters
    parser.add_argument('--frame_interval', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--max_sequence_length', type=int, default=300)
    parser.add_argument('--encoder_dim', type=int, default=256)

    # Vision Encoder Parameters
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--phase1', type=int, default=50)
    parser.add_argument('--phase2', type=int, default=0)
    parser.add_argument('--phase1_lr', type=float, default=1e-3)
    parser.add_argument('--phase2_lr', type=float, default=1e-4)
    parser.add_argument('--update_steps', type=int, default=15)
    parser.add_argument('--bank', default=False, action='store_true')

    # Vision Encoder - ViT
    parser.add_argument('--model', type=str, default='NFNET')
    parser.add_argument('--blocks', type=int, default=4)

    # Other Parameters
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--seed', type=int, required=False, default=np.random.randint(2**30), help="Random seed")
    parser.add_argument('--result_path', type=str, required=False, default=f"{os.getcwd()}/results", help="Results folder")
    parser.add_argument('--ckpt_path', type=str, required=False, default=f"/common/users/jj740/Multimodal/ckpts", help="Checkpoints folder")
    parser.add_argument('--data_path', type=str, required=False, default=f"/common/users/jj740/data", help="Data folder")
    parser.add_argument('--test', default=False, action='store_true', help="Test mode")
    parser.add_argument('--performance', default=False, action='store_true', help="Test performance")
    parser.add_argument('--depth', default=False, action='store_true', help="Train depth")

    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    np.random.seed(seed)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        embeddings = torch.nn.functional.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)

        # Positive pairs
        pos_sim = torch.exp(similarity_matrix) * mask
        neg_sim = torch.exp(similarity_matrix) * (1 - mask)

        # Contrastive loss
        loss = -torch.log(pos_sim.sum(1) / (pos_sim.sum(1) + neg_sim.sum(1)))
        return loss.mean()


def train(rgb_model, train_loader, device, args):
    rgb_model.to(device)
    rgb_model.freeze()

    rgb_optimizer = torch.optim.Adam(
        [param for param in rgb_model.parameters() if param.requires_grad],
        lr=args.phase1_lr,
    )

    contrastive_loss = ContrastiveLoss(temperature=0.07)

    ##### BANK #####
    rgb_memory_bank = []
    label_bank = []
    ##### BANK #####

    rgb_losses = []

    best_loss = 1000000

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(args.phase1):
        rgb_epoch_loss = 0
        rgb_model.train()

        for step, (video_frames, label, _) in (pbar := tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1} / {args.phase1}')):
            video_frames, label = video_frames.to(device), label.to(device)
            B, T, C, H, W = video_frames.shape
            video_frames = video_frames.view(B * T, C, H, W)
            label = label.repeat_interleave(T)

            rgb_embeddings = rgb_model(video_frames)

            if args.bank:
                rgb_memory_bank.append(rgb_embeddings)
                label_bank.append(label)
                if (step + 1) % args.update_steps == 0:
                    rgb_memory_bank = torch.cat(rgb_memory_bank, dim=0)
                    label_bank = torch.cat(label_bank, dim=0)

                    rgb_loss = contrastive_loss(rgb_memory_bank, label_bank)

                    rgb_optimizer.zero_grad()
                    rgb_loss.backward()
                    rgb_optimizer.step()

                    # Clear memory bank after update
                    rgb_memory_bank = []
                    label_bank = []

                    rgb_losses.append(rgb_loss.item())
                    rgb_epoch_loss += rgb_loss.item()

                    pbar.set_postfix({'rgb_loss': rgb_loss.item()})
            else:
                rgb_loss = contrastive_loss(rgb_embeddings, label)

                rgb_optimizer.zero_grad()
                rgb_loss.backward()
                rgb_optimizer.step()

                rgb_losses.append(rgb_loss.item())
                rgb_epoch_loss += rgb_loss.item()

                pbar.set_postfix({'rgb_loss': rgb_loss.item()})

            if args.test:
                if step >= 50:
                    if not args.depth:
                        torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_rgb_best.pth')
                    else:
                        torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_depth_best.pth')
                    sys.exit(0)

        if rgb_epoch_loss < best_loss:
            best_loss = rgb_epoch_loss
            if not args.depth:
                torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_rgb_best.pth')
            else:
                torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_depth_best.pth')
            print("Best ckpt Update")

        if not args.depth:
            store_csv(rgb_losses, f'{args.name}_rgb_loss.csv', args.result_path, header=f'Phase 1 Epoch: {epoch + 1}')
        else:
            store_csv(rgb_losses, f'{args.name}_depth_loss.csv', args.result_path, header=f'Phase 1 Epoch: {epoch + 1}')

    if not args.depth:
        store_csv(rgb_losses, f'{args.name}_rgb_loss.csv', args.result_path, header=f'Phase 1')
    else:
        store_csv(rgb_losses, f'{args.name}_depth_loss.csv', args.result_path, header=f'Phase 1')

    rgb_losses = []

    rgb_memory_bank = []
    label_bank = []

    rgb_model.unfreeze()

    if args.model == 'NFNET':
        rgb_optimizer = torch.optim.Adam([
            {'params': rgb_model.frame_encoder.stages[3].parameters(), 'lr': args.phase1_lr},
            {'params': rgb_model.frame_encoder.final_conv.parameters(), 'lr': args.phase1_lr},
            {'params': rgb_model.post_proj.parameters(), 'lr': args.phase2_lr},
        ])
    elif args.model == 'ViT_L':
        rgb_optimizer = torch.optim.Adam([
            {'params': rgb_model.frame_encoder.parameters(), 'lr': args.phase1_lr},
            {'params': rgb_model.post_proj.parameters(), 'lr': args.phase2_lr},
        ])
    elif args.model == 'ViT_Base':
        rgb_optimizer = torch.optim.Adam([
            {'params': rgb_model.frame_encoder.parameters(), 'lr': args.phase1_lr},
            {'params': rgb_model.post_proj.parameters(), 'lr': args.phase2_lr},
        ])


    for epoch in range(args.phase2):
        rgb_epoch_loss = 0
        rgb_model.train()

        for step, (video_frames, label, _) in (pbar := tqdm(enumerate(train_loader), desc=f'Epoch {epoch + 1} / {args.phase2}')):
            video_frames, label = video_frames.to(device), label.to(device)
            B, T, C, H, W = video_frames.shape
            video_frames = video_frames.view(B * T, C, H, W)
            label = label.repeat_interleave(T)

            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                rgb_embeddings = rgb_model(video_frames)

                if args.bank:
                    rgb_memory_bank.append(rgb_embeddings)
                    label_bank.append(label)
                    if (step + 1) % args.update_steps == 0:
                        rgb_memory_bank = torch.cat(rgb_memory_bank, dim=0)
                        label_bank = torch.cat(label_bank, dim=0)

                        rgb_loss = contrastive_loss(rgb_memory_bank, label_bank)

                        scaler.scale(rgb_loss).backward()
                        scaler.step(rgb_optimizer)
                        scaler.update()

                        rgb_optimizer.zero_grad()

                        # Clear memory bank after update
                        rgb_memory_bank = []
                        label_bank = []

                        rgb_losses.append(rgb_loss.item())
                        rgb_epoch_loss += rgb_loss.item()

                        pbar.set_postfix({'rgb_loss': rgb_loss.item()})
                else:
                    rgb_loss = contrastive_loss(rgb_embeddings, label)

                    rgb_optimizer.zero_grad()
                    rgb_loss.backward()
                    rgb_optimizer.step()

                    rgb_losses.append(rgb_loss.item())
                    rgb_epoch_loss += rgb_loss.item()

                    pbar.set_postfix({'rgb_loss': rgb_loss.item()})

        if rgb_epoch_loss < best_loss:
            best_loss = rgb_epoch_loss
            if not args.depth:
                torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_rgb_best.pth')
            else:
                torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_depth_best.pth')
            print("Best ckpt Update")

        if not args.depth:
            store_csv(rgb_losses, f'{args.name}_rgb_loss.csv', args.result_path, header=f'Phase 2 Epoch: {epoch + 1}')
        else:
            store_csv(rgb_losses, f'{args.name}_depth_loss.csv', args.result_path, header=f'Phase 2 Epoch: {epoch + 1}')

        rgb_losses = []

    # if not args.depth:
    #     store_csv(rgb_losses, f'{args.name}_rgb_loss.csv', args.result_path, header=f'Phase 2')
    # else:
    #     store_csv(rgb_losses, f'{args.name}_depth_loss.csv', args.result_path, header=f'Phase 2')


if __name__ == '__main__':
    args = parseargs()
    args.sequence_length = args.max_sequence_length // args.frame_interval
    torch.cuda.empty_cache()
    if args.test:
        breakpoint()

    set_seed(args.seed)
    print("########### Parameters ###########")
    for arg, value in vars(args).items():
        print(f"{arg:<15}{str(value):<10}")

    print("########## Load Dataset ##########")
    if not args.depth:
        train_loader, test_loader = VideoDataset.data_loader(
            RGB_path=[f'{args.data_path}/NTURGBD/nturgbd_rgb_s00{i}' for i in range(1, 3)],
            args=args
        )
    else:
        train_loader, test_loader = ImageDataset.data_loader(
            Depth_path=f'{args.data_path}/NTURGBD/nturgb+d_depth_masked',
            args=args
        )

    print("########### Load Model ###########")
    rgb_model = VideoEncoderWithPositionalEncoding(args)

    if args.load:
        if args.depth:
            rgb_model.load_state_dict(torch.load(f'{args.ckpt_path}/{args.name}_depth_best.pth'))
        else:
            rgb_model.load_state_dict(torch.load(f'{args.ckpt_path}/{args.name}_rgb_best.pth'))

    total_params = sum(p.numel() for p in rgb_model.parameters())
    print(f"Model Size (number of parameters): {total_params}")
    model_size_bytes = sum(p.numel() * p.element_size() for p in rgb_model.parameters())
    print(f"Model Size (in bytes): {model_size_bytes / (1024 ** 2):.2f} MB")

    device = torch.device(args.device) if args.device is not None else torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("########## Devices used ##########")
    print(device)
    pathlib.Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)

    print("######### Start Training #########")
    train(
        rgb_model=rgb_model,
        train_loader=train_loader,
        device=device,
        args=args
    )

    print("########## Model Saving ##########")
    pathlib.Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
    if not args.depth:
        torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_rgb_final.pth')
    else:
        torch.save(rgb_model.state_dict(), f'{args.ckpt_path}/{args.name}_depth_final.pth')
    print(f"Model Saved Successfully at {args.ckpt_path}")
    torch.cuda.empty_cache()
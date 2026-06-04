import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model_experimental import MNISTDiffusion
from utils import ExponentialMovingAverage

import os
import math
import argparse
from tqdm.auto import tqdm
import csv
from typing import Optional

from torchmetrics.image.fid import FrechetInceptionDistance

def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # [0,1] to [-1,1]
    ])

    train_dataset = MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root="./mnist_data", train=False, download=True, transform=preprocess)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),\
           DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--experiment', type=str, choices=['uniform', 'A', 'B', 'C', 'D', 'all'], default='uniform', help='Choose the timestep sampling distribution')
    parser.add_argument('--noise_schedule', type=str, choices=['linear', 'cosine'], default='cosine', help='Choose the noise schedule')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)    
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ckpt', type=str, help='define checkpoint path', default='')
    parser.add_argument('--n_samples', type=int, help='define sampling amounts after every epoch trained', default=36)
    parser.add_argument('--model_base_dim', type=int, help='base dim of Unet', default=64)
    parser.add_argument('--timesteps', type=int, help='sampling steps of DDPM', default=1000)
    parser.add_argument('--model_ema_steps', type=int, help='ema model evaluation interval', default=10)
    parser.add_argument('--model_ema_decay', type=float, help='ema model decay', default=0.995)
    parser.add_argument('--no_clip', action='store_true', help='set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu', action='store_true', help='cpu training')

    # Evaluation Arguments
    parser.add_argument('--eval_freq', type=int, help='frequency of epochs to run intermediate FID and save grid', default=8)
    parser.add_argument('--fid_eval_samples', type=int, help='number of samples for intermediate fast FID', default=1000)
    parser.add_argument('--fid_final_samples', type=int, help='number of samples for final FID', default=10000)
    parser.add_argument('--fid_batch_size', type=int, help='batch size for generating FID images', default=256)
    parser.add_argument('--fid_dir', type=str, help='directory to save final FID images when --save_fid_images is set', default='results/fid_samples')
    
    return parser.parse_args()


def real_mnist_to_uint8_rgb(images: torch.Tensor) -> torch.Tensor:
    images = ((images.detach().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError(f"FID expects 1 or 3 image channels, got {images.shape[1]}.")
    return images

def generated_to_uint8_rgb(images: torch.Tensor) -> torch.Tensor:
    images = (images.detach().clamp(-1.0, 1.0) * 255.0).round().to(torch.uint8)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError(f"FID expects 1 or 3 image channels, got {images.shape[1]}.")
    return images

def select_device(use_cpu: bool) -> str:
    if use_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def fid_metric_device(device: str) -> torch.device:
    device = torch.device(device)
    return device if device.type == "cuda" else torch.device("cpu")

@torch.no_grad()
def build_fid_metric_with_reals(
    real_loader: DataLoader,
    *,
    num_samples: int,
    device: str,
    feature: int = 2048
) -> FrechetInceptionDistance:
    metric_device = fid_metric_device(device)
    metric = FrechetInceptionDistance(feature=feature, normalize=False, reset_real_features=False).to(metric_device)

    seen = 0
    real_total = math.ceil(num_samples / real_loader.batch_size) if real_loader.batch_size is not None else None
    for real, _ in tqdm(real_loader, desc="     FID real", total=real_total, leave=False):
        real = real[: max(0, num_samples - seen)]
        if real.numel() == 0:
            break
        real = real.to(metric_device, non_blocking=True)
        metric.update(real_mnist_to_uint8_rgb(real), real=True)
        seen += real.shape[0]
        if seen >= num_samples:
            break

    return metric

@torch.no_grad()
def compute_fid(
    model: nn.Module,
    real_loader: DataLoader,
    *,
    num_samples: int,
    batch_size: int,
    device: str,
    no_clip: bool = False,
    real_metric: Optional[FrechetInceptionDistance] = None,
    feature: int = 2048
    ) -> float:
    metric_device = fid_metric_device(device)
    if real_metric is None:
        metric = build_fid_metric_with_reals(
            real_loader,
            num_samples=num_samples,
            device=device,
            feature=feature,
        )
    else:
        metric = real_metric
        metric.reset()
    model.eval()

    # Process FAKE images
    generated = 0
    num_batches = math.ceil(num_samples / batch_size)
    fid_progress = tqdm(range(num_batches), desc="     FID fake", leave=False)
    
    for _ in fid_progress:
        current_batch = min(batch_size, num_samples - generated)
        samples = model.sampling(current_batch, clipped_reverse_diffusion=not no_clip, device=device)
        samples = samples.to(metric_device, dtype=torch.float32, non_blocking=True)
        metric.update(generated_to_uint8_rgb(samples), real=False)
        generated += current_batch

    score = float(metric.compute().item())
    metric.reset()
    return score

def main(args):
    device = select_device(args.cpu)
    
    # Initialise dataloaders ONCE outside the loop to avoid redundant data loading
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=args.batch_size, image_size=28)
    
    # Prepare real FID statistics ONCE outside the loop to save time
    eval_real_metric = None
    if args.eval_freq > 0:
        print(f"Preparing real FID stats for {args.fid_eval_samples} eval images...")
        eval_real_metric = build_fid_metric_with_reals(
            test_dataloader,
            num_samples=args.fid_eval_samples,
            device=device,
            feature=2048,
        )

    # Determine the queue of experiments based on the command-line choice
    if args.experiment == 'all':
        experiments_to_run = ['uniform', 'A', 'B', 'C', 'D']
    else:
        experiments_to_run = [args.experiment]

    if args.noise_schedule == 'linear':
        noise_schedule_fn = 'linear' 
    else:        
        noise_schedule_fn = 'cosine'

    # Sequentially run each scheduled configuration
    for current_exp in experiments_to_run:
        print("\n" + "="*70)
        print(f" >>> STARTING EXPERIMENT: {current_exp} <<<")
        print("="*70 + "\n")

        # Initialise a fresh model instance with the current experiment setting
        model = MNISTDiffusion(timesteps=args.timesteps,
                               image_size=28,
                               in_channels=1,
                               base_dim=args.model_base_dim,
                               dim_mults=[2,4],
                               noise_schedule=noise_schedule_fn,
                               experiment=current_exp).to(device)

        # Torchvision EMA setting (re-calculated fresh for this run)
        adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs*len(train_dataloader), pct_start=0.25, anneal_strategy='cos')
        loss_fn = nn.MSELoss(reduction='mean')

        if args.ckpt:
            ckpt = torch.load(args.ckpt)
            model_ema.load_state_dict(ckpt["model_ema"])
            model.load_state_dict(ckpt["model"])

        global_steps = 0
        
        # Define the isolated directory using the current loop's experiment name
        base_dir = os.path.join("results", f"experiment_{current_exp}_{args.epochs}_{noise_schedule_fn}")
        os.makedirs(base_dir, exist_ok=True)

        print(f"Training pipeline started for Experiment {current_exp}...")
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"[{current_exp}] Epoch {epoch}/{args.epochs}", leave=True)
            
            for j, (image, target) in enumerate(progress_bar):
                noise = torch.randn_like(image).to(device)
                image = image.to(device)
                
                pred = model(image, noise)
                loss = loss_fn(pred, noise)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                if global_steps % args.model_ema_steps == 0:
                    model_ema.update_parameters(model)
                    
                current_loss = loss.detach().cpu().item()
                epoch_loss += current_loss
                global_steps += 1
                
                progress_bar.set_postfix(
                    loss=f"{current_loss:.4f}", 
                    lr=f"{scheduler.get_last_lr()[0]:.6f}"
                )
                
            avg_loss = epoch_loss / len(train_dataloader)

            # ---------------------------------------------------------
            # Intermediate Evaluation & Checkpointing
            # ---------------------------------------------------------
            if args.eval_freq > 0 and epoch % args.eval_freq == 0:
                print(f"\n  -> Saving checkpoint and running evaluation for Epoch {epoch}...")
                
                ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
                torch.save(ckpt, os.path.join(base_dir, f"steps_{current_exp}_{epoch:03d}_{noise_schedule_fn}.pt"))
                
                model_ema.eval()
                
                with torch.no_grad():
                    grid_samples = model_ema.module.sampling(args.n_samples, clipped_reverse_diffusion=not args.no_clip, device=device)
                grid_path = os.path.join(base_dir, f"exp_{current_exp}_epoch_{epoch:03d}_{noise_schedule_fn}.png")
                save_image(grid_samples.to(torch.float32), grid_path, nrow=int(math.sqrt(args.n_samples)))

                score_val = compute_fid(
                    model=model_ema.module,
                    real_loader=test_dataloader,
                    num_samples=args.fid_eval_samples,
                    batch_size=args.fid_batch_size,
                    device=device,
                    no_clip=args.no_clip,
                    real_metric=eval_real_metric,
                )
                
                print(f"  -> Intermediate FID Score ({args.fid_eval_samples} samples): {score_val:.4f}\n")

                csv_path = os.path.join(base_dir, "intermediate_fid_scores.csv")
                file_exists = os.path.isfile(csv_path)
                
                with open(csv_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["Epoch", "FID_Score"])
                    writer.writerow([epoch, f"{score_val:.4f}"])

        # Final checkpoint catch-all for this specific experiment
        if args.epochs % args.eval_freq != 0 or args.eval_freq == 0:
            print("\n  -> Saving final training checkpoint...")
            ckpt = {"model": model.state_dict(), "model_ema": model_ema.state_dict()}
            torch.save(ckpt, os.path.join(base_dir, f"steps_{current_exp}_final_{noise_schedule_fn}.pt"))

        # ---------------------------------------------------------
        # Final Evaluation (10k Samples)
        # ---------------------------------------------------------
        print(f"\nTraining complete for {current_exp}. Calculating final FID score for {args.fid_final_samples} samples...")
        
        final_score = compute_fid(
            model=model_ema.module,
            real_loader=test_dataloader,
            num_samples=args.fid_final_samples,
            batch_size=args.fid_batch_size,
            device=device,
            no_clip=args.no_clip,
            real_metric=None  # Passing None forces it to compute 10k real stats freshly
        )

        print(f"\n>>> [Experiment {current_exp}] Final FID Score: {final_score:.4f} <<<")
        
        # Save the final score to the CSV file
        csv_path = os.path.join(base_dir, "intermediate_fid_scores.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Epoch", "FID_Score"])
            # Label as "Final (10k)" to distinguish from intermediate epoch scores
            writer.writerow(["Final (10k)", f"{final_score:.4f}"])
            
    print("\n========================================================")
    print(" ALL SCHEDULED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("========================================================\n")

if __name__=="__main__":
    args = parse_args()
    main(args)

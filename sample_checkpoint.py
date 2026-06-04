import torch
import os
import math
import argparse
from torchvision.utils import save_image
from model_experimental import MNISTDiffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from a trained MNISTDiffusion checkpoint")
    
    # Checkpoint and output arguments
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the saved checkpoint (.pt file)')
    parser.add_argument('--out_dir', type=str, default='generated_samples', help='Directory to save the generated images')
    parser.add_argument('--filename', type=str, default='sample.png', help='Filename for the output grid image')
    
    # Sampling arguments
    parser.add_argument('--n_samples', type=int, default=36, help='Number of images to generate')
    parser.add_argument('--no_clip', action='store_true', help='Set to normal sampling method without clip x_0')
    parser.add_argument('--cpu', action='store_true', help='Force CPU generation')
    
    # Model architecture arguments (must match how the model was trained)
    parser.add_argument('--experiment', type=str, choices=['uniform', 'A', 'B', 'C', 'D'], default='uniform', help='Timestep sampling distribution used during training')
    parser.add_argument('--model_base_dim', type=int, default=64, help='Base dim of Unet used during training')
    parser.add_argument('--timesteps', type=int, default=1000, help='Sampling steps of DDPM')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialise device
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Initialise a fresh model instance with the exact settings used during training
    print("Initialising model architecture...")
    model = MNISTDiffusion(
        timesteps=args.timesteps,
        image_size=28,
        in_channels=1,
        base_dim=args.model_base_dim,
        dim_mults=[2, 4],
        experiment=args.experiment
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # Prioritise EMA weights for sampling as they usually provide better visual quality
    if "model_ema" in checkpoint:
        print("Loading EMA weights...")
        # Since model_ema's state dict structurally matches the base model, 
        # we can load it directly into our standard model instance.
        model.load_state_dict(checkpoint["model_ema"])
    elif "model" in checkpoint:
        print("EMA weights not found, loading standard model weights...")
        model.load_state_dict(checkpoint["model"])
    else:
        # Fallback just in case the checkpoint was saved differently
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Generate samples
    print(f"Generating {args.n_samples} samples...")
    with torch.no_grad():
        samples = model.sampling(
            n_samples=args.n_samples, 
            clipped_reverse_diffusion=not args.no_clip, 
            device=device
        )
    
    # Save the generated images to a grid
    out_path = os.path.join(args.out_dir, args.filename)
    # Convert from [-1, 1] (or model output space) properly before saving
    save_image(
        samples.to(torch.float32), 
        out_path, 
        nrow=int(math.sqrt(args.n_samples)), 
        normalize=True, 
        value_range=(-1, 1)
    )
    
    print(f"Successfully saved samples to {out_path}")

if __name__ == "__main__":
    main()



# How to run example:
# python sample_checkpoint.py --ckpt results/experiment_uniform_40/steps_uniform_final.pt
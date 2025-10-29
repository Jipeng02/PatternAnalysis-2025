"""
Improved Training Script for MambaIRv2 Colorization
Uses ImageNet-1K-128x128 dataset with dynamic sampling and robust NaN handling
"""

import os, random, torch
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from module import MambaIRv2
from dataset import HFImageNetDataset
from utils import combined_loss_safe
import lpips
from datasets import load_dataset
import argparse

# ==== Argument Parser ====
parser = argparse.ArgumentParser(description='Train MambaIRv2 on ImageNet-1K-128x128')
parser.add_argument('--num_classes', type=int, default=100, 
                    help='Number of ImageNet classes to use (default: 100, max: 1000)')
parser.add_argument('--samples_per_epoch', type=int, default=3000,
                    help='Number of samples per epoch (default: 3000)')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size (default: 1)')
parser.add_argument('--accumulation_steps', type=int, default=4,
                    help='Gradient accumulation steps (default: 4)')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='Learning rate (default: 5e-5)')
parser.add_argument('--pretrained_path', type=str, default='./mambairv2_ColorDN_15.pth',
                    help='Path to pretrained weights')
parser.add_argument('--save_dir', type=str, default='./checkpoints_imagenet',
                    help='Directory to save checkpoints')
parser.add_argument('--vis_dir', type=str, default='./vis_imagenet',
                    help='Directory to save visualizations')
args = parser.parse_args()

# ==== Environment Setup ====
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ==== Create directories ====
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.vis_dir, exist_ok=True)

# ==== Model ====
print("\n=== Initializing Model ===")
model = MambaIRv2(
    img_size=128, patch_size=1, in_chans=3,
    embed_dim=174, d_state=16,
    depths=(6,6,6,6,6,6), num_heads=[6,6,6,6,6,6],
    window_size=16, inner_rank=64, num_tokens=128,
    convffn_kernel_size=5, mlp_ratio=2.0,
    upsampler='', upscale=1,
    resi_connection='1conv', img_range=1.0
).to(device)

# Load pretrained weights
if os.path.exists(args.pretrained_path):
    ckpt = torch.load(args.pretrained_path, map_location="cpu")
    state_dict = ckpt.get("params", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"✓ Loaded pretrained weights: {args.pretrained_path}")
else:
    print(f"⚠ Pretrained model not found at {args.pretrained_path}, training from scratch")

# Unfreeze all parameters
for p in model.parameters():
    p.requires_grad = True
print("✓ Unfroze all parameters for full fine-tuning")

# ==== Load HuggingFace ImageNet-128 ====
print("\n=== Loading Dataset ===")
print("Loading benjamin-paine/imagenet-1k-128x128...")
hf_dataset = load_dataset("benjamin-paine/imagenet-1k-128x128", split="train")
print(f"Original dataset size: {len(hf_dataset)}")

# Analyze class distribution
all_labels = [item['label'] for item in hf_dataset]
unique_labels = sorted(list(set(all_labels)))
print(f"Total classes: {len(unique_labels)}")
print(f"Average samples per class: {len(all_labels) / len(unique_labels):.0f}")

# Class subset
if args.num_classes < 1000:
    print(f"\nUsing subset: first {args.num_classes} classes")
    selected_labels = unique_labels[:args.num_classes]
    filtered_indices = [i for i, label in enumerate(all_labels) if label in selected_labels]
    hf_dataset = hf_dataset.select(filtered_indices)
    print(f"Filtered dataset size: {len(hf_dataset)} ({len(hf_dataset)/len(all_labels)*100:.1f}%)")
else:
    print(f"\nUsing full dataset: {len(hf_dataset)}")

# ==== Transform ====
transform = T.Compose([T.Resize((128,128)), T.ToTensor()])

# ==== Dynamic Sampling Function ====
def get_loader(hf_dataset, transform, batch_size=1, samples_per_epoch=3000):
    """Create dataloader with random sampling"""
    num_samples = min(samples_per_epoch, len(hf_dataset))
    idxs = random.sample(range(len(hf_dataset)), num_samples)
    subset = hf_dataset.select(idxs)
    dataset = HFImageNetDataset(subset, transform, use_augmentation=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=2, pin_memory=False)
    return loader

# ==== Optimizer & Scheduler ====
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': args.lr, 'weight_decay': 1e-5}
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=1e-7
)

# ==== LPIPS ====
lpips_fn = lpips.LPIPS(net='vgg').to(device)
for p in lpips_fn.parameters():
    p.requires_grad = False

# ==== Visualization Function ====
def visualize_samples(epoch, gray, color, pred, vis_dir, max_show=4):
    """Save visualization grid"""
    gray = gray[:max_show].cpu()
    color = color[:max_show].cpu()
    pred = pred[:max_show].detach().cpu().clamp(0, 1)
    grid = torch.cat([gray, pred, color], dim=0)
    grid = vutils.make_grid(grid, nrow=max_show, padding=2)
    path = os.path.join(vis_dir, f"epoch_{epoch+1}.png")
    vutils.save_image(grid, path)
    print(f"  Saved visualization: {path}")

# ==== Training Loop ====
print("\n=== Training Started ===")
print(f"Epochs: {args.epochs}")
print(f"Samples per epoch: {args.samples_per_epoch}")
print(f"Batch size: {args.batch_size}")
print(f"Accumulation steps: {args.accumulation_steps}")
print(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
print(f"Learning rate: {args.lr}\n")

best_loss = float('inf')
nan_count = 0
max_nan_tolerance = 10

for epoch in range(args.epochs):
    # Create new dataloader with random sampling
    loader = get_loader(hf_dataset, transform, 
                       batch_size=args.batch_size, 
                       samples_per_epoch=args.samples_per_epoch)
    print(f"\nEpoch {epoch+1}/{args.epochs}: Sampled {min(args.samples_per_epoch, len(hf_dataset))} images")

    model.train()
    running_loss = running_loss_lab = running_loss_lpips = 0.0
    batch_count = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (gray, color) in enumerate(loader):
        gray = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        # Skip if NaN in input
        if torch.isnan(gray).any() or torch.isnan(color).any():
            continue

        # Forward pass
        pred = model(gray)
        
        # NaN check in prediction
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            nan_count += 1
            if nan_count > max_nan_tolerance:
                print(f"  ⚠ Too many NaN predictions, stopping epoch")
                break
            continue
        
        pred = pred.clamp(0, 1)
        
        # Compute loss with safe version
        loss, loss_lab, loss_lpips = combined_loss_safe(
            pred, color, lpips_fn, w_ab=0.8, w_L=0.1, w_lpips=0.3
        )

        # NaN check in loss
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > max_nan_tolerance:
                print(f"  ⚠ Too many NaN losses, stopping epoch")
                break
            continue

        # Gradient accumulation
        (loss / args.accumulation_steps).backward()

        running_loss += loss.item() * gray.size(0)
        running_loss_lab += loss_lab.item() * gray.size(0)
        running_loss_lpips += loss_lpips.item() * gray.size(0)
        batch_count += 1

        # Optimizer step
        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check gradient norm
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                continue
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            nan_count = 0  # Reset counter on successful step

        # Progress logging
        if (batch_idx + 1) % 100 == 0 and batch_count > 0:
            avg_total = running_loss / (batch_count * gray.size(0))
            avg_lab = running_loss_lab / (batch_count * gray.size(0))
            avg_lpips = running_loss_lpips / (batch_count * gray.size(0))
            print(f"  Batch {batch_idx+1}/{len(loader)}: "
                  f"Loss={avg_total:.6f} (Lab={avg_lab:.6f}, LPIPS={avg_lpips:.6f})")

    # Check if epoch was stopped early
    if nan_count > max_nan_tolerance:
        print(f"Epoch {epoch+1} stopped due to excessive NaN")
        break

    # Scheduler step
    scheduler.step()
    torch.cuda.empty_cache()

    # Epoch summary
    if batch_count > 0:
        avg_loss = running_loss / (batch_count * args.batch_size)
        avg_lab = running_loss_lab / (batch_count * args.batch_size)
        avg_lpips = running_loss_lpips / (batch_count * args.batch_size)
        current_lr = optimizer.param_groups[0]['lr']

        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Loss: {avg_loss:.6f} (Lab: {avg_lab:.6f}, LPIPS: {avg_lpips:.6f})")
        print(f"  LR: {current_lr:.2e} | Valid batches: {batch_count}/{len(loader)}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (Loss: {avg_loss:.6f})")

        # Visualization every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                gray, color = next(iter(loader))
                gray, color = gray.to(device), color.to(device)
                pred = model(gray).clamp(0, 1)
                visualize_samples(epoch, gray, color, pred, args.vis_dir)
        
        print("="*60 + "\n")
    else:
        print(f"Epoch {epoch+1}: All batches invalid (NaN detected)")
        break

# ==== Save final model ====
final_path = os.path.join(args.save_dir, "final_model.pth")
torch.save(model.state_dict(), final_path)

print("\n" + "="*60)
print("Training Completed!")
print(f"✓ Best Loss: {best_loss:.6f}")
print(f"✓ Final model: {final_path}")
print(f"✓ Best model: {os.path.join(args.save_dir, 'best_model.pth')}")
print(f"✓ Visualizations: {args.vis_dir}")
print("="*60)

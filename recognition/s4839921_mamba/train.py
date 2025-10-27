from module import MambaIRv2
from dataset import ColorizationDataset, AugColorizationDataset
from utils import colorization_loss, combined_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch, torch.nn 
from torch.utils.data import DataLoader
import torchvision.transforms as T
import lpips
model = MambaIRv2(
    img_size=128, # image size
    patch_size=1,
    in_chans=3,       
    embed_dim=174,
    d_state=16,
    depths=(6, 6, 6, 6, 6, 6),
    num_heads= [6, 6, 6, 6, 6, 6],
    window_size=16,
    inner_rank=64,
    num_tokens=128,
    convffn_kernel_size=5,
    mlp_ratio=2.0,
    upsampler='',  # set to '' for no upsampling
    upscale=1,  # no upsampling
    resi_connection='1conv'

)

#Stage 1: Train conv_first, conv_after_body, conv_last without augmentation
# ==== Device ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# ==== Load pretrained ====
ckpt_path = './mambairv2_ColorDN_15.pth'  # path to pretrained model (change to your path)
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint.get('params', checkpoint)
model.load_state_dict(state_dict, strict=False)

# ==== Freeze backbone; only train conv_first / conv_after_body / conv_last ====
for p in model.parameters():
    p.requires_grad = False
train_modules = []
if hasattr(model, 'conv_first'):       
    train_modules.append(model.conv_first)
if hasattr(model, 'conv_after_body'):  
    train_modules.append(model.conv_after_body)
if hasattr(model, 'conv_last'):        
    train_modules.append(model.conv_last)
for m in train_modules:
    for p in m.parameters():
        p.requires_grad = True

# ==== Dataset & DataLoader ====
img_dir   = 'YOUR_COCO2017/VAL2017'  # 5000 images for training
transform = T.Compose([T.Resize((128,128)), T.ToTensor()])
dataset   = ColorizationDataset(img_dir, transform=transform)
loader    = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

# ==== Optimizer ====
def param_groups_for_decay(modules):
    decay, no_decay = [], []
    for m in modules:
        for n,p in m.named_parameters():
            if not p.requires_grad: continue
            if n.endswith('bias') or 'bn' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
    return [{'params': decay, 'weight_decay': 1e-4, 'lr': 3e-4},
            {'params': no_decay, 'weight_decay': 0.0, 'lr': 3e-4}]

optimizer = torch.optim.AdamW(param_groups_for_decay(train_modules))

# ==== Training Loop ====
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_count = 0

    for batch_idx, (gray, color) in enumerate(loader):
        gray  = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        pred = model(gray)
        
        pred = pred.clamp(0, 1)
        
        loss = colorization_loss(pred, color, w_ab=1.0, w_L=0.15)

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item() * gray.size(0)
        batch_count += 1
        
        if (batch_idx + 1) % 50 == 0:
            avg_so_far = running_loss / (batch_count * gray.size(0))
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)}: Loss={avg_so_far:.6f}")

    avg_loss = running_loss / (batch_count * loader.batch_size)


# ==== Save last only ====
save_path = "./checkpoints/stage_1.pth" # change to your desired path
torch.save(model.state_dict(), save_path)
print(f"Saved final weights to: {save_path}")
print("Stage 1 training complete.")

# Stage 2: train entire model with augmentation and cosine annealing LR scheduler and LPIPS loss

# ==== Unfreeze all parameters ====
for p in model.parameters():
    p.requires_grad = True

# ==== Dataset & DataLoader ====
img_dir = './YOUR_COCO2017/VAL2017'  # 5000 images for training
transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
dataset = AugColorizationDataset(img_dir, transform=transform, use_augmentation=True)
batch_size = 1
accumulation_steps = 4  # Effective batch_size = 1 * 4 = 4
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)

# ==== Optimizer ====
optimizer = torch.optim.AdamW([
    {'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
])

# ==== Learning Rate Scheduler ====
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

# LPIPS loss
lpips_fn = lpips.LPIPS(net='vgg').to(device)
for p in lpips_fn.parameters():
    p.requires_grad = False

# ==== Training Loop ====
num_epochs = 15

best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_loss_lab = 0.0
    running_loss_lpips = 0.0
    batch_count = 0
    
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (gray, color) in enumerate(loader):
        gray = gray.to(device, non_blocking=True)
        color = color.to(device, non_blocking=True)

        # forward pass
        pred = model(gray)
        pred = pred.clamp(0, 1)
        
        # combined loss: Lab + LPIPS
        loss, loss_lab, loss_lpips = combined_loss(
            pred, color, lpips_fn,
            w_ab=1.0, w_L=0.15, w_lpips=0.5
        )

        # gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * accumulation_steps * gray.size(0)
        running_loss_lab += loss_lab.item() * gray.size(0)
        running_loss_lpips += loss_lpips.item() * gray.size(0)
        batch_count += 1

        # every accumulation_steps steps, update weights
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Clear cache periodically
            if (batch_idx + 1) % (accumulation_steps * 10) == 0:
                torch.cuda.empty_cache()
        
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_loss / (batch_count * gray.size(0))
            avg_lab = running_loss_lab / (batch_count * gray.size(0))
            avg_lpips = running_loss_lpips / (batch_count * gray.size(0))
            print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}: "
                  f"Total={avg_total:.6f} (Lab={avg_lab:.6f}, LPIPS={avg_lpips:.6f})")

    # end Epoch
    scheduler.step()
    torch.cuda.empty_cache()  # Epoch end cleanup
    
    if batch_count > 0:
        avg_loss = running_loss / (batch_count * batch_size)
        avg_loss_lab = running_loss_lab / (batch_count * batch_size)
        avg_loss_lpips = running_loss_lpips / (batch_count * batch_size)
        current_lr = optimizer.param_groups[0]['lr']
        
        print("\n" + "="*60)
        print(f"Epoch {epoch+1}/{num_epochs} finished")
        print(f"  Total Loss: {avg_loss:.6f} (Lab: {avg_loss_lab:.6f}, LPIPS: {avg_loss_lpips:.6f})")
        print(f"  Learning Rate: {current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./best_stage_2.pth") #change to your desired path
            print(f"  ✓ Saved best model (Loss: {avg_loss:.6f})")

        print("="*60 + "\n")


# ==== Save final model ====
save_path = "./checkpoints/stage_2.pth" # change to your desired path
torch.save(model.state_dict(), save_path)
print(f"\n✓ Training completed!")
print(f"✓ Final model saved to: {save_path}")
print(f"✓ Best model (Loss={best_loss:.6f}) saved to: ./checkpoints/best_stage_2.pth") #change to your desired path
print("Stage 2 training complete.")
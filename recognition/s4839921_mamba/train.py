from module import MambaIRv2
from dataset import ColorizationDataset
import torch, torch.nn 
from torch.utils.data import DataLoader
import torchvision.transforms as T
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
ckpt_path = './new_color_model_last_no_lora.pth'  # 你指定的初始权重
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

# ==== Lab utilities & ab-dominant loss ====
# Convert sRGB to CIE Lab
def _srgb_to_linear(x):
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, torch.pow((x+a)/(1+a) + 1e-8, 2.4))

def _rgb_to_xyz(rgb):  # [B,3,H,W] -> [B,3,H,W]
    rgb = rgb.clamp(0, 1)
    x = _srgb_to_linear(rgb).permute(0,2,3,1)  # [B,H,W,3]
    M = rgb.new_tensor([[0.4124564,0.3575761,0.1804375],
                        [0.2126729,0.7151522,0.0721750],
                        [0.0193339,0.1191920,0.9503041]])
    xyz = torch.matmul(x, M.T).permute(0,3,1,2).contiguous()
    return xyz

def _f_lab(t):
    d = 6/29
    return torch.where(t > d**3, torch.pow(t + 1e-8, 1/3), t/(3*d**2) + 4/29)

def rgb_to_lab(rgb):
    xyz = _rgb_to_xyz(rgb)
    Xn,Yn,YnZ = 0.95047,1.0,1.08883
    x = xyz[:,0]/Xn; y = xyz[:,1]/Yn; z = xyz[:,2]/YnZ
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L,a,b], dim=1)

def colorization_loss(pred_rgb, gt_rgb, w_ab=1.0, w_L=0.15):
    pred_lab = rgb_to_lab(pred_rgb)
    gt_lab   = rgb_to_lab(gt_rgb)
    L1,a1,b1 = pred_lab[:,0:1], pred_lab[:,1:2], pred_lab[:,2:3]
    L2,a2,b2 = gt_lab[:,0:1],   gt_lab[:,1:2],   gt_lab[:,2:3]
    loss_ab = (a1-a2).abs().mean() + (b1-b2).abs().mean()
    loss_L  = (L1-L2).abs().mean()
    return w_ab*loss_ab + w_L*loss_L

# ==== Training Loop ====
num_epochs = 10

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



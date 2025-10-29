import torch

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

def combined_loss(pred_rgb, gt_rgb, lpips_fn, w_ab=1.0, w_L=0.15, w_lpips=0.5):
    """Lab L1 loss + LPIPS perceptual loss"""
    # Lab loss
    loss_lab = colorization_loss(pred_rgb, gt_rgb, w_ab=w_ab, w_L=w_L)
    
    # LPIPS perceptual loss
    pred_rgb_norm = pred_rgb * 2.0 - 1.0
    gt_rgb_norm = gt_rgb * 2.0 - 1.0
    loss_lpips = lpips_fn(pred_rgb_norm, gt_rgb_norm).mean()
    
    total_loss = loss_lab + w_lpips * loss_lpips
    
    return total_loss, loss_lab, loss_lpips


# ==== Improved Loss Functions with NaN Handling ====
def _srgb_to_linear_safe(x):
    """Safe version with clamping"""
    a = 0.055
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x/12.92, ((x+a)/(1+a))**2.4)

def _rgb_to_xyz_safe(rgb):
    """Safe RGB to XYZ conversion"""
    rgb = rgb.clamp(0, 1)
    x = _srgb_to_linear_safe(rgb).permute(0,2,3,1)
    M = rgb.new_tensor([[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]])
    xyz = torch.matmul(x, M.T).permute(0,3,1,2).contiguous()
    return xyz.clamp(min=1e-8)

def _f_lab_safe(t):
    """Safe f function for Lab conversion"""
    d = 6/29
    t = t.clamp(min=1e-8)
    return torch.where(t > d**3, t.pow(1/3), t/(3*d**2) + 4/29)

def rgb_to_lab_safe(rgb):
    """Safe RGB to Lab conversion with NaN handling"""
    rgb = rgb.clamp(0, 1)
    xyz = _rgb_to_xyz_safe(rgb)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[:,0]/Xn, xyz[:,1]/Yn, xyz[:,2]/Zn
    fx, fy, fz = _f_lab_safe(x), _f_lab_safe(y), _f_lab_safe(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    lab = torch.stack([L, a, b], dim=1)
    
    # NaN check
    if torch.isnan(lab).any():
        return torch.zeros_like(lab)
    
    return lab

def colorization_loss_safe(pred_rgb, gt_rgb, w_ab=0.8, w_L=0.1):
    """Safe colorization loss with NaN handling"""
    pred_rgb = pred_rgb.clamp(0, 1)
    gt_rgb = gt_rgb.clamp(0, 1)
    
    pred_lab = rgb_to_lab_safe(pred_rgb)
    gt_lab = rgb_to_lab_safe(gt_rgb)
    
    L1, a1, b1 = pred_lab[:,0:1], pred_lab[:,1:2], pred_lab[:,2:3]
    L2, a2, b2 = gt_lab[:,0:1], gt_lab[:,1:2], gt_lab[:,2:3]
    
    loss_ab = (a1-a2).abs().mean() + (b1-b2).abs().mean()
    loss_L = (L1-L2).abs().mean()
    
    return w_ab * loss_ab + w_L * loss_L

def combined_loss_safe(pred_rgb, gt_rgb, lpips_fn, w_ab=0.8, w_L=0.1, w_lpips=0.3):
    """Safe combined loss with NaN handling"""
    pred_rgb = pred_rgb.clamp(0, 1)
    gt_rgb = gt_rgb.clamp(0, 1)
    
    loss_lab = colorization_loss_safe(pred_rgb, gt_rgb, w_ab, w_L)
    
    pred_norm = pred_rgb * 2 - 1
    gt_norm = gt_rgb * 2 - 1
    loss_lpips = lpips_fn(pred_norm, gt_norm).mean()
    
    # NaN handling
    if torch.isnan(loss_lab):
        loss_lab = torch.tensor(0.0, device=pred_rgb.device)
    if torch.isnan(loss_lpips):
        loss_lpips = torch.tensor(0.0, device=pred_rgb.device)
    
    total_loss = loss_lab + w_lpips * loss_lpips
    
    return total_loss, loss_lab, loss_lpips
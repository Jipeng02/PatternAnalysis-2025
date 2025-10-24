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
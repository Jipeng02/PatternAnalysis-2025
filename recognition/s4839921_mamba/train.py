from module import MambaIRv2
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


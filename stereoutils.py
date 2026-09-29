"""DSG-SD core utilities.

DSG-SD stands for Disparity-Structure Guided StereoDiffusion. This file
contains the training-free stereo-generation modules used in the project:
structure-guided disparity refinement, depth-boundary-aware SPSMD masking,
and structure-aware latent alignment. The pretrained Stable Diffusion and
DPT depth-estimation weights are not modified.
"""

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange,repeat
import gc
from torchvision.utils import save_image
import sys
sys.path.append('./stablediffusion')
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler

# ===================== DSG-SD ABLATION FLAGS START =====================
# Component flags can be controlled by environment variables.
DSG_USE_DPRIME = os.environ.get("DSG_USE_DPRIME", "1") == "1"
DSG_USE_ALIGN = os.environ.get("DSG_USE_ALIGN", "1") == "1"
DSG_USE_SPSMDMASK = os.environ.get("DSG_USE_SPSMDMASK", "1") == "1"
# ====================== DSG-SD ABLATION FLAGS END ======================


def _original_stereo_shift_torch(input_images, depthmaps,sacle_factor=8,shift_both = False,stereo_offset_exponent=1.0):
    """Apply the original StereoDiffusion latent stereo shift.

    This function shifts the left latent/image features according to the disparity
    map and returns a two-view tensor. DSG-SD keeps this function as the base
    stereo operation, then wraps it with structure-aware guidance.
    """
    '''input: [B, C, H, W] depthmap: [B, H, W]'''

    def _norm_depth(depth,max_val=1):
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)
        return out
    
    def _create_stereo(input_images,depthmaps,sacle_factor,stereo_offset_exponent):
        b, c, h, w = input_images.shape
        derived_image = torch.zeros_like(input_images)
        sacle_factor_px = (sacle_factor / 100.0) * input_images.shape[-1]
        if True:
            for batch in range(b):
                for row in range(h):
                    for col in range(w) if sacle_factor_px < 0 else range(w - 1, -1, -1):
                        col_d = col + int((depthmaps[batch,row,col] ** stereo_offset_exponent) * sacle_factor_px)
                        if 0 <= col_d < w:
                            derived_image[batch,:,row,col_d] = input_images[batch,:,row,col]
        return derived_image
    depthmaps = _norm_depth(depthmaps)
    
    if shift_both is False:
        left = input_images
        balance = 0
    else:
        balance = 0.5
        left = _create_stereo(input_images,depthmaps,+1 * sacle_factor * balance,stereo_offset_exponent)
    right = _create_stereo(input_images,depthmaps,-1 * sacle_factor * (1 - balance),stereo_offset_exponent)
    return torch.concat([left,right],axis=0)



class BNAttention():
    def __init__(self, start_step=4, total_steps=50, direction='uni'):

        self.total_steps = total_steps
        self.start_step = start_step
        self.cur_step = 0
        self.cur_att_layer = 0
        self.direction = direction

    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        n_samples =attn.shape[0]//num_heads // 2
        q = rearrange(q, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)
        k = rearrange(k, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)
        v = rearrange(v, "(s b h) n d -> (b h) (s n) d", h=num_heads,b=n_samples)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        del q,k
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(b h) (s n) d -> (s b) n (h d)",b=n_samples,s=2,h=num_heads)
        return out
    
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if is_cross or (self.cur_step < self.start_step):
            out = torch.einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            return out
        
        n_samples =attn.shape[0]//num_heads // 4
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        _num_heads = num_heads * n_samples
        if self.direction == 'bi':
            out_u = self.attn_batch(qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c = self.attn_batch(qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)
        elif self.direction == 'uni':
            out_u = self.attn_batch(qu, ku[:_num_heads], vu[:_num_heads], sim[:_num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c = self.attn_batch(qc, kc[:_num_heads], vc[:_num_heads], sim[:_num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)
        return out

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        self.cur_step = self.cur_att_layer//32
        return out


def regiter_attention_editor_diffusers(model, editor):
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if 'Attention' in net.__class__.__name__:
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    try:
        sub_nets = model.model.diffusion_model.named_children()
    except: 
        sub_nets = model.unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "input" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name or "output" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count

# ===================== DSG-SD STRUCTURE-GUIDED DISPARITY REFINEMENT START =====================
import torch
import torch.nn.functional as F

print("DSG-SD structure-guided disparity refinement loaded")


def _to_4d_tensor(x):
    """Convert a 2D/3D tensor into BCHW form and record the added dimensions."""
    added = 0

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        added = 2
    elif x.dim() == 3:
        x = x.unsqueeze(1)
        added = 1
    elif x.dim() == 4:
        added = 0
    else:
        return None, -1

    return x, added


def _from_4d_tensor(x, added):
    """Restore a BCHW tensor to its original dimensionality."""
    if added == 2:
        return x[0, 0]
    if added == 1:
        return x[:, 0]
    return x


def normalize_map_01(x, eps=1e-6):
    """Normalize each feature map independently to the [0, 1] range."""
    xmin = x.flatten(2).amin(-1).view(x.shape[0], x.shape[1], 1, 1)
    xmax = x.flatten(2).amax(-1).view(x.shape[0], x.shape[1], 1, 1)
    return ((x - xmin) / (xmax - xmin + eps)).clamp(0.0, 1.0)


def smooth_feature_map(x, k=3):
    """Apply replicate-padded average smoothing without changing spatial size."""
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(x, kernel_size=k, stride=1)


def avg_pool_with_replicate_padding(x, kernel_size):
    """Average-pool a map with replicate padding for stable local context estimation."""
    if isinstance(kernel_size, int):
        ky = kernel_size
        kx = kernel_size
    else:
        ky, kx = kernel_size

    pad_left = kx // 2
    pad_right = kx // 2
    pad_top = ky // 2
    pad_bottom = ky // 2

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return F.avg_pool2d(x, kernel_size=(ky, kx), stride=1)


def compute_spatial_gradient(x):
    """Compute a simple absolute horizontal-plus-vertical gradient map."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)

    gx[:, :, :, 1:] = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    gy[:, :, 1:, :] = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    return gx + gy


def compute_latent_structure_map(input_images, target_hw, target_batch, dtype, device):
    """Estimate a latent-space structure map from the current latent tensor.

    The map highlights connected image/latent structures that should receive
    more stable disparity treatment during stereo shifting.
    """
    try:
        if not torch.is_tensor(input_images):
            return None

        if input_images.dim() != 4:
            return None

        z = input_images.detach().float()
        z_gray = z.mean(dim=1, keepdim=True)

        if z_gray.shape[-2:] != target_hw:
            z_gray = F.interpolate(
                z_gray,
                size=target_hw,
                mode="bilinear",
                align_corners=False
            )

        if z_gray.shape[0] != target_batch:
            z_gray = z_gray[:1].expand(target_batch, -1, -1, -1)

        z_edge = normalize_map_01(compute_spatial_gradient(z_gray))
        return z_edge.to(device=device, dtype=dtype)

    except Exception:
        return None


def estimate_thin_structure_confidence(structure):
    """Estimate whether a structure map contains elongated thin visual structures."""
    h_line = avg_pool_with_replicate_padding(structure, (1, 7))
    v_line = avg_pool_with_replicate_padding(structure, (7, 1))
    thin_structure_confidence = torch.maximum(h_line, v_line)
    thin_structure_confidence = smooth_feature_map(thin_structure_confidence, 3)
    return thin_structure_confidence.clamp(0.0, 1.0)


def refine_disparity_for_structure(input_images, disp):
    """Refine the predicted disparity map before latent stereo shifting.

    This module strengthens weak but connected structures, smooths noisy or weak
    regions, and limits excessive changes near broad foreground objects. It is
    the structure-guided disparity refinement stage of DSG-SD.
    """
    # D-prime component gate
    if not DSG_USE_DPRIME:
        return disp

    if not torch.is_tensor(disp):
        return disp

    original_dtype = disp.dtype
    original_device = disp.device

    d4, added = _to_4d_tensor(disp)

    if d4 is None:
        return disp

    d4 = d4.float()

    sign = torch.sign(d4)
    mag = d4.abs()

    maxv = mag.flatten(2).amax(-1).view(mag.shape[0], mag.shape[1], 1, 1).clamp(min=1e-6)
    norm = (mag / maxv).clamp(0.0, 1.0)

    # 1. Disparity structure
    disp_edge = normalize_map_01(compute_spatial_gradient(norm))

    # 2. Latent structure
    latent_edge = compute_latent_structure_map(
        input_images=input_images,
        target_hw=d4.shape[-2:],
        target_batch=d4.shape[0],
        dtype=d4.dtype,
        device=d4.device
    )

    if latent_edge is None:
        latent_edge = torch.zeros_like(disp_edge)

    # 3. Combined structure map
    # Same spirit as the final structure-connected disparity refinement.
    structure = torch.maximum(disp_edge, latent_edge)
    structure = smooth_feature_map(structure, 3)
    structure = structure.clamp(0.0, 1.0)

    # 4. Connectivity propagation
    # Structure connectivity is propagated to stabilize weak but connected regions.
    conn = F.max_pool2d(structure, kernel_size=5, stride=1, padding=2)
    conn = smooth_feature_map(conn, 5)
    conn = conn.clamp(0.0, 1.0)

    # 5. Far / weak disparity mask
    far = ((0.48 - norm) / 0.48).clamp(0.0, 1.0)

    # 6. Gentle near-object guard
    # Protect only near/high-disparity broad regions.
    # Do NOT suppress far structured background.
    thin_structure_confidence = estimate_thin_structure_confidence(structure)

    blob = avg_pool_with_replicate_padding(structure, 9).clamp(0.0, 1.0)

    near = ((norm - 0.45) / 0.55).clamp(0.0, 1.0)

    near_object_guard = near * blob * (1.0 - 0.75 * thin_structure_confidence).clamp(0.0, 1.0)
    near_object_guard = smooth_feature_map(near_object_guard, 3)
    near_object_guard = near_object_guard.clamp(0.0, 1.0)

    # 7. Edge-aware smoothing
    # Edge-aware smoothing with tiny reduction inside the near-object guard.
    smooth = smooth_feature_map(d4, 3)

    smooth_w = (0.20 * (1.0 - structure) + 0.04 * far).clamp(0.0, 0.24)
    smooth_w = smooth_w * (1.0 - 0.18 * near_object_guard).clamp(0.0, 1.0)

    d_smooth = d4 * (1.0 - smooth_w) + smooth * smooth_w

    sign2 = torch.sign(d_smooth)
    mag2 = d_smooth.abs()
    norm2 = (mag2 / maxv).clamp(0.0, 1.0)

    # 8. Gamma lift for weak disparity
    # Use a fixed gamma lift for weak-disparity regions.
    gamma = 0.68
    lifted = norm2.pow(gamma) * maxv

    # 9. Structure-connected parallax floor
    # Use a structure-connected parallax floor with gentle reduction inside near-object regions.
    structured_far = (0.65 * structure + 0.35 * conn) * far

    floor = 0.13 * maxv * structured_far
    floor = floor * (1.0 - 0.20 * near_object_guard).clamp(0.0, 1.0)

    target_mag = torch.maximum(lifted, floor)

    # 10. Adaptive gain
    # Adaptive gain with a gentle near-object guard.
    risk = F.max_pool2d(disp_edge, kernel_size=3, stride=1, padding=1).clamp(0.0, 1.0)

    alpha_far = 0.26
    alpha_conn = 0.18
    alpha_risk = 0.07

    gain = 1.0
    gain = gain + alpha_far * structured_far
    gain = gain + alpha_conn * conn * far
    gain = gain - alpha_risk * risk * (1.0 - 0.50 * structure)

    # Gentle near-object guard.
    # Only reduces excessive boost on near blob-like regions.
    gain = gain * (1.0 - 0.08 * near_object_guard).clamp(0.0, 1.0)
    gain = gain.clamp(0.86, 1.22)
    out_mag = mag2 * gain

    # 11. Blend with target mainly in far regions
    # Blend toward the refined target mainly in far/structured regions, with gentle reduction near objects.
    blend = (0.58 * far + 0.16 * structure).clamp(0.0, 0.68)
    blend = blend * (1.0 - 0.25 * near_object_guard).clamp(0.0, 1.0)

    out_mag = out_mag * (1.0 - blend) + target_mag * blend

    # Safe upper bound to avoid excessive disparity amplification
    out_mag = torch.minimum(out_mag, 1.14 * maxv)

    d_out = sign2 * out_mag

    # 12. Partial mean preservation
    # Partial mean preservation to avoid cancelling useful improvement.
    original_mean = mag.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    refined_mean = d_out.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

    preserve = 0.72 + 0.28 * (original_mean / refined_mean)
    preserve = preserve.clamp(0.86, 1.08)

    d_out = d_out * preserve

    return _from_4d_tensor(d_out.to(dtype=original_dtype, device=original_device), added)


def _structure_refined_stereo_shift(*args, **kwargs):
    """Wrapper around the original stereo shift that refines disparity before shifting."""
    args = list(args)

    input_images = args[0] if len(args) >= 1 else kwargs.get("input_images", None)

    if len(args) >= 2 and torch.is_tensor(args[1]):
        args[1] = refine_disparity_for_structure(input_images, args[1])
        return _original_stereo_shift_torch(*args, **kwargs)

    for key in ["depthmaps", "depthmap", "disparity", "disp"]:
        if key in kwargs and torch.is_tensor(kwargs[key]):
            kwargs[key] = refine_disparity_for_structure(input_images, kwargs[key])
            break

    return _original_stereo_shift_torch(*args, **kwargs)

# ====================== DSG-SD STRUCTURE-GUIDED DISPARITY REFINEMENT END ======================


# ===================== DSG-SD STEREO SHIFT WRAPPER START =====================
def stereo_shift_torch(*args, **kwargs):
    """Run stereo shift using the structure-refined disparity cue."""
    return _structure_refined_stereo_shift(*args, **kwargs)

# ====================== DSG-SD STEREO SHIFT WRAPPER END ======================




# ===================== DSG-SD DEPTH-BOUNDARY-AWARE SPSMD MASKING START =====================
import torch
import torch.nn.functional as F

if "_DEPTH_SAFE_BASE_STEREO_SHIFT" not in globals():
    _DEPTH_SAFE_BASE_STEREO_SHIFT = stereo_shift_torch


def _to_4d_depth_safe(x):
    """Convert a disparity/depth tensor into BCHW form for boundary processing."""
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    return x


def normalize_depth_map_01(x, eps=1e-6):
    """Normalize a depth/disparity-derived map to the [0, 1] range."""
    xmax = x.amax(dim=(-2, -1), keepdim=True)
    xmin = x.amin(dim=(-2, -1), keepdim=True)
    return ((x - xmin) / (xmax - xmin + eps)).clamp(0.0, 1.0)


def compute_depth_boundary_mask(disparity):
    """Compute a normalized depth-boundary mask from the disparity map.

    High values indicate strong disparity changes where direct SPSMD copying is
    risky because it may copy across object/depth boundaries.
    """
    d = _to_4d_depth_safe(disparity).float()

    gx = torch.zeros_like(d)
    gy = torch.zeros_like(d)

    gx[:, :, :, 1:] = torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1])
    gy[:, :, 1:, :] = torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :])

    edge = normalize_depth_map_01(gx + gy)

    # Slight dilation around depth boundaries.
    edge = F.max_pool2d(edge, kernel_size=3, stride=1, padding=1)

    return edge.clamp(0.0, 1.0)


def resize_depth_boundary_to_mask(edge, mask):
    """Resize a depth-boundary map to match the SPSMD copy mask shape."""
    if not torch.is_tensor(edge) or not torch.is_tensor(mask):
        return edge

    if mask.dim() == 2:
        target_hw = mask.shape[-2:]
        target_b = 1
    elif mask.dim() == 3:
        target_hw = mask.shape[-2:]
        target_b = mask.shape[0]
    else:
        target_hw = mask.shape[-2:]
        target_b = mask.shape[0]

    edge = F.interpolate(
        edge.float(),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )

    if edge.shape[0] != target_b:
        edge = edge[:1].expand(target_b, -1, -1, -1)

    return edge


def suppress_copy_near_depth_boundaries(mask, disparity):
    """Reduce the SPSMD copy mask near strong depth boundaries.

    This keeps periodic latent refreshing in safe low-boundary areas while avoiding
    over-copying around occlusions and foreground-object edges.
    """
    # SPSMD mask component gate
    if not DSG_USE_SPSMDMASK:
        return mask

    if mask is None or not torch.is_tensor(mask):
        return mask

    edge = compute_depth_boundary_mask(disparity)
    edge = resize_depth_boundary_to_mask(edge, mask)

    if edge is None:
        return mask

    # Key idea:
    # Do NOT reduce disparity.
    # Only reduce SPSMD copy mask near depth boundaries.
    edge_threshold = 0.38

    if mask.dtype == torch.bool:
        if mask.dim() == 2:
            edge2 = edge[0, 0]
        elif mask.dim() == 3:
            edge2 = edge[:, 0]
        else:
            edge2 = edge

        released = mask & (edge2 < edge_threshold)
        return released

    release = (1.0 - 0.85 * edge).clamp(0.0, 1.0)

    if mask.dim() == 2:
        release = release[0, 0]
    elif mask.dim() == 3:
        release = release[:, 0]

    released = mask.float() * release.to(device=mask.device, dtype=mask.float().dtype)

    return released.to(device=mask.device)


def stereo_shift_torch(x, disparity, *args, **kwargs):
    """Run stereo shifting and suppress the returned copy mask near depth boundaries."""
    result = _DEPTH_SAFE_BASE_STEREO_SHIFT(x, disparity, *args, **kwargs)

    if isinstance(result, tuple) and len(result) >= 2:
        shifted = result[0]
        mask = result[1]
        new_mask = suppress_copy_near_depth_boundaries(mask, disparity)

        if len(result) == 2:
            return shifted, new_mask

        return (shifted, new_mask) + tuple(result[2:])

    return result


print("DSG-SD depth-boundary-aware SPSMD mask installed", flush=True)
# ====================== DSG-SD DEPTH-BOUNDARY-AWARE SPSMD MASKING END ======================



# ===================== DSG-SD STRUCTURE-AWARE LATENT ALIGNMENT START =====================
IMAGE_STRUCTURE_MASK = None
DEPTH_BOUNDARY_MASK = None

try:
    _STRUCTURE_ALIGN_BASE_STEREO_SHIFT = stereo_shift_torch
except Exception:
    _STRUCTURE_ALIGN_BASE_STEREO_SHIFT = None


def resize_guidance_mask(mask, ref):
    """Resize a guidance mask to the spatial size and batch size of a latent tensor."""
    if mask is None or not torch.is_tensor(mask):
        return None

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)

    out = F.interpolate(mask.float(), size=ref.shape[-2:], mode="bilinear", align_corners=False)

    if out.shape[0] != ref.shape[0]:
        out = out[:1].expand(ref.shape[0], -1, -1, -1)

    return out.to(device=ref.device, dtype=ref.dtype)


def apply_structure_aware_latent_alignment(orig_latent, shifted_latent):
    """Apply structure-aware latent alignment to the shifted right-view latent.

    The function uses image-structure and depth-boundary masks to strengthen
    safe thin/texture structures while suppressing copying across hard depth
    boundaries. This is the final structure-aware latent alignment stage of
    DSG-SD.
    """
    global IMAGE_STRUCTURE_MASK, DEPTH_BOUNDARY_MASK

    # Alignment component gate
    if not DSG_USE_ALIGN:
        return shifted_latent

    if IMAGE_STRUCTURE_MASK is None or DEPTH_BOUNDARY_MASK is None:
        return shifted_latent

    if not torch.is_tensor(orig_latent) or not torch.is_tensor(shifted_latent):
        return shifted_latent

    if orig_latent.shape != shifted_latent.shape:
        return shifted_latent

    structure = resize_guidance_mask(IMAGE_STRUCTURE_MASK, shifted_latent)
    depth_boundary = resize_guidance_mask(DEPTH_BOUNDARY_MASK, shifted_latent)

    if structure is None or depth_boundary is None:
        return shifted_latent

    # Thin image-structure cue:
    # strong image edge but weak depth boundary.
    safe_structure_region = (structure * (1.0 - depth_boundary)).clamp(0.0, 1.0)

    delta = shifted_latent - orig_latent

    # Preserve safe thin-structure displacement slightly more.
    delta = delta * (1.0 + 0.22 * safe_structure_region)

    # Avoid over-copying hard geometry boundaries.
    delta = delta * (1.0 - 0.08 * depth_boundary)

    out = orig_latent + delta
    return out


if _STRUCTURE_ALIGN_BASE_STEREO_SHIFT is not None:
    def stereo_shift_torch(*args, **kwargs):
        """Run final stereo shift with structure-aware latent alignment applied."""
        result = _STRUCTURE_ALIGN_BASE_STEREO_SHIFT(*args, **kwargs)

        try:
            orig = args[0]

            if torch.is_tensor(result):
                result = apply_structure_aware_latent_alignment(orig, result)
                print("DSG-SD structure-aware latent alignment applied", flush=True)
                return result

            if isinstance(result, tuple) and len(result) >= 1 and torch.is_tensor(result[0]):
                shifted = apply_structure_aware_latent_alignment(orig, result[0])
                print("DSG-SD structure-aware latent alignment applied", flush=True)
                return (shifted,) + tuple(result[1:])

        except Exception as e:
            print(f"DSG-SD structure-aware latent alignment skipped: {e}", flush=True)

        return result


print("DSG-SD structure-aware latent alignment loaded", flush=True)
# ====================== DSG-SD STRUCTURE-AWARE LATENT ALIGNMENT END ======================

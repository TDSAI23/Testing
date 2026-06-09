"""
Reference Mean Guidance (RMG) for ComfyUI
Based on: "Follow the Mean: Reference-Guided Flow Matching"
https://arxiv.org/abs/2605.10302

Implements equation 8 (endpoint mean blending) which is numerically stable
and equivalent to equation 11 (velocity correction):

    µ_guided = (1 - beta) * µ_model + beta * µ_reference

Works with any flow-matching model using ComfyUI's CONST sampling type
(AuraFlow, QIE 2511, SD3, etc.) where sigma = t ∈ [0, 1].

Usage:
  1. Load reference images representing your target style (SHRR outputs)
  2. Connect VAE for encoding references into latent space
  3. Set beta (guidance strength, 0.0–1.0, start at 0.3)
  4. Wire patched model into your existing KSampler/SamplerCustomAdvanced

For plain AuraFlow testing: use text-to-image, no image conditioning.
For QIE: image conditioning is in the conditioning dict, not in input_x,
so the correction applies cleanly to the noisy latent only.
"""

import math
import torch
import torch.nn.functional as F
import comfy.model_management


# ---------------------------------------------------------------------------
# Reference mean computation  (equation 6 from the paper)
# ---------------------------------------------------------------------------

def compute_reference_mean(ref_latents, x_t, t_scalar):
    """
    Closed-form endpoint mean over reference bank.

    Args:
        ref_latents : [M, C, H, W]  pre-encoded reference latents (on device)
        x_t         : [B, C, H, W]  current noisy latent
        t_scalar    : float          current timestep t ∈ [0, 1]  (= sigma for AuraFlow)

    Returns:
        mu_ref      : [B, C, H, W]  weighted sum of reference latents
    """
    M = ref_latents.shape[0]
    B = x_t.shape[0]

    # Protect against numerical blow-up when t ≈ 1  (denominator → 0)
    one_minus_t = max(1.0 - t_scalar, 1e-4)

    # Equation 6: w_m = softmax_m( -‖x_t - t·x_ref‖² / (2·(1-t)²) )
    # ref_latents: [M, C, H, W]  →  expand to [M, 1, C, H, W] for broadcast
    # x_t:         [B, C, H, W]  →  expand to [1, B, C, H, W] for broadcast
    refs_exp = ref_latents.unsqueeze(1)          # [M, 1, C, H, W]
    x_exp    = x_t.unsqueeze(0)                  # [1, B, C, H, W]

    diff     = x_exp - t_scalar * refs_exp        # [M, B, C, H, W]
    dist_sq  = (diff ** 2).sum(dim=(-3, -2, -1))  # [M, B]

    log_w    = -dist_sq / (2.0 * one_minus_t ** 2)
    weights  = torch.softmax(log_w, dim=0)        # [M, B]  sums to 1 over M

    # Weighted sum: [M, B, 1, 1, 1] * [M, 1, C, H, W] → sum over M → [B, C, H, W]
    w_exp    = weights.view(M, B, 1, 1, 1)
    mu_ref   = (w_exp * refs_exp).sum(dim=0)      # [B, C, H, W]

    return mu_ref


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class RMGuidance:
    """
    Reference Mean Guidance node.

    Patches the model to blend its endpoint mean prediction toward
    a bank of reference style images at each denoising step.

    Inputs
    ------
    model           : MODEL   — the flow-matching model to patch
    reference_images: IMAGE   — batch of style reference images [N, H, W, 3]
    vae             : VAE     — VAE used to encode reference images
    beta            : FLOAT   — guidance strength (0 = off, 1 = full reference)
    beta_schedule   : LIST    — "constant" or "cosine_decay"
                                cosine_decay reduces beta toward 0 at end of
                                schedule (preserves fine detail from base model)

    Notes
    -----
    - For AuraFlow plain text-to-image: works directly, no extra config needed.
    - For QIE 2511 with image conditioning: the image conditioning tensor lives
      in the conditioning dict 'c', not in input_x. The correction therefore
      applies only to the noisy latent channels, which is correct.
    - Reference images should be representative SHRR outputs at generation
      resolution. 5–20 images is sufficient per the paper.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("MODEL",),
                "reference_images": ("IMAGE",),   # [N, H, W, 3] float32 0..1
                "vae":              ("VAE",),
                "beta":             ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "beta_schedule":    (["constant", "cosine_decay"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    CATEGORY     = "advanced/guidance"

    # ------------------------------------------------------------------

    def patch(self, model, reference_images, vae, beta, beta_schedule):

        # ---- 1. Encode reference images into latent space ----------------
        device = comfy.model_management.get_torch_device()

        # reference_images: [N, H, W, 3] float32 0..1 (ComfyUI convention)
        # VAE expects [N, H, W, 3] on the right device
        with torch.no_grad():
            ref_latents = vae.encode(reference_images[:, :, :, :3])

        # Handle dict output from some VAE implementations
        if isinstance(ref_latents, dict):
            ref_latents = ref_latents.get("samples", list(ref_latents.values())[0])

        ref_latents = ref_latents.to(device).float()
        print(f"[RMG] ref_latents shape: {ref_latents.shape}, x_t will be logged on first step")

        # ---- 2. Build wrapper function -----------------------------------
        m = model.clone()

        # Determine sigma_max for cosine_decay schedule normalisation.
        # For AuraFlow multiplier=1.0 so sigma_max ≈ 1.0.
        try:
            sigma_max = float(m.model.model_sampling.sigma_max)
        except Exception:
            sigma_max = 1.0

        def rmg_wrapper(apply_model_fn, args):
            """
            Wraps the denoiser forward pass to apply RMG correction.

            args keys (from ComfyUI samplers.py):
                "input"        : x_t  [B, C, H, W]  noisy latent
                "timestep"     : sigma * multiplier  (= sigma for AuraFlow)
                "c"            : conditioning dict
                "cond_or_uncond": batch membership flags
            """
            x_t      = args["input"]
            timestep = args["timestep"]   # shape [B] or scalar

            # ---- sigma → t -----------------------------------------------
            # For AuraFlow: multiplier=1.0, so timestep = sigma = t ∈ [0,1]
            # timestep may be a 1-D tensor [B]; take the first element as scalar
            if isinstance(timestep, torch.Tensor):
                t_scalar = timestep.flatten()[0].item()
            else:
                t_scalar = float(timestep)

            # Clamp to valid range
            t_scalar = max(min(t_scalar, 1.0 - 1e-6), 1e-6)

            # ---- beta schedule -------------------------------------------
            if beta_schedule == "cosine_decay":
                # beta full at t=1 (high noise), decays to 0 at t=0 (clean)
                beta_t = beta * math.cos(math.pi * 0.5 * (1.0 - t_scalar))
            else:
                beta_t = beta

            # ---- run base model ------------------------------------------
            v_model = apply_model_fn(args["input"], args["timestep"], **args["c"])   # [B, C, H, W]  velocity output

            if beta_t == 0.0:
                return v_model

            # ---- endpoint mean from model output (CONST formulation) -----
            # CONST: x_0_pred = x_t - v * sigma    (calculate_denoised)
            # sigma = t for AuraFlow
            mu_model = x_t - v_model * t_scalar   # [B, C, H, W]

            # ---- reference endpoint mean (equation 6) --------------------
            # Move references to same device/dtype as x_t
            refs = ref_latents.to(dtype=x_t.dtype, device=x_t.device)

            # Log shapes on first step for debugging
            if t_scalar > 0.98:
                print(f"[RMG] x_t shape: {x_t.shape}, refs shape: {refs.shape}")

            # Resize reference latents to match x_t spatial dims if needed.
            # F.interpolate requires 4D input [N, C, H, W] — handle 3D case.
            if refs.shape[-2:] != x_t.shape[-2:]:
                need_squeeze = refs.dim() == 3
                if need_squeeze:
                    refs = refs.unsqueeze(1)   # [N, H, W] -> [N, 1, H, W]
                refs = F.interpolate(refs, size=x_t.shape[-2:], mode='bilinear', align_corners=False)
                if need_squeeze:
                    refs = refs.squeeze(1)     # [N, 1, H, W] -> [N, H, W]

            mu_ref = compute_reference_mean(refs, x_t, t_scalar)   # [B, C, H, W]

            # ---- blend endpoint means (equation 8) -----------------------
            mu_guided = (1.0 - beta_t) * mu_model + beta_t * mu_ref

            # ---- convert guided mean back to velocity --------------------
            # v = (x_t - x_0_pred) / sigma  →  v_guided = (x_t - mu_guided) / t
            v_guided = (x_t - mu_guided) / t_scalar

            return v_guided

        m.set_model_unet_function_wrapper(rmg_wrapper)
        return (m,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RMGuidance": RMGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RMGuidance": "Reference Mean Guidance (RMG)",
}

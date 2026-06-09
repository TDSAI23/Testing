"""
Reference Channel Guidance (RCG) for ComfyUI
Variant of Reference Mean Guidance (arxiv 2605.10302) that operates
exclusively in the per-channel mean domain.

Motivation
----------
The original RMG implementation pulls the full spatial latent toward
reference images, which causes composition and object-shape leakage when
the reference bank contains different products/subjects than the generation.

RCG solves this by reducing both the noisy latent and the reference latents
to their per-channel spatial means before computing the guidance correction.
The correction is then applied as a spatially-uniform per-channel shift —
steering the tonal/stylistic channel distribution without touching spatial
composition, object placement, or geometry.

Mathematics
-----------
Standard RMG (eq. 8):
    mu_guided = (1 - beta) * mu_model + beta * mu_reference_spatial

RCG (this implementation):
    x_chan        = mean(x_t,    dims=[T,H,W])           [B, C]
    refs_chan[m]  = mean(refs[m], dims=[T,H,W])          [M, C]

    # Softmax weights from channel-mean distances (eq. 6, channel-only)
    dist[b,m] = ||x_chan[b] - t * refs_chan[m]||^2       [B, M]
    w[b,m]    = softmax_m(-dist / (2*(1-t)^2))           [B, M]

    # Weighted reference channel mean
    mu_ref_chan = sum_m(w[b,m] * refs_chan[m])            [B, C]

    # Spatially-uniform correction broadcast back to full latent shape
    delta       = mu_ref_chan - mean(mu_model, dims=[T,H,W])   [B, C]
    mu_guided   = mu_model + beta * delta.reshape(B,C,1,...,1)

Result: composition is entirely determined by the base model + conditioning.
Only the per-channel tonal/stylistic bias is steered toward the reference bank.

Confirmed working format for QIE 2511 / AuraFlow:
    VAE encode returns : [1, C, N, H, W]  (packed batch dim)
    x_t in wrapper     : [B, C, T, H, W]  (T=1 temporal dim)
"""

import math
import torch
import comfy.model_management


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def spatial_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Reduce all non-channel dims to produce per-channel mean.
    Works for any tensor shape [*, C, d1, d2, ..., dK] where
    dim 0 is batch and dim 1 is channels.

    Returns [B, C] float32.
    """
    spatial_dims = tuple(range(2, x.ndim))
    return x.float().mean(dim=spatial_dims)          # [B, C]


def compute_channel_weights(
    x_chan: torch.Tensor,       # [B, C]  channel mean of noisy latent
    refs_chan: torch.Tensor,    # [M, C]  channel mean of reference latents
    t: float,
) -> torch.Tensor:
    """
    Softmax weights over reference bank from eq. 6, evaluated in channel space.

    w[b,m] = softmax_m( -||x_chan[b] - t * refs_chan[m]||^2 / (2*(1-t)^2) )

    Returns [B, M] float32, summing to 1 over M.
    """
    one_minus_t = max(1.0 - t, 1e-4)

    # [B, 1, C] - [1, M, C]  →  [B, M, C]
    diff    = x_chan.unsqueeze(1) - t * refs_chan.unsqueeze(0)
    dist_sq = diff.pow(2).sum(dim=-1)                # [B, M]

    log_w   = -dist_sq / (2.0 * one_minus_t ** 2)
    return torch.softmax(log_w, dim=1)               # [B, M]


def channel_correction(
    mu_model: torch.Tensor,     # [B, C, ...]  endpoint mean from model
    refs_chan: torch.Tensor,    # [M, C]       channel means of reference bank
    t: float,
    beta: float,
) -> torch.Tensor:
    """
    Compute spatially-uniform per-channel correction delta and apply it.

    Returns mu_guided with same shape as mu_model.
    """
    B, C    = mu_model.shape[0], mu_model.shape[1]

    # Channel mean of current endpoint mean prediction
    mu_chan = spatial_mean(mu_model)                 # [B, C]

    # Weights from channel-mean similarity to references
    weights = compute_channel_weights(mu_chan, refs_chan, t)  # [B, M]

    # Weighted reference channel mean
    mu_ref_chan = torch.einsum('bm,mc->bc', weights, refs_chan)  # [B, C]

    # Channel delta: where does the reference bank want to push each channel?
    delta = mu_ref_chan - mu_chan                     # [B, C]

    # Reshape delta to broadcast over all spatial/temporal dims
    extra = mu_model.ndim - 2
    delta_full = delta.view(B, C, *([1] * extra))    # [B, C, 1, ..., 1]
    delta_full = delta_full.to(mu_model.dtype)

    return mu_model + beta * delta_full              # [B, C, ...]


# ---------------------------------------------------------------------------
# VAE encode helpers
# ---------------------------------------------------------------------------

def encode_references(vae, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Encode reference images and normalise to [M, C, ...] format.

    QIE 2511 VAE returns [1, C, N, H, W] for a batch of N images.
    Standard VAEs return [N, C, H, W].
    Both are handled and reshaped to [N, C, ...spatial...].
    """
    with torch.no_grad():
        latents = vae.encode(images[:, :, :, :3])

    if isinstance(latents, dict):
        latents = latents.get("samples", next(iter(latents.values())))

    latents = latents.to(device).float()

    # QIE packs N images into dim-2: [1, C, N, H, W] → [N, C, 1, H, W]
    if latents.ndim == 5 and latents.shape[0] == 1 and latents.shape[2] > 1:
        N = latents.shape[2]
        latents = latents[0].permute(1, 0, 2, 3).unsqueeze(2)  # [N, C, 1, H, W]

    print(f"[RCG] encoded {latents.shape[0]} references  shape={latents.shape}")
    return latents


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class RCGuidance:
    """
    Reference Channel Guidance.

    Steers the tonal/stylistic channel distribution of the generation toward
    a bank of reference style images, without leaking reference composition
    or spatial structure into the output.

    Inputs
    ------
    model            : MODEL  — flow-matching model to patch
    reference_images : IMAGE  — style reference bank [N, H, W, 3]
    vae              : VAE    — must be the same VAE used for generation
    beta             : FLOAT  — guidance strength (try 0.05–0.30)
    beta_schedule    : CHOICE — constant | cosine_decay
                                cosine_decay: full beta at high noise,
                                fades to zero at low noise
    t_min            : FLOAT  — ignore steps below this t (skip fine-detail)
    t_max            : FLOAT  — ignore steps above this t (skip layout steps)

    Recommended starting point
    --------------------------
    beta=0.10, cosine_decay, t_min=0.0, t_max=1.0

    Increase beta until style influence is visible; if composition starts
    drifting, lower t_max to 0.6 to exclude high-noise layout steps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("MODEL",),
                "reference_images": ("IMAGE",),
                "vae":              ("VAE",),
                "beta": ("FLOAT", {
                    "default": 0.10,
                    "min":     0.0,
                    "max":     1.0,
                    "step":    0.01,
                    "display": "slider",
                }),
                "beta_schedule": (["cosine_decay", "constant"],),
                "t_min": ("FLOAT", {
                    "default": 0.0,
                    "min":     0.0,
                    "max":     1.0,
                    "step":    0.05,
                    "display": "slider",
                }),
                "t_max": ("FLOAT", {
                    "default": 1.0,
                    "min":     0.0,
                    "max":     1.0,
                    "step":    0.05,
                    "display": "slider",
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION     = "patch"
    CATEGORY     = "advanced/guidance"

    def patch(self, model, reference_images, vae, beta, beta_schedule, t_min, t_max):
        device      = comfy.model_management.get_torch_device()
        ref_latents = encode_references(vae, reference_images, device)

        # Pre-compute reference channel means once — no per-step encode cost
        refs_chan = spatial_mean(ref_latents)         # [M, C]
        print(f"[RCG] refs_chan shape={refs_chan.shape}  (M references, C channels)")

        m = model.clone()

        def rcg_wrapper(apply_model_fn, args):
            x_t      = args["input"]
            timestep = args["timestep"]

            # sigma == t for AuraFlow (multiplier=1.0)
            t = timestep.flatten()[0].item() if isinstance(timestep, torch.Tensor) \
                else float(timestep)
            t = max(min(t, 1.0 - 1e-6), 1e-6)

            # Run base model first — correction is applied to its output
            v_model = apply_model_fn(args["input"], args["timestep"], **args["c"])

            # Gate: skip if outside active t window
            if t < t_min or t > t_max:
                return v_model

            # Effective beta at this timestep
            if beta_schedule == "cosine_decay":
                # Full strength at t=t_max, zero at t=t_min
                phase   = (t - t_min) / max(t_max - t_min, 1e-6)
                beta_t  = beta * math.cos(math.pi * 0.5 * (1.0 - phase))
            else:
                beta_t  = beta

            if beta_t <= 0.0:
                return v_model

            try:
                # Endpoint mean from model output
                # CONST formulation: x0_pred = x_t - v * sigma  (sigma = t)
                mu_model = x_t - v_model * t            # [B, C, ...]

                # Per-channel correction — no spatial reference leakage
                refs_c   = refs_chan.to(dtype=x_t.dtype, device=x_t.device)
                mu_guided = channel_correction(mu_model, refs_c, t, beta_t)

                # Convert corrected endpoint mean back to velocity
                # v = (x_t - x0_pred) / t
                return (x_t - mu_guided) / t

            except Exception as exc:
                print(f"[RCG] correction skipped at t={t:.3f}: {exc}")
                return v_model

        m.set_model_unet_function_wrapper(rcg_wrapper)
        return (m,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "RCGuidance": RCGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RCGuidance": "Reference Channel Guidance (RCG)",
}

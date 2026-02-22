"""
encoder.py — MEMORY LOGIC
Loads Wan 2.1 VAE, encodes video frames to latents, saves as .safetensors.
Sequential VRAM clearing between each video. Designed for Colab T4 GPU (16GB).
All outputs saved directly to Google Drive to preserve Colab RAM/disk.
"""

import os
import gc
import torch
from safetensors.torch import save_file

from core.video_tools import prepare_video_batch


def load_vae(model_id: str = "Wan-AI/Wan2.1-T2V-1.3B", device: str = "cuda"):
    """
    Load the Wan 2.1 AutoencoderKLWan from HuggingFace Hub or local cache.
    Uses float16 to halve VRAM. Downloads once, caches in models/ dir.

    Args:
        model_id: HuggingFace model ID or local path
        device: 'cuda' for GPU

    Returns:
        VAE model on device in eval mode
    """
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae = vae.to(device)
    vae.eval()
    return vae


def clear_vram():
    """Aggressively free VRAM between encoding steps."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def encode_frames_to_latent(vae, frames_tensor: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Encode a batch of video frames through the VAE encoder.

    Args:
        vae: Loaded AutoencoderKLWan
        frames_tensor: Shape (num_frames, 3, H, W), float16, range [-1, 1]
        device: Target device

    Returns:
        Latent tensor from VAE encoder
    """
    with torch.no_grad():
        # Add batch dim: (1, C, N, H, W) — diffusers video VAE expects (B, C, T, H, W)
        # Our frames are (N, C, H, W), rearrange to (1, C, N, H, W)
        video_input = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        latent_dist = vae.encode(video_input)
        latent = latent_dist.latent_dist.sample()

    return latent.cpu()


def cache_single_video(
    vae,
    frames_tensor: torch.Tensor,
    caption: str,
    video_name: str,
    output_dir: str,
    device: str = "cuda",
) -> dict:
    """
    Encode one video's frames and save as .safetensors to output_dir.
    The safetensors file contains the latent tensor.
    The caption is saved as a separate .txt alongside it.

    Args:
        vae: Loaded VAE
        frames_tensor: (N, C, H, W) float16 tensor
        caption: Caption text
        video_name: Stem name for output file
        output_dir: Where to save .safetensors file
        device: CUDA device

    Returns:
        Dict with file_path, latent_shape, size_mb
    """
    # Encode
    latent = encode_frames_to_latent(vae, frames_tensor, device)

    # Save latent as .safetensors
    os.makedirs(output_dir, exist_ok=True)
    safetensor_path = os.path.join(output_dir, f"{video_name}.safetensors")
    save_file({"latent": latent.squeeze(0).contiguous()}, safetensor_path)

    # Save caption alongside
    caption_path = os.path.join(output_dir, f"{video_name}.txt")
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption)

    size_mb = round(os.path.getsize(safetensor_path) / (1024 * 1024), 2)

    # VRAM cleanup after each video
    del latent, frames_tensor
    clear_vram()

    return {
        "file_path": safetensor_path,
        "latent_shape": "encoded",
        "size_mb": size_mb,
    }


def run_caching_pipeline(
    raw_dir: str,
    output_dir: str,
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    num_frames: int = 8,
    auto_sync: bool = False,
    drive_dest: str = "/content/drive/MyDrive/wan_latents",
    progress=None,
) -> str:
    """
    Main orchestrator: load VAE → iterate videos → encode → save .safetensors.
    Clears VRAM between each video. Optionally auto-syncs each file to Drive.

    Args:
        raw_dir: Path to data/raw/ with .mp4 + .txt pairs
        output_dir: Path to data/cached/ (or Drive path directly)
        model_id: HuggingFace model ID for Wan 2.1 VAE
        num_frames: Frames to extract per video
        auto_sync: If True, sync each .safetensors to Drive immediately
        drive_dest: Drive destination path for auto-sync
        progress: Gradio progress callback (optional)

    Returns:
        Final status log string
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines = []

    # ── Step 1: Load VAE ─────────────────────────────────────────────────
    log_lines.append("🔄 Loading Wan 2.1 VAE...")
    try:
        vae = load_vae(model_id, device)
        log_lines.append(f"✅ VAE loaded on {device} (float16)")
    except Exception as e:
        return f"❌ Failed to load VAE: {str(e)}"

    # ── Step 2: Prepare all video-caption pairs ──────────────────────────
    try:
        video_batches = list(prepare_video_batch(raw_dir, num_frames))
    except Exception as e:
        del vae
        clear_vram()
        return f"❌ Error scanning raw directory: {str(e)}"

    total = len(video_batches)
    log_lines.append(f"📁 Found {total} video-caption pairs")

    # ── Step 3: Encode each video ────────────────────────────────────────
    for i, (frames_tensor, caption, video_name) in enumerate(video_batches):
        step_label = f"[{i + 1}/{total}] {video_name}"

        if progress is not None:
            try:
                progress((i + 1) / total, desc=f"Encoding {video_name}...")
            except Exception:
                pass

        log_lines.append(f"⚡ {step_label} — encoding...")

        try:
            result = cache_single_video(vae, frames_tensor, caption, video_name, output_dir, device)
            log_lines.append(
                f"   ✅ Saved {video_name}.safetensors ({result['size_mb']} MB)"
            )

            # Auto-sync to Drive if enabled
            if auto_sync:
                from core.drive_sync import sync_single_file
                sync_msg = sync_single_file(result["file_path"], drive_dest)
                log_lines.append(f"   ☁️ {sync_msg}")

        except Exception as e:
            log_lines.append(f"   ❌ {step_label} failed: {str(e)}")
            clear_vram()

    # ── Cleanup ──────────────────────────────────────────────────────────
    del vae
    clear_vram()
    log_lines.append(f"\n🏁 Pipeline complete — {total} videos processed")

    return "\n".join(log_lines)

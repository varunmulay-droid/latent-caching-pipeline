"""
app.py — MASTER UI
Gradio-powered central interface for the WAN 2.1 Latent Caching Pipeline.
Designed for Google Colab GPU environment.

Tabs:
  1. 📂 Data Manager   — Upload & inspect raw video + caption pairs
  2. ⚡ Encode & Cache  — Run VAE encoding pipeline → .safetensors output
  3. ☁️ Drive Sync      — Mount Drive, sync checkpoints, auto-sync toggle

Run:
  python app.py
"""

import os
import sys
import glob
import warnings
import gradio as gr

# ── Path Setup (Colab-friendly) ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
CACHED_DIR = os.path.join(BASE_DIR, "data", "cached")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(CACHED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from core.video_tools import scan_raw_directory
from core.encoder import run_caching_pipeline
from core.drive_sync import (
    mount_drive,
    sync_to_drive,
    get_drive_status,
    get_sync_log,
    DEFAULT_DRIVE_PATH,
)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA MANAGER
# ═════════════════════════════════════════════════════════════════════════════

def refresh_data_table():
    """Scan data/raw/ and return a formatted status table."""
    pairs = scan_raw_directory(RAW_DIR)
    if not pairs:
        return "📭 No files found in data/raw/. Upload .mp4 and matching .txt caption files."

    rows = []
    for p in pairs:
        status = "✅ Paired" if p["has_caption"] else "⚠️ Missing caption"
        rows.append(f"| {p['video']} | {p['caption_file']} | {status} |")

    header = "| Video | Caption | Status |\n|-------|---------|--------|"
    return header + "\n" + "\n".join(rows)


def handle_upload(files):
    """Save uploaded files to data/raw/."""
    if not files:
        return "No files selected.", refresh_data_table()

    saved = []
    for file_obj in files:
        filename = os.path.basename(file_obj.name)
        dest = os.path.join(RAW_DIR, filename)
        with open(file_obj.name, "rb") as src:
            with open(dest, "wb") as dst:
                dst.write(src.read())
        saved.append(filename)

    return f"✅ Uploaded {len(saved)} file(s): {', '.join(saved)}", refresh_data_table()


def get_data_stats():
    """Return quick stats about data/raw/ and data/cached/."""
    mp4_count    = len(glob.glob(os.path.join(RAW_DIR, "*.mp4")))
    txt_count    = len(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    cached_count = len(glob.glob(os.path.join(CACHED_DIR, "*.safetensors")))

    total_cached_mb = 0
    for f in glob.glob(os.path.join(CACHED_DIR, "*.safetensors")):
        total_cached_mb += os.path.getsize(f) / (1024 * 1024)

    return (
        f"📹 **{mp4_count}** videos  •  📝 **{txt_count}** captions  •  "
        f"💾 **{cached_count}** cached latents ({total_cached_mb:.1f} MB)"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ENCODE & CACHE
# ═════════════════════════════════════════════════════════════════════════════

def run_encoding(model_id, num_frames, auto_sync_enabled, drive_path, progress=gr.Progress()):
    """Execute the full caching pipeline."""
    if not model_id.strip():
        model_id = "Wan-AI/Wan2.1-T2V-1.3B"

    log = run_caching_pipeline(
        raw_dir=RAW_DIR,
        output_dir=CACHED_DIR,
        model_id=model_id.strip(),
        num_frames=int(num_frames),
        auto_sync=auto_sync_enabled,
        drive_dest=drive_path.strip() if drive_path else DEFAULT_DRIVE_PATH,
        progress=progress,
    )
    return log, get_cached_files_display()


def get_cached_files_display():
    """List all cached .safetensors files with sizes."""
    files = sorted(glob.glob(os.path.join(CACHED_DIR, "*.safetensors")))
    if not files:
        return "No cached latents yet. Run the encoder first."

    rows = []
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f) / (1024 * 1024)
        rows.append(f"| {name} | {size:.2f} MB |")

    header = "| File | Size |\n|------|------|"
    return header + "\n" + "\n".join(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DRIVE SYNC
# ═════════════════════════════════════════════════════════════════════════════

def do_mount_drive():
    """Mount Google Drive."""
    msg    = mount_drive()
    status = format_drive_status(DEFAULT_DRIVE_PATH)
    return msg, status


def do_sync_now(drive_path):
    """Batch-sync all cached latents to Drive."""
    path   = drive_path.strip() if drive_path else DEFAULT_DRIVE_PATH
    msg    = sync_to_drive(CACHED_DIR, path)
    status = format_drive_status(path)
    log    = get_sync_log()
    return msg, status, log


def format_drive_status(drive_path: str) -> str:
    """Format Drive status for the UI."""
    info = get_drive_status(drive_path)
    return (
        f"🔗 **Mounted:** {'Yes ✅' if info['mounted'] else 'No ❌'}\n\n"
        f"📂 **Path:** `{info['dest_path']}`\n\n"
        f"📦 **Files on Drive:** {info['files_on_drive']}\n\n"
        f"💾 **Total size:** {info['total_size_mb']} MB\n\n"
        f"🕐 **Last sync:** {info['last_sync']}"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  THEME & CSS  (defined at module level so both build_ui and launch can use)
# ═════════════════════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    block_background_fill="#12121a",
    block_background_fill_dark="#12121a",
    block_border_color="#1e1e2e",
    block_border_color_dark="#1e1e2e",
    block_label_text_color="#a78bfa",
    block_label_text_color_dark="#a78bfa",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    body_text_color="#cbd5e1",
    body_text_color_dark="#cbd5e1",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_dark="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
    button_primary_background_fill_hover_dark="#6d28d9",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#1e1e2e",
    button_secondary_background_fill_dark="#1e1e2e",
    button_secondary_text_color="#a78bfa",
    button_secondary_text_color_dark="#a78bfa",
    input_background_fill="#1a1a2e",
    input_background_fill_dark="#1a1a2e",
    input_border_color="#2d2d44",
    input_border_color_dark="#2d2d44",
    shadow_drop="0 4px 14px rgba(124, 58, 237, 0.08)",
    shadow_drop_lg="0 8px 24px rgba(124, 58, 237, 0.12)",
)

CSS = """
.gradio-container {
    max-width: 960px !important;
    margin: auto;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.status-bar {
    padding: 10px 16px;
    background: linear-gradient(135deg, #1a1a2e, #16162a);
    border: 1px solid #2d2d44;
    border-radius: 8px;
    font-size: 0.9rem;
}
"""


# ═════════════════════════════════════════════════════════════════════════════
#  GRADIO UI ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def build_ui():
    """Construct the full Gradio Blocks application."""

    # Suppress the Gradio deprecation warnings for theme/css in gr.Blocks()
    # These are warnings only — the parameters still work correctly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        blocks = gr.Blocks(theme=THEME, css=CSS, title="WAN 2.1 Latent Cacher")

    with blocks:

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML("<div class='main-title'>⚡ WAN 2.1 Latent Cacher</div>")
        gr.HTML("<div class='sub-title'>Encode videos → safetensors latents → Google Drive</div>")

        gr.Markdown(value=get_data_stats, elem_classes=["status-bar"])

        # ── TAB 1: DATA MANAGER ──────────────────────────────────────────
        with gr.Tab("📂 Data Manager", id="data"):
            gr.Markdown(
                "### Upload & Inspect Raw Data\n"
                "Place `.mp4` + matching `.txt` caption files. "
                "Each video needs a caption with the same filename."
            )

            with gr.Row():
                upload_btn = gr.File(
                    label="Upload .mp4 / .txt files",
                    file_count="multiple",
                    file_types=[".mp4", ".txt"],
                    scale=2,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

            upload_status = gr.Markdown("")
            data_table    = gr.Markdown(value=refresh_data_table, label="Video-Caption Pairs")

            upload_btn.upload(
                fn=handle_upload,
                inputs=[upload_btn],
                outputs=[upload_status, data_table],
            )
            refresh_btn.click(
                fn=lambda: ("", refresh_data_table()),
                outputs=[upload_status, data_table],
            )

        # ── TAB 2: ENCODE & CACHE ────────────────────────────────────────
        with gr.Tab("⚡ Encode & Cache", id="encode"):
            gr.Markdown(
                "### VAE Encoding Pipeline\n"
                "Encodes raw videos through Wan 2.1 VAE and saves latents as `.safetensors`."
            )

            with gr.Row():
                model_input = gr.Textbox(
                    value="Wan-AI/Wan2.1-T2V-1.3B",
                    label="Model ID (HuggingFace)",
                    scale=3,
                )
                frames_slider = gr.Slider(
                    minimum=1,
                    maximum=24,
                    value=8,
                    step=1,
                    label="Frames per Video",
                    scale=1,
                )

            gr.Markdown("**Resolution:** 480 × 832 (Wan 2.1 native)")

            with gr.Row():
                auto_sync_toggle = gr.Checkbox(
                    label="☁️ Auto-sync each file to Drive after encoding",
                    value=False,
                )
                drive_path_encode = gr.Textbox(
                    value=DEFAULT_DRIVE_PATH,
                    label="Drive sync path",
                )

            start_btn = gr.Button("▶ Start Caching Pipeline", variant="primary", size="lg")

            encode_log = gr.Textbox(
                label="Pipeline Log",
                lines=15,
                interactive=False,
                show_copy_button=True,
            )
            cached_display = gr.Markdown(value=get_cached_files_display, label="Cached Latents")

            start_btn.click(
                fn=run_encoding,
                inputs=[model_input, frames_slider, auto_sync_toggle, drive_path_encode],
                outputs=[encode_log, cached_display],
            )

        # ── TAB 3: DRIVE SYNC ────────────────────────────────────────────
        with gr.Tab("☁️ Drive Sync", id="drive"):
            gr.Markdown(
                "### Google Drive Checkpoint Sync\n"
                "Mount Drive and sync cached `.safetensors` latents for persistent storage."
            )

            # gr.Markdown does not accept scale=; use gr.Column for proportional layout
            with gr.Row():
                with gr.Column(scale=1):
                    mount_btn = gr.Button("🔗 Mount Google Drive", variant="primary")
                with gr.Column(scale=2):
                    mount_status = gr.Markdown("Click Mount to connect Google Drive.")

            drive_path_input = gr.Textbox(
                value=DEFAULT_DRIVE_PATH,
                label="Drive Destination Path",
            )

            drive_status_display = gr.Markdown(
                value=lambda: format_drive_status(DEFAULT_DRIVE_PATH),
                label="Drive Status",
            )

            with gr.Row():
                sync_btn    = gr.Button("📤 Sync All to Drive", variant="primary")
                sync_result = gr.Markdown("")

            sync_log_display = gr.Textbox(
                value=get_sync_log,
                label="Sync Log",
                lines=8,
                interactive=False,
            )

            mount_btn.click(
                fn=do_mount_drive,
                outputs=[mount_status, drive_status_display],
            )
            sync_btn.click(
                fn=do_sync_now,
                inputs=[drive_path_input],
                outputs=[sync_result, drive_status_display, sync_log_display],
            )

        # ── FOOTER ───────────────────────────────────────────────────────
        gr.HTML(
            "<div style='text-align:center; color:#475569; font-size:0.8rem; margin-top:1rem;'>"
            "WAN 2.1 Latent Cacher Engine • Colab GPU Optimized • safetensors output"
            "</div>"
        )

    return blocks


# ═════════════════════════════════════════════════════════════════════════════
#  LAUNCH
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # Colab needs share=True for public URL
        show_error=True,
    )
def build_ui():
    """Construct the full Gradio Blocks application."""

    with gr.Blocks(title="WAN 2.1 Latent Cacher") as app:  # theme/css moved to launch()

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML("<div class='main-title'>⚡ WAN 2.1 Latent Cacher</div>")
        gr.HTML("<div class='sub-title'>Encode videos → safetensors latents → Google Drive</div>")

        stats_display = gr.Markdown(value=get_data_stats, elem_classes=["status-bar"])

        # ── TAB 1: DATA MANAGER ──────────────────────────────────────────
        with gr.Tab("📂 Data Manager", id="data"):
            gr.Markdown("### Upload & Inspect Raw Data\nPlace `.mp4` + matching `.txt` caption files. Each video needs a caption with the same filename.")

            with gr.Row():
                upload_btn = gr.File(
                    label="Upload .mp4 / .txt files",
                    file_count="multiple",
                    file_types=[".mp4", ".txt"],
                    scale=2,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

            upload_status = gr.Markdown("")
            data_table = gr.Markdown(value=refresh_data_table, label="Video-Caption Pairs")

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
            gr.Markdown("### VAE Encoding Pipeline\nEncodes raw videos through Wan 2.1 VAE and saves latents as `.safetensors`.")

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

            with gr.Row():
                gr.Markdown("**Resolution:** 480 × 832 (Wan 2.1 native)")

            with gr.Row():
                auto_sync_toggle = gr.Checkbox(
                    label="☁️ Auto-sync each file to Drive after encoding",
                    value=False,
                )
                drive_path_encode = gr.Textbox(
                    value=DEFAULT_DRIVE_PATH,
                    label="Drive sync path",
                    visible=True,
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
            gr.Markdown("### Google Drive Checkpoint Sync\nMount Drive and sync cached `.safetensors` latents for persistent storage.")

            with gr.Row():
                # ✅ FIX: Markdown doesn't support scale= ; wrap button+markdown in columns instead
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
                sync_btn = gr.Button("📤 Sync All to Drive", variant="primary")
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

    return app


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        theme=THEME,   # ✅ FIX: moved here from gr.Blocks()
        css=CSS,       # ✅ FIX: moved here from gr.Blocks()
    )
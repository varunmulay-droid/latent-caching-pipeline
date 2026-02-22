"""
drive_sync.py — GOOGLE DRIVE CHECKPOINT SYNC
Mounts Google Drive in Colab and syncs .safetensors latent caches to Drive
to free up Colab RAM/disk. All operations are Colab-native.
"""

import os
import shutil
import glob
import time
from datetime import datetime


# ── State ────────────────────────────────────────────────────────────────────
_drive_mounted = False
_last_sync_time = None
_sync_log = []

DEFAULT_DRIVE_PATH = "/content/drive/MyDrive/wan_latents"


def mount_drive() -> str:
    """
    Mount Google Drive in Colab environment.

    Returns:
        Status message string.
    """
    global _drive_mounted

    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        _drive_mounted = True
        return "✅ Google Drive mounted successfully at /content/drive"
    except ImportError:
        return "⚠️ Not running in Google Colab — Drive mount skipped. Use a local path instead."
    except Exception as e:
        _drive_mounted = False
        return f"❌ Drive mount failed: {str(e)}"


def is_drive_mounted() -> bool:
    """Check if Drive is currently mounted."""
    return _drive_mounted or os.path.isdir("/content/drive/MyDrive")


def ensure_drive_dir(drive_dest: str) -> str:
    """
    Create the destination directory on Drive if it doesn't exist.

    Args:
        drive_dest: Full path on mounted Drive

    Returns:
        Status message
    """
    try:
        os.makedirs(drive_dest, exist_ok=True)
        return f"✅ Drive directory ready: {drive_dest}"
    except Exception as e:
        return f"❌ Failed to create directory: {str(e)}"


def sync_to_drive(source_dir: str, drive_dest: str = DEFAULT_DRIVE_PATH) -> str:
    """
    Copy all .safetensors files from source_dir to Google Drive destination.
    Skips files that already exist on Drive with the same size.

    Args:
        source_dir: Local path containing .safetensors files (e.g. data/cached/)
        drive_dest: Destination path on mounted Drive

    Returns:
        Summary string of sync operation
    """
    global _last_sync_time, _sync_log

    if not os.path.isdir(source_dir):
        return f"❌ Source directory not found: {source_dir}"

    # Ensure destination exists
    dir_status = ensure_drive_dir(drive_dest)
    if "❌" in dir_status:
        return dir_status

    safetensor_files = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))

    if not safetensor_files:
        return "⚠️ No .safetensors files found to sync."

    synced = 0
    skipped = 0
    errors = []

    for src_file in safetensor_files:
        filename = os.path.basename(src_file)
        dest_file = os.path.join(drive_dest, filename)

        try:
            # Skip if already exists with same size
            if os.path.exists(dest_file):
                src_size = os.path.getsize(src_file)
                dst_size = os.path.getsize(dest_file)
                if src_size == dst_size:
                    skipped += 1
                    continue

            shutil.copy2(src_file, dest_file)
            synced += 1
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    _last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"✅ Synced {synced} files, skipped {skipped} unchanged"
    if errors:
        summary += f", ❌ {len(errors)} errors"

    _sync_log.append({"time": _last_sync_time, "synced": synced, "skipped": skipped, "errors": errors})

    return summary


def sync_single_file(file_path: str, drive_dest: str = DEFAULT_DRIVE_PATH) -> str:
    """
    Sync a single .safetensors file to Drive immediately after encoding.
    Used for auto-sync mode (sync after each video is cached).

    Args:
        file_path: Absolute path to the .safetensors file
        drive_dest: Destination directory on Drive

    Returns:
        Status message
    """
    global _last_sync_time, _sync_log

    if not os.path.isfile(file_path):
        return f"❌ File not found: {file_path}"

    ensure_drive_dir(drive_dest)
    filename = os.path.basename(file_path)
    dest_file = os.path.join(drive_dest, filename)

    try:
        shutil.copy2(file_path, dest_file)
        _last_sync_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _sync_log.append({"time": _last_sync_time, "synced": 1, "file": filename})
        return f"✅ Auto-synced {filename} → Drive"
    except Exception as e:
        return f"❌ Auto-sync failed for {filename}: {str(e)}"


def get_drive_status(drive_dest: str = DEFAULT_DRIVE_PATH) -> dict:
    """
    Get current Drive sync status for the UI.

    Returns:
        Dict with mounted, dest_path, last_sync, files_on_drive, total_size_mb
    """
    mounted = is_drive_mounted()
    files_on_drive = []
    total_size = 0

    if mounted and os.path.isdir(drive_dest):
        for f in sorted(glob.glob(os.path.join(drive_dest, "*.safetensors"))):
            size = os.path.getsize(f)
            total_size += size
            files_on_drive.append({
                "name": os.path.basename(f),
                "size_mb": round(size / (1024 * 1024), 2),
            })

    return {
        "mounted": mounted,
        "dest_path": drive_dest,
        "last_sync": _last_sync_time or "Never",
        "files_on_drive": len(files_on_drive),
        "file_list": files_on_drive,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }


def get_sync_log() -> str:
    """Return formatted sync log for the UI."""
    if not _sync_log:
        return "No sync operations yet."

    lines = []
    for entry in _sync_log[-20:]:  # Last 20 entries
        ts = entry.get("time", "?")
        synced = entry.get("synced", 0)
        fname = entry.get("file", "")
        if fname:
            lines.append(f"[{ts}] Auto-synced: {fname}")
        else:
            skipped = entry.get("skipped", 0)
            lines.append(f"[{ts}] Batch sync: {synced} synced, {skipped} skipped")

    return "\n".join(lines)

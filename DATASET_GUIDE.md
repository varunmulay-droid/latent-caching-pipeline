# 📦 Dataset Preparation Guide

> Step-by-step instructions for preparing your video + caption dataset for the WAN 2.1 Latent Cacher.

---

## Step 1 — Collect Your Source Videos

Record or download short video clips of the subject/style you want your LoRA to learn.

### Requirements

| Parameter | Requirement |
|-----------|-------------|
| **Format** | `.mp4` only |
| **Duration** | 2–10 seconds per clip (sweet spot: **3–5 sec**) |
| **FPS** | 24–30 fps recommended |
| **Resolution** | 720p or higher (auto-resized to 480 × 832 by the pipeline) |
| **Min frames** | At least 8 frames (any clip ≥ 1 second at 24fps works) |

### How many clips?

| Goal | Videos Needed |
|------|---------------|
| Quick pipeline test | 5 |
| Character / object LoRA | 10–15 |
| Style LoRA | 15–25 |
| Complex concept | 30–50 |

> **Quality > Quantity.** 10 well-curated clips beat 50 noisy ones.

---

## Step 2 — Trim & Clean Your Clips

If your source footage is long, trim it into focused 3–5 second clips.

### Using FFmpeg (free, works in Colab)

```bash
# Trim a clip from 00:12 to 00:16 (4 seconds)
ffmpeg -i raw_footage.mp4 -ss 00:00:12 -to 00:00:16 -c:v libx264 -c:a aac clip_01.mp4

# Batch trim multiple segments
ffmpeg -i raw_footage.mp4 -ss 00:00:05 -to 00:00:09 -c:v libx264 clip_01.mp4
ffmpeg -i raw_footage.mp4 -ss 00:00:22 -to 00:00:26 -c:v libx264 clip_02.mp4
ffmpeg -i raw_footage.mp4 -ss 00:01:03 -to 00:01:07 -c:v libx264 clip_03.mp4
```

### Checklist before proceeding

- [ ] Each clip is 2–10 seconds
- [ ] No black frames / blank segments
- [ ] Subject is clearly visible throughout
- [ ] No watermarks or text overlays (these will be learned by the LoRA)

---

## Step 3 — Name Your Files Consistently

Use **descriptive, lowercase, underscore-separated** filenames. Every video must have a matching caption file with the **exact same name**.

### ✅ Correct naming

```
cat_playing_ball.mp4
cat_playing_ball.txt

sunset_over_ocean.mp4
sunset_over_ocean.txt

robot_arm_welding.mp4
robot_arm_welding.txt
```

### ❌ Wrong naming (will fail)

```
Cat_Playing_Ball.mp4   +   cat_playing_ball.txt    ← Case mismatch
clip (1).mp4           +   clip (1).txt            ← Spaces & parens cause issues
video.mp4              +   caption.txt             ← Names don't match
```

---

## Step 4 — Write Your Captions

Create a `.txt` file for **every** `.mp4`. The caption tells the model what the video shows.

### Caption format

- Plain text, UTF-8 encoding
- 1–3 sentences
- Be **specific and descriptive**

### What to include in every caption

| Element | Example |
|---------|---------|
| **Subject** | "A fluffy orange tabby cat" |
| **Action** | "playing with a red ball" |
| **Environment** | "on a wooden floor in a living room" |
| **Lighting** | "warm natural sunlight from a window" |
| **Camera** | "shot from a low angle, slight dolly movement" |
| **Style** (optional) | "cinematic color grading, shallow depth of field" |

### Example captions

**`cat_playing_ball.txt`**
```
A fluffy orange tabby cat playing with a small red ball on a wooden floor.
Warm natural sunlight streams through a window, creating soft shadows.
Low angle shot with shallow depth of field.
```

**`sunset_over_ocean.txt`**
```
A golden sunset over a calm ocean, waves gently lapping at the shore.
Vibrant orange and purple sky reflected in the water.
Wide landscape shot, timelapse motion in the clouds.
```

**`robot_arm_welding.txt`**
```
An industrial robot arm welding metal parts, bright sparks flying.
Dark factory floor with dramatic contrast between shadows and welding light.
Medium close-up, steady tripod shot.
```

### Caption tips

| Do | Don't |
|----|-------|
| Describe what's **visible** | Describe emotions or abstract concepts |
| Include camera/lighting details | Write single-word labels ("cat") |
| Be consistent in style across captions | Mix wildly different caption formats |
| Use natural language | Use tags like "high quality, 4k, masterpiece" |

---

## Step 5 — Organize Into the Upload Folder

Place all paired files into a single folder:

```
my_dataset/
├── cat_playing_ball.mp4
├── cat_playing_ball.txt
├── cat_sleeping.mp4
├── cat_sleeping.txt
├── cat_jumping.mp4
├── cat_jumping.txt
├── cat_eating.mp4
├── cat_eating.txt
├── cat_running.mp4
├── cat_running.txt
├── ... (10-20 pairs)
```

### Verify your pairs

Run this quick check in Python or Colab to make sure everything matches:

```python
import os, glob

folder = "my_dataset"  # or wherever your files are
videos = sorted(glob.glob(os.path.join(folder, "*.mp4")))
missing = []

for v in videos:
    stem = os.path.splitext(os.path.basename(v))[0]
    txt = os.path.join(folder, f"{stem}.txt")
    if os.path.exists(txt):
        print(f"  ✅ {stem}.mp4 ↔ {stem}.txt")
    else:
        print(f"  ❌ {stem}.mp4 — MISSING CAPTION")
        missing.append(stem)

print(f"\nTotal: {len(videos)} videos, {len(missing)} missing captions")
```

---

## Step 6 — Upload to the Gradio UI

1. Open the **📂 Data Manager** tab in the WAN Latent Cacher
2. Click the **Upload** widget
3. Select **all** your `.mp4` and `.txt` files at once
4. Click **🔄 Refresh** to verify pairing status

You should see:

```
| Video               | Caption              | Status     |
|---------------------|----------------------|------------|
| cat_playing_ball.mp4| cat_playing_ball.txt | ✅ Paired  |
| cat_sleeping.mp4    | cat_sleeping.txt     | ✅ Paired  |
| cat_jumping.mp4     | cat_jumping.txt      | ✅ Paired  |
```

If any show `⚠️ Missing caption`, go back and fix the naming.

---

## Step 7 — Run the Caching Pipeline

Once all pairs show ✅:

1. Switch to the **⚡ Encode & Cache** tab
2. Keep defaults (Model: `Wan-AI/Wan2.1-T2V-1.3B`, Frames: `8`)
3. Enable **☁️ Auto-sync to Drive** if you want live backup
4. Click **▶ Start Caching Pipeline**

The pipeline will:
- Extract 8 evenly-spaced frames per video
- Center-crop and resize to 480 × 832
- Encode through Wan 2.1 VAE (float16)
- Save as `.safetensors` to `data/cached/`
- Auto-sync each file to Google Drive (if enabled)

---

## Quick Reference Card

```
DATASET CHECKLIST
─────────────────
□ 10-20 clips, each 3-5 seconds
□ All .mp4 format, 720p+, 24-30fps
□ Matching .txt caption for EVERY video
□ Filenames: lowercase, underscores, no spaces
□ Captions: 1-3 sentences describing subject, action, environment, lighting
□ No watermarks, black frames, or blurry footage
□ Verified all pairs match (✅ in Data Manager)
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "Missing caption for X" error | Create a `.txt` file with the exact same name as the `.mp4` |
| "Video has only N frames, need 8" | Your clip is too short — use a clip ≥ 1 second |
| Blurry latent outputs | Source video is too low-res — use 720p+ footage |
| LoRA overfits to training data | Add more variety — different angles, lighting, backgrounds |
| LoRA doesn't learn the concept | Captions are too vague — be more descriptive and consistent |
